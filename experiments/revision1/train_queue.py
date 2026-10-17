"""C1/C2/C5 training queue: extra training seeds for the full model and ablations.

::

    D:/aiharp-gpu/Scripts/python.exe -m experiments.revision1.train_queue --list
    D:/aiharp-gpu/Scripts/python.exe -m experiments.revision1.train_queue --lane 0 --lanes 2 --workers 5

Each run is ``agents.train`` exactly as experiments/campaign_train.py runs it
(staged curriculum, 1,000 updates, deterministic GPU PPO), differing only in
``--seed`` (training seed: network init and episode draws). Seed 0 of every
variant is the committed campaign run (checkpoints/campaign/<variant>), so seeds
1-4 are trained here. ``ref_s0_u2000`` (C5) copies checkpoints/campaign/ref and
resumes it exactly with the finetune cap raised from 800 to 1,800 (200 + 1,800 =
2,000 updates); finetune sampling does not depend on progress, so this equals a
fresh 2,000-update run of seed 0. The original campaign directory is untouched.

Output: checkpoints/revision1/<run>/ (git-ignored) and, committed,
results/revision1/C_train/<run>.json (seed, command, commit, config hash, summary).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from common.config import PROJECT_ROOT, config_hash
from experiments.revision1.provenance import git_commit, out_dir, rel, write_json

ROOT = PROJECT_ROOT / "checkpoints" / "revision1"
VARIANTS = {"ref": [], "gcn": ["--encoder", "gcn"], "mlp": ["--encoder", "mlp"],
            "star": ["--star-graph"],
            # revision-2 R7: causal-relevance node feature zeroed (train and eval:
            # graph switches travel in the checkpoint config). Seed 0 is the
            # committed checkpoints/campaign/no_relevance.
            "norel": ["--drop-node-feature", "relevance_causal"]}
SEEDS = (1, 2)


#: Revised plan (user, 2026-10-16): C1 = 2 extra full-model seeds (3 in total
#: with the committed seed 0); then C2 = 1 extra seed per ablation (2 in total);
#: C5 (the 2,000-update run) skipped. Lane order: lane 0 ref_s1, gcn_s1, star_s1;
#: lane 1 ref_s2, mlp_s1.
#: Revision-2 (user, 2026-10-17): R6 = full-model seeds 3 and 4; R7 = no-relevance seed 1.
PLAN: tuple[tuple[str, int], ...] = (("ref", 1), ("ref", 2), ("gcn", 1), ("mlp", 1),
                                     ("star", 1), ("ref", 3), ("ref", 4), ("norel", 1))


def queue() -> list[tuple[str, list[str], dict]]:
    """(name, train flags, info); the full model's seeds first, then ablations."""
    return [(f"{v}_s{s}", ["--seed", str(s)] + VARIANTS[v], {"variant": v, "seed": s})
            for v, s in PLAN]


def live(name: str, stale_s: float = 900.0) -> bool:
    """A run another process is still training: no summary, checkpoint fresh."""
    d = ROOT / name
    ck = d / "ckpt_latest.pt"
    return (not (d / "run_summary.json").exists() and ck.exists()
            and time.time() - ck.stat().st_mtime < stale_s)


def status(name: str) -> str:
    d = ROOT / name
    if (d / "run_summary.json").exists() and (name != "ref_s0_u2000" or _extended(d)):
        return "done"
    if (d / "ckpt_latest.pt").exists():
        return "partial"
    return "pending"


def _extended(d: Path) -> bool:
    try:
        return int(json.loads((d / "run_summary.json").read_text())["total_updates"]) > 1000
    except Exception:
        return False


def command(name: str, flags: list[str], workers: str) -> list[str]:
    d = ROOT / name
    base = [sys.executable, "-u", "-m", "agents.train", "--keep-awake", "--quiet"]
    if name == "ref_s0_u2000":
        if not (d / "ckpt_latest.pt").exists():
            d.mkdir(parents=True, exist_ok=True)
            src = PROJECT_ROOT / "checkpoints" / "campaign" / "ref"
            for f in ("ckpt_latest.pt", "history.jsonl"):
                shutil.copy2(src / f, d / f)
        return base + ["--resume", str(d / "ckpt_latest.pt"), "--out", str(d),
                       "--workers", workers] + flags
    if status(name) == "partial":
        return base + ["--resume", str(d / "ckpt_latest.pt"), "--workers", workers]
    return base + ["--out", str(d), "--workers", workers] + flags


def record(name: str, info: dict, cmd: list[str], hours: float, rc: int) -> None:
    import torch

    d = ROOT / name
    meta: dict = {"run": name, **info, "rc": rc, "hours_this_session": round(hours, 3),
                  "git_commit": git_commit(),
                  "command": " ".join(["python"] + cmd[1:]).replace(str(PROJECT_ROOT), "."),
                  "checkpoint_dir": rel(d)}
    ck = d / "ckpt_final.pt"
    if ck.exists():
        st = torch.load(str(ck), map_location="cpu", weights_only=False)
        meta["config_hash"] = config_hash(st["config"])
        meta["training_seed"] = int(st["config"]["training"]["seed"])
        meta["encoder"] = st["config"]["encoder"]["type"]
        meta["include_neighbour_edges"] = st["config"]["graph"].get("include_neighbour_edges", True)
        meta["final_update"] = int(st["update"])
        from experiments.evaluate_agent import checkpoint_sha

        meta["checkpoint_sha"] = checkpoint_sha(ck)
    if (d / "run_summary.json").exists():
        meta["run_summary"] = json.loads((d / "run_summary.json").read_text())
    write_json(out_dir("C_train") / f"{name}.json", meta)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lane", type=int, default=0)
    ap.add_argument("--lanes", type=int, default=1)
    ap.add_argument("--workers", default="5")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--wait-for", nargs="*", default=[],
                    help="start only once these runs are done (a lane for later runs)")
    ap.add_argument("--eval-jobs", type=int, default=2,
                    help="parallel jobs for the final-checkpoint evaluation launched "
                         "(in the background) after each run; 0 disables it")
    args = ap.parse_args(argv)
    while any(status(n) != "done" for n in args.wait_for):
        time.sleep(120)
    q = [e for e in queue() if args.only is None or e[0] in args.only]
    if args.list:
        for i, (n, f, _) in enumerate(q):
            print(f"lane {i % args.lanes}  {n:14s} {status(n):8s} {' '.join(f)}")
        return 0
    ROOT.mkdir(parents=True, exist_ok=True)
    for i, (name, flags, info) in enumerate(q):
        if i % args.lanes != args.lane or status(name) == "done":
            continue
        t0 = time.time()
        if live(name):
            # Adopt a run started by an earlier queue process instead of
            # launching a second copy on the same directory.
            print(f"[queue] {name}: already training elsewhere; waiting for it", flush=True)
            while live(name):
                time.sleep(60)
            cmd = ["(adopted)", "-m", "agents.train", "--out", str(ROOT / name)] + flags
            rc = 0 if (ROOT / name / "run_summary.json").exists() else 1
            if rc:
                print(f"[queue] {name}: adopted run went stale without a summary; resuming",
                      flush=True)
        if not live(name) and status(name) != "done":
            cmd = command(name, flags, args.workers)
            log = ROOT / f"{name}.log"
            print(f"[queue] {name}: {' '.join(cmd)}", flush=True)
            with log.open("a", encoding="utf-8") as fh:
                rc = subprocess.run(cmd, cwd=PROJECT_ROOT, stdout=fh,
                                    stderr=subprocess.STDOUT).returncode
        record(name, info, cmd, (time.time() - t0) / 3600, rc)
        print(f"[queue] {name}: exit {rc}, {(time.time() - t0) / 3600:.2f} h", flush=True)
        ck = ROOT / name / "ckpt_final.pt"
        if args.eval_jobs and status(name) == "done" and ck.exists():
            # Final checkpoint only (user decision), headline cells, both gate
            # arms, sampled; in the background so the lane's next run starts now.
            ev = [sys.executable, "-u", "-m", "experiments.revision1.agent_eval",
                  "--checkpoint", str(ck), "--name", name, "--taus", "0.5", "0",
                  "--modes", "sampled", "--jobs", str(args.eval_jobs)]
            elog = (PROJECT_ROOT / "logs" / "revision1" / f"eval_{name}.log").open("a", encoding="utf-8")
            subprocess.Popen(ev, cwd=PROJECT_ROOT, stdout=elog, stderr=subprocess.STDOUT)
            print(f"[queue] {name}: evaluation launched -> logs/revision1/eval_{name}.log", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
