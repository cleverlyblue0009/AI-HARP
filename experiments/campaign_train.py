"""Train the reference agent and every retrained ablation, one after another.

::

    D:/aiharp-gpu/Scripts/python.exe -m experiments.campaign_train            # run / resume all
    D:/aiharp-gpu/Scripts/python.exe -m experiments.campaign_train --list     # show the queue
    D:/aiharp-gpu/Scripts/python.exe -m experiments.campaign_train --only ref gcn

Every run uses the staged curriculum of configs/agent.yaml (as of the user's
post-feasibility decisions: coverage targets on all 32 training seeds, lambda
cap 500) and the same training seed, so each ablation differs from the
reference in exactly the switch it names. Runs are sequential (one GPU, one
rollout pool). A run with run_summary.json is skipped; one with
ckpt_latest.pt is resumed exactly (agents/train.py --resume). Output goes to
checkpoints/campaign/<name>/, a log per run alongside.

"10 seeds each" in the campaign brief is the evaluation seeds (0-9), paired
across ablations. Retraining each ablation over 10 training seeds would be
~10 x 3.2 h x 10 runs; single-training-seed variance is a stated limitation.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from common.config import PROJECT_ROOT

#: name -> extra agents.train flags. Order is run order: the reference first.
QUEUE: dict[str, list[str]] = {
    "ref": [],
    "gcn": ["--encoder", "gcn"],
    "mlp": ["--encoder", "mlp"],
    "star": ["--star-graph"],
    "heads1": ["--heads", "1"],
    "heads8": ["--heads", "8"],
    "k4": ["--neighbour-cap", "4"],
    # "k20": ["--neighbour-cap", "20"] -- dropped on cost (user decision).
    # Measured 35.6 s per update against 4.9-8.6 s for every other run: raising
    # the cap from 12 to 20 roughly triples the neighbour-to-neighbour edges and
    # the attention over them, so its 1,000 updates cost ~9 h against ~2 h. The
    # cap axis keeps two points, k4 = 4 against the reference's 12.
    "no_relevance": ["--drop-node-feature", "relevance_causal"],
    # Every baseline meeting the urban d=2 training target waits up to 20-50
    # slots (1 + slot epochs); defer_3 = 3 epochs cannot express that.
    "long_wait": ["--defer-epochs", "5", "20", "50"],
}

ROOT = PROJECT_ROOT / "checkpoints" / "campaign"


def status(name: str) -> str:
    d = ROOT / name
    if (d / "run_summary.json").exists():
        return "done"
    if (d / "ckpt_latest.pt").exists():
        return "partial"
    return "pending"


def command(name: str) -> list[str]:
    d = ROOT / name
    base = [sys.executable, "-u", "-m", "agents.train", "--keep-awake"]
    if status(name) == "partial":
        return base + ["--resume", str(d / "ckpt_latest.pt")]
    return base + ["--out", str(d)] + QUEUE[name]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", nargs="*", default=None, help="run only these queue entries")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args(argv)
    names = args.only or list(QUEUE)
    unknown = [n for n in names if n not in QUEUE]
    if unknown:
        raise SystemExit(f"unknown runs {unknown}; queue: {list(QUEUE)}")

    ROOT.mkdir(parents=True, exist_ok=True)
    if args.list:
        for n in names:
            print(f"{n:14s} {status(n):8s} {' '.join(QUEUE[n]) or '(reference)'}")
        return 0

    for n in names:
        if status(n) == "done":
            print(f"[campaign] {n}: done, skipping", flush=True)
            continue
        cmd = command(n)
        (ROOT / n).mkdir(parents=True, exist_ok=True)
        log = ROOT / f"{n}.log"
        print(f"[campaign] {n}: {' '.join(cmd)}", flush=True)
        t0 = time.time()
        with log.open("a", encoding="utf-8") as fh:
            rc = subprocess.run(cmd, cwd=PROJECT_ROOT, stdout=fh, stderr=subprocess.STDOUT).returncode
        summary = ROOT / n / "run_summary.json"
        stopped = json.loads(summary.read_text())["stopped"] if summary.exists() else None
        print(f"[campaign] {n}: exit {rc}, {stopped}, {(time.time() - t0) / 3600:.2f} h", flush=True)
        if rc != 0:
            print(f"[campaign] stopping the queue: {n} failed; see {log}", flush=True)
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
