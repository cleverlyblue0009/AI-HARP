"""Score every finished campaign checkpoint against the committed cells.

::

    D:/aiharp-env/python.exe -m experiments.evaluate_ablations            # all finished runs
    D:/aiharp-env/python.exe -m experiments.evaluate_ablations --only gcn mlp
    D:/aiharp-env/python.exe -m experiments.evaluate_ablations --list

Each ablation in ``checkpoints/campaign/<name>`` is the reference run with one
switch changed (experiments/campaign_train.py), so its evaluation must be the
reference's evaluation with nothing else changed: the same cells
(results/pareto_cells.json), seeds, biases, gate tau and deadline guard, and
the sampled policy the constraint trained. Output goes to
``results/agent_<name>/<policy-mode>/``.

Runs one checkpoint at a time in a subprocess (a fresh interpreter per run, so
no torch state carries over), skips runs whose evaluation already exists, and
skips runs that have not finished training (no run_summary.json). Safe to
re-run while the training queue is still working through the queue.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from common.config import PROJECT_ROOT

CAMPAIGN = PROJECT_ROOT / "checkpoints" / "campaign"


def finished_runs() -> list[str]:
    """Campaign runs whose training completed, in queue order."""
    from experiments.campaign_train import QUEUE

    return [n for n in QUEUE if (CAMPAIGN / n / "run_summary.json").exists()
            and (CAMPAIGN / n / "ckpt_final.pt").exists()]


def out_dir(name: str, policy_mode: str) -> Path:
    return PROJECT_ROOT / "results" / f"agent_{name}" / policy_mode


def is_scored(name: str, policy_mode: str) -> bool:
    return (out_dir(name, policy_mode) / "agent_evaluation.json").exists()


def evaluate(name: str, policy_mode: str, seeds: int, extra: list[str] | None = None) -> int:
    cmd = [sys.executable, "-m", "experiments.evaluate_agent",
           "--checkpoint", str(CAMPAIGN / name / "ckpt_final.pt"),
           "--seeds", str(seeds), "--policy-mode", policy_mode,
           "--out-dir", str(out_dir(name, policy_mode).relative_to(PROJECT_ROOT))] + (extra or [])
    log = CAMPAIGN / f"eval_{name}_{policy_mode}.log"
    with log.open("w", encoding="utf-8") as fh:
        return subprocess.run(cmd, cwd=PROJECT_ROOT, stdout=fh, stderr=subprocess.STDOUT).returncode


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--policy-mode", default="sampled", choices=["sampled", "argmax"])
    ap.add_argument("--skip-ref", action="store_true", help="the reference is already scored")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--force", action="store_true", help="re-score even if output exists")
    args, extra = ap.parse_known_args(argv)

    names = args.only or finished_runs()
    if args.skip_ref:
        names = [n for n in names if n != "ref"]
    if args.list:
        for n in names:
            print(f"{n:14s} trained={'yes' if (CAMPAIGN / n / 'run_summary.json').exists() else 'no':3s} "
                  f"scored={'yes' if is_scored(n, args.policy_mode) else 'no'}")
        return 0

    for n in names:
        if not (CAMPAIGN / n / "ckpt_final.pt").exists():
            print(f"[ablation-eval] {n}: not finished training, skipping", flush=True)
            continue
        if is_scored(n, args.policy_mode) and not args.force:
            print(f"[ablation-eval] {n}: already scored, skipping", flush=True)
            continue
        print(f"[ablation-eval] {n} ({args.policy_mode}) start", flush=True)
        t0 = time.time()
        rc = evaluate(n, args.policy_mode, args.seeds, extra)
        summary = out_dir(n, args.policy_mode) / "agent_evaluation.json"
        note = ""
        if summary.exists():
            js = json.loads(summary.read_text(encoding="utf-8"))
            note = f", updates={js.get('updates')}, tau={js.get('tau')}"
        print(f"[ablation-eval] {n}: exit {rc}, {(time.time() - t0) / 60:.1f} min{note}", flush=True)
        if rc != 0:
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
