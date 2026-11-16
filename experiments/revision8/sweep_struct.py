"""Revision 8 (ADDENDUM section 5): custom-simulator rows for the trained STRUCT arms.

    python -m experiments.revision8.sweep_struct --arm plain --seedset SEL  --jobs 4
    python -m experiments.revision8.sweep_struct --arm plain --seedset EVAL --jobs 4     # Tier A + Tier B (200-229)

Each trained run (results/revision8/train/<arm>_s<k>/theta.json, k = 0..9) x delta in {-2..3} x the four headline cells x
the seed set; fallback mobility, default channel, epoch mode, every randomisation flag off. One
experiments.revision1.runner row per run, resumable (finished task hashes are skipped). Output:
results/revision8/runs/struct_<arm>__<SEL|EVAL>__headline__default__fallback.csv. Refuses to start unless all ten
runs have finished their 80 generations (no partial-budget evaluation).
"""

from __future__ import annotations

import argparse
import json
import sys

from common.config import PROJECT_ROOT
from experiments.revision1.provenance import stamp, write_json
from experiments.revision1.runner import run_tasks, task
from experiments.revision5.sweep import HEADLINE, seeds

DELTAS = (-2.0, -1.0, 0.0, 1.0, 2.0, 3.0)
TRAIN = PROJECT_ROOT / "results" / "revision8" / "train"
RUNS = PROJECT_ROOT / "results" / "revision8" / "runs"


def thetas(arm: str, train_seeds=range(10)) -> dict[str, list[float]]:
    out = {}
    for k in train_seeds:
        p = TRAIN / f"{arm}_s{k}" / "theta.json"
        if not p.exists():
            raise SystemExit(f"{p} missing: run {arm}_s{k} has not finished its budget; no partial evaluation")
        d = json.loads(p.read_text())
        if d["generations"] != 80:
            raise SystemExit(f"{p}: {d['generations']} generations, pre-registered 80")
        out[f"{arm}_s{k}"] = d["theta"]
    return out


def main(argv=None) -> int:
    import numpy
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", required=True, choices=["plain", "rand"])
    ap.add_argument("--seedset", required=True, choices=["SEL", "EVAL"])
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args(argv)
    print(f"[env] {sys.executable} numpy {numpy.__version__}", flush=True)
    th = thetas(a.arm)
    sd = seeds(a.seedset)
    tasks = []
    for sc, d in HEADLINE:                      # cell-major
        for run, theta in th.items():
            for s in sd:
                for dl in DELTAS:
                    tasks.append(task(sc, d, "struct", s, param="delta", value=dl, fixed={"theta": theta},
                                      backend="fallback"))
    RUNS.mkdir(parents=True, exist_ok=True)
    name = f"struct_{a.arm}__{a.seedset}__headline__default__fallback"
    meta = RUNS / f"{name}_meta.json"
    stamps = json.loads(meta.read_text()) if meta.exists() else []
    stamps.append(stamp(f"revision8/{name}", {"arm": a.arm, "runs": list(th), "deltas": DELTAS}, seeds=sd,
                        n_runs=len(tasks)))
    write_json(meta, stamps)
    rows = run_tasks(tasks, RUNS / f"{name}.csv", jobs=a.jobs, log_every=500)
    print(f"[revision8] {name}: {len(rows)} / {len(tasks)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
