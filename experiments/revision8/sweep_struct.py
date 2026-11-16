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
    ap.add_argument("--slice", default=None, help="i/n: only tasks with int(task_hash, 16) %% n == i (D14 runners)")
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
    if a.slice:
        si, sn = (int(x) for x in a.slice.split("/"))
        tasks = [t for t in tasks if int(t["task_hash"], 16) % sn == si]
    RUNS.mkdir(parents=True, exist_ok=True)
    name = f"struct_{a.arm}__{a.seedset}__headline__default__fallback" + (f"__slice{si}of{sn}" if a.slice else "")
    meta = RUNS / f"{name}_meta.json"
    stamps = json.loads(meta.read_text()) if meta.exists() else []
    stamps.append(stamp(f"revision8/{name}", {"arm": a.arm, "runs": list(th), "deltas": DELTAS}, seeds=sd,
                        n_runs=len(tasks)))
    write_json(meta, stamps)
    rows = run_tasks(tasks, RUNS / f"{name}.csv", jobs=a.jobs, log_every=500)
    print(f"[revision8] {name}: {len(rows)} / {len(tasks)} rows")
    return 0


def merge(arm: str, seedset: str) -> int:
    """Concatenate the slice files of one (arm, seed set) into the canonical file; every task exactly once."""
    import csv
    base = f"struct_{arm}__{seedset}__headline__default__fallback"
    parts = sorted(RUNS.glob(f"{base}__slice*of*.csv"))
    rows, header = {}, None
    for p in parts:
        with open(p, newline="") as f:
            r = csv.DictReader(f)
            header = header or r.fieldnames
            for row in r:
                if row["task_hash"] in rows and rows[row["task_hash"]] != row:
                    raise SystemExit(f"{p.name}: task {row['task_hash']} differs between slices")
                rows[row["task_hash"]] = row
    expected = len(thetas(arm)) * len(DELTAS) * len(HEADLINE) * len(seeds(seedset))
    if len(rows) != expected:
        raise SystemExit(f"{base}: {len(rows)} rows merged from {len(parts)} slices, expected {expected}")
    with open(RUNS / f"{base}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for k in sorted(rows):
            w.writerow(rows[k])
    print(f"[revision8] merged {len(parts)} slices -> {base}.csv ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    import sys as _s
    if len(_s.argv) > 1 and _s.argv[1] == "merge":
        raise SystemExit(merge(_s.argv[2], _s.argv[3]))
    raise SystemExit(main())
