"""Revision 9 S2 (descriptive): custom-simulator rows for the STRUCT9 runs (results/revision9/train/<arm>_s{0..4}),
exactly as revision-8 sweep_struct (delta -2..3, four headline cells, SEL 400-419, EVAL 200-229, fallback mobility,
default channel, every randomisation flag off), split into slices by task hash for runners and merged afterwards.

    python -m experiments.revision9.sweep9 --arm plain --seedset SEL --slice 3/20 --jobs 4
    python -m experiments.revision9.sweep9 merge plain SEL
"""

from __future__ import annotations

import argparse
import csv
import json
import sys

from common.config import PROJECT_ROOT
from experiments.revision1.provenance import stamp, write_json
from experiments.revision1.runner import run_tasks, task
from experiments.revision5.sweep import HEADLINE, seeds

DELTAS = (-2.0, -1.0, 0.0, 1.0, 2.0, 3.0)
TRAIN = PROJECT_ROOT / "results" / "revision9" / "train"
RUNS = PROJECT_ROOT / "results" / "revision9" / "runs"
K = range(5)


def thetas(arm: str) -> dict[str, list[float]]:
    out = {}
    for k in K:
        d = json.loads((TRAIN / f"{arm}_s{k}" / "theta.json").read_text())
        if d["generations"] != 80:
            raise SystemExit(f"{arm}_s{k}: {d['generations']} generations, pre-registered 80")
        out[f"{arm}_s{k}"] = d["theta"]
    return out


def base(arm: str, ss: str) -> str:
    return f"struct9_{arm}__{ss}__headline__default__fallback"


def main(argv=None) -> int:
    import numpy
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", required=True, choices=["plain", "rand"])
    ap.add_argument("--seedset", required=True, choices=["SEL", "EVAL"])
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--slice", default=None)
    a = ap.parse_args(argv)
    print(f"[env] {sys.executable} numpy {numpy.__version__}", flush=True)
    th, sd = thetas(a.arm), seeds(a.seedset)
    tasks = [task(sc, d, "struct", s, param="delta", value=dl, fixed={"theta": t}, backend="fallback")
             for sc, d in HEADLINE for run, t in th.items() for s in sd for dl in DELTAS]
    name = base(a.arm, a.seedset)
    if a.slice:
        si, sn = (int(x) for x in a.slice.split("/"))
        tasks = [t for t in tasks if int(t["task_hash"], 16) % sn == si]
        name += f"__slice{si}of{sn}"
    RUNS.mkdir(parents=True, exist_ok=True)
    write_json(RUNS / f"{name}_meta.json", [stamp(f"revision9/{name}", {"arm": a.arm, "runs": list(th)}, seeds=sd,
                                                  n_runs=len(tasks))])
    rows = run_tasks(tasks, RUNS / f"{name}.csv", jobs=a.jobs, log_every=500)
    print(f"[revision9] {name}: {len(rows)} / {len(tasks)} rows")
    return 0


def merge(arm: str, ss: str) -> int:
    b = base(arm, ss)
    rows, header = {}, None
    for p in sorted(RUNS.glob(f"{b}__slice*of*.csv")):
        with open(p, newline="") as f:
            r = csv.DictReader(f)
            header = header or r.fieldnames
            for row in r:
                if row["task_hash"] in rows and rows[row["task_hash"]] != row:
                    raise SystemExit(f"{p.name}: task {row['task_hash']} differs between slices")
                rows[row["task_hash"]] = row
    expected = len(K) * len(DELTAS) * len(HEADLINE) * len(seeds(ss))
    if len(rows) != expected:
        raise SystemExit(f"{b}: {len(rows)} rows, expected {expected}")
    with open(RUNS / f"{b}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for k in sorted(rows):
            w.writerow(rows[k])
    print(f"[revision9] merged -> {b}.csv ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "merge":
        raise SystemExit(merge(sys.argv[2], sys.argv[3]))
    raise SystemExit(main())
