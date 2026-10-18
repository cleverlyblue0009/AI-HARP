"""R8 rung B/C training targets: 0.95 x best of flooding / DV-CAST in ORACLE RWCR.

::

    python -m experiments.revision3.oracle_targets --jobs 8

Same cells and seeds as experiments/coverage_targets.py (every training scenario x
curriculum density x training weather x training hazard; training seeds 100-131), same
two reference policies at their registry defaults, but the ceiling is measured in ORACLE
RWCR (what the evaluation scores) instead of causal coverage. Each run also records causal
coverage, which gives rung A's surrogate check (causal coverage vs oracle RWCR).

Writes results/revision3/R8/oracle_coverage_targets.json in the CoverageTargets format
({"provenance": ..., "cells": {"scenario|density|weather|hazard": {"ceiling", "per_policy",
"seeds"}}}) and the per-run rows to results/revision3/R8/oracle_target_runs.csv.
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np

from agents.constrained_reward import curriculum_all_densities
from common.config import load_yaml
from experiments.revision1.provenance import out_dir, stamp, write_json
from experiments.revision1.runner import default_jobs, run_tasks, task

POLICIES = ("flooding", "dvcast")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=default_jobs())
    args = ap.parse_args(argv)
    cfg = load_yaml("agent.yaml")
    t = cfg["training"]
    seeds = list(range(int(t["train_seed_pool"]["start"]),
                       int(t["train_seed_pool"]["start"]) + int(cfg["objective"]["target_seeds"])))
    cells = [(sc, float(d), w, h) for sc in t["train_scenarios"] for d in curriculum_all_densities(t)
             for w in t["train_weather"] for h in t["train_hazards"]]
    tasks = [task(sc, d, p, s, weather=w, hazard_type=h, causal=True)
             for sc, d, w, h in cells for p in POLICIES for s in seeds]
    d = out_dir("R8", 3)
    rows = run_tasks(tasks, d / "oracle_target_runs.csv", jobs=args.jobs)
    acc = defaultdict(lambda: defaultdict(list))
    for r in rows:
        acc[f"{r['scenario']}|{float(r['density']):g}|{r['weather']}|{r['hazard_type']}"][r["policy"]].append(float(r["rwcr"]))
    out = {"provenance": {"reference_policies": list(POLICIES), "coverage": "oracle_rwcr",
                          "n_seeds": len(seeds), "seeds": seeds,
                          "note": "R8 rung B/C targets; same cells/seeds as coverage_targets.json"},
           "cells": {}}
    for k, pp in acc.items():
        per = {p: float(np.nanmean(v)) for p, v in pp.items()}
        out["cells"][k] = {"ceiling": max(per.values()), "per_policy": per, "seeds": seeds}
    out["provenance"]["stamp"] = stamp("R8/oracle_targets", {"cells": len(cells), "policies": POLICIES},
                                       seeds=seeds)
    write_json(d / "oracle_coverage_targets.json", out)
    print(f"{len(out['cells'])} cells; {len(rows)} runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
