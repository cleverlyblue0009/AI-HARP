"""E1: headline numbers from a fresh clone (run INSIDE the clone).

::

    python -m experiments.revision1.e1_headline --jobs 3

Re-simulates, with this checkout's code and an empty trace cache, the setting
behind every headline reference point (per-cell oracle k* and the best fixed
setting, 4 cells x 10 seeds) and the two agent points behind 0.717 (rural d=20,
bias -2 and +1, tau=0.5, sampled; needs checkpoints/campaign/ref/ckpt_final.pt,
which is git-ignored and must be copied in). Compares every mean with the
committed results/pareto_cells.json and results/agent_ref/sampled/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from analysis.comparator import load_cells, per_cell_oracle_best, best_fixed_baseline
from common.config import PROJECT_ROOT
from experiments.revision1.provenance import out_dir, stamp, write_json
from experiments.revision1.runner import aggregate, run_tasks, task

R = PROJECT_ROOT / "results"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=3)
    args = ap.parse_args(argv)
    cells = load_cells()
    oracle = per_cell_oracle_best(cells)
    fixed = best_fixed_baseline(cells)
    targets = []
    for c in cells:
        tq, mb = c.resolve_target(0.95), oracle.miss_bound[c.key]
        pol = oracle.label(c.key)
        best = min([p for p in c.curves[pol].points if p.quality >= tq and c.within_miss_bound(p, mb)],
                   key=lambda p: p.cost)
        for p in (best, c.find(fixed.policy, fixed.param, fixed.value)):
            targets.append((c, p))
    tasks, groups = [], []
    for c, p in targets:
        ts = [task(c.key.scenario, c.key.density, p.policy, s, param=p.param, value=p.value)
              for s in range(10)]
        tasks += ts
        groups.append((str(c.key), p, {t["task_hash"] for t in ts}))
    ck = PROJECT_ROOT / "checkpoints" / "campaign" / "ref" / "ckpt_final.pt"
    agent_ref = {p.value: p for p in next(c for c in load_cells(R / "agent_ref" / "sampled" / "pareto_cells_agent.json")
                                          if c.key.density == 20 and c.key.scenario == "rural_highway").curves["ai_harp"].points}
    if ck.exists():
        from experiments.evaluate_agent import agent_params

        for b in (-2.0, 1.0):
            ts = [task("rural_highway", 20.0, "ai_harp", s, param="suppression_bias", value=b,
                       fixed=agent_params(ck, tau=0.5, deterministic=False)) for s in range(10)]
            tasks += ts
            groups.append(("rural_highway/d=20 agent", agent_ref[b], {t["task_hash"] for t in ts}))
    rows = run_tasks(tasks, out_dir("E1") / "e1_clone_runs.csv", jobs=args.jobs)
    out = []
    for name, ref, hashes in groups:
        mine = aggregate([r for r in rows if r["task_hash"] in hashes])
        out.append({"cell": name, "setting": ref.label, "committed_rwcr": ref.quality,
                    "rerun_rwcr": mine.quality, "committed_cost": ref.cost, "rerun_cost": mine.cost,
                    "max_abs_diff": max(abs(mine.quality - ref.quality), abs(mine.cost - ref.cost))})
    res = {"points": out, "max_abs_diff": max(o["max_abs_diff"] for o in out),
           "agent_checked": ck.exists(),
           "provenance": stamp("E1", {"what": "headline reference settings + agent anchors"},
                               seeds=list(range(10)))}
    write_json(out_dir("E1") / "e1_headline.json", res)
    for o in out:
        print(f"{o['cell']:38s} {o['setting']:36s} rwcr {o['rerun_rwcr']:.6f} vs {o['committed_rwcr']:.6f}  "
              f"cost {o['rerun_cost']:.6f} vs {o['committed_cost']:.6f}")
    print("max |diff|:", res["max_abs_diff"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
