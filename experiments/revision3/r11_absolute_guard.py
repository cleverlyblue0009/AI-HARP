"""R11: the absolute actionable-miss level Algorithm 1's guard admits, per headline cell.

::

    python -m experiments.revision3.r11_absolute_guard

admitted level = best actionable miss of any baseline setting in the cell + tolerance,
tolerance in {0.02, 0.05 (paper), 0.10}. Beside it, the actionable miss of the points
Algorithm 1 selects at the paper's q* (0.95 x ceiling): per-cell oracle k*, best fixed
setting, and the agent's qualifying (cheapest) point per training seed 0-4 (tau = 0.5).
Writes results/revision3/R11/r11_absolute_guard.{json,csv}.
"""

from __future__ import annotations

import csv

from analysis.comparator import best_fixed_baseline, compare_policy, per_cell_oracle_best
from experiments.revision1.provenance import out_dir, stamp, write_json
from experiments.revision2.common import available_seeds, baseline_rows, cells_for

MISS = "actionable_deadline_miss_rate"


def main() -> int:
    base = baseline_rows("headline")
    seeds = available_seeds("headline")
    rows = []
    for tol in (0.02, 0.05, 0.10):
        ref = cells_for("headline", None, base=base)
        oracle = per_cell_oracle_best(ref, miss_margin=tol)
        fixed = best_fixed_baseline(ref, miss_margin=tol)
        agents = {s: cells_for("headline", s, base=base) for s in seeds}
        for c in ref:
            tq, mb = c.resolve_target(0.95), oracle.miss_bound[c.key]
            op = oracle.label(c.key)
            o = (min([p for p in c.curves[op].points if p.quality >= tq and c.within_miss_bound(p, mb)],
                     key=lambda p: p.cost) if op != "none" else None)
            f = c.find(fixed.policy, fixed.param, fixed.value)
            row = {"cell": str(c.key).replace("/clear/fog_bank", ""), "tolerance": tol,
                   "best_miss_any_setting": c.best_miss_rate(), "admitted_level": mb,
                   "oracle_setting": None if o is None else o.label, "oracle_miss": None if o is None else o.extras.get(MISS),
                   "fixed_setting": fixed.label,
                   "fixed_miss": f.extras.get(MISS) if f is not None and fixed.cost(c.key) < float("inf") else None}
            for s in seeds:
                ac = next(x for x in agents[s] if x.key == c.key)
                pts = [p for p in ac.curves["ai_harp"].points if p.quality >= tq and ac.within_miss_bound(p, mb)]
                a = min(pts, key=lambda p: p.cost) if pts else None
                row[f"agent_s{s}_miss"] = None if a is None else a.extras.get(MISS)
            rows.append(row)
    d = out_dir("R11", 3)
    with (d / "r11_absolute_guard.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    write_json(d / "r11_absolute_guard.json", {"rows": rows, "provenance": stamp(
        "R11", {"tolerances": [0.02, 0.05, 0.10], "q_star": 0.95}, seeds={"eval": list(range(10)), "train": seeds})})
    f = lambda v: "–" if v is None else f"{v:.3f}"  # noqa: E731
    for r in rows:
        print(f"{r['cell']:18s} tol {r['tolerance']:.2f} best {f(r['best_miss_any_setting'])} admits <= {f(r['admitted_level'])} | "
              f"oracle {f(r['oracle_miss'])} fixed {f(r['fixed_miss'])} | agent " +
              " ".join(f(r[f'agent_s{s}_miss']) for s in seeds))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
