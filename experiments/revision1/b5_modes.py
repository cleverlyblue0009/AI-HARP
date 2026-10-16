"""B5: sampled vs argmax evaluation of the reference agent, all four headline cells.

::

    python -m experiments.revision1.b5_modes

From results/revision1/agent_runs/ref_s0.csv (tau=0.5, both modes, biases
-2..+3, eval seeds 0-9) against the per-seed baseline sweep. Reports per cell:
best RWCR over biases per mode, the gap (sampled - argmax), matched cost /
regret / margin per mode (Algorithm 1), and, per bias, the paired per-seed RWCR
difference with a Wilcoxon p. Also checks every per-seed mean against the
committed results/agent_ref/{sampled,argmax}/pareto_cells_agent.json.
"""

from __future__ import annotations

import json

import numpy as np

from analysis.comparator import compare_policy, load_cells
from analysis.stats import paired_test
from common.config import PROJECT_ROOT
from experiments.revision1.provenance import out_dir, stamp, write_json
from experiments.revision1.runner import cells_from_rows, load_rows

R = PROJECT_ROOT / "results"


def agent_rows(mode: str, tau: float = 0.5):
    out = []
    for r in load_rows(R / "revision1" / "agent_runs" / "ref_s0.csv"):
        p = json.loads(r["params_json"])
        if float(p.get("tau", -1)) == tau and bool(p.get("deterministic")) == (mode == "argmax") \
                and "random_gate_rate" not in p:
            out.append(r)
    return out


def main() -> int:
    base = [r for r in load_rows(R / "revision1" / "B_sweeps" / "headline_runs.csv")
            if r["param"] or r["policy"] == "flooding"]
    rows = {m: agent_rows(m) for m in ("sampled", "argmax")}
    res = {m: compare_policy(cells_from_rows(base + rows[m]), "ai_harp") for m in rows}
    curves = {m: {(c.key.scenario, c.key.density): c.curves["ai_harp"]
                  for c in cells_from_rows(rows[m])} for m in rows}
    # reproduction check against the committed curves
    check = {}
    for m in rows:
        for c in load_cells(R / "agent_ref" / m / "pareto_cells_agent.json"):
            mine = {p.value: p for p in curves[m][(c.key.scenario, c.key.density)].points}
            for p in c.curves["ai_harp"].points:
                q = mine.get(float(p.value))
                check[f"{m}|{c.key}|{p.value}"] = None if q is None else max(
                    abs(q.quality - p.quality), abs(q.cost - p.cost))
    out = {"cells": {}, "max_abs_diff_vs_committed": max(v for v in check.values() if v is not None),
           "n_points_checked": sum(v is not None for v in check.values())}
    for key in curves["sampled"]:
        e = {}
        for m in rows:
            pts = curves[m][key].points
            best = max(pts, key=lambda p: p.quality)
            ck = next(k for k in res[m].agent_cost if (k.scenario, k.density) == key)
            fin = lambda x: float(x) if np.isfinite(x) else None  # noqa: E731
            e[m] = {"best_rwcr": best.quality, "best_bias": best.value,
                    "matched_cost": fin(res[m].agent_cost[ck]), "regret": fin(res[m].regret_per_cell[ck]),
                    "margin": fin(res[m].margin_per_cell[ck]),
                    "target": next(c for c in cells_from_rows(base) if (c.key.scenario, c.key.density) == key)
                    .resolve_target(0.95, "relative", ("ai_harp",))}
        e["best_rwcr_gap_sampled_minus_argmax"] = e["sampled"]["best_rwcr"] - e["argmax"]["best_rwcr"]
        per_bias = []
        for b in sorted({float(r["value"]) for r in rows["sampled"]}):
            def seeds(m):
                rr = sorted([r for r in rows[m] if (r["scenario"], float(r["density"])) == key
                             and float(r["value"]) == b], key=lambda r: int(r["seed"]))
                return [float(r["rwcr"]) for r in rr], [float(r["tx_per_at_risk_informed"]) for r in rr]
            (sr, sc), (ar, ac) = seeds("sampled"), seeds("argmax")
            t = paired_test(sr, ar, "rwcr")
            per_bias.append({"bias": b, "rwcr_sampled": float(np.mean(sr)), "rwcr_argmax": float(np.mean(ar)),
                             "rwcr_diff": float(np.mean(sr) - np.mean(ar)), "p_wilcoxon": t.p_value,
                             "rank_biserial": t.rank_biserial,
                             "cost_sampled": float(np.mean(sc)), "cost_argmax": float(np.mean(ac))})
        e["per_bias"] = per_bias
        out["cells"][f"{key[0]}|{key[1]:g}"] = e
    out["provenance"] = stamp("B5", {"agent_rows": "results/revision1/agent_runs/ref_s0.csv",
                                     "baseline_rows": "results/revision1/B_sweeps/headline_runs.csv",
                                     "tau": 0.5}, seeds=list(range(10)))
    write_json(out_dir("B5") / "b5_modes.json", out)
    print("max |diff| vs committed:", out["max_abs_diff_vs_committed"], "over", out["n_points_checked"])
    for k, e in out["cells"].items():
        print(k, {m: {kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in e[m].items()}
                  for m in ("sampled", "argmax")}, "gap %.4f" % e["best_rwcr_gap_sampled_minus_argmax"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
