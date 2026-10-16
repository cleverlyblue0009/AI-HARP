"""C3: how much of the margin is learning, how much is the fallback?

::

    python -m experiments.revision1.c3_fallback

Headline cells, eval seeds 0-9, Algorithm 1 (q* = 0.95 x ceiling, deadline guard).
Variants, each scored ALONE against the 55 baseline settings (so no agent
variant is a reference for another):

* ``learned gate``  reference checkpoint, tau = 0.5 (the paper's headline arm)
* ``network only``  same checkpoint, tau = 0 (no fallback at all)
* ``random gate``   same checkpoint; the confidence test replaced by a coin with
  the learned gate's measured fallback rate in that (cell, bias)
  (agents/ai_harp.py random_gate_rate; experiments/revision1/agent_eval.py)
* ``pure weighted_p`` the fallback alone: weighted_p's own operating curve
  (11 max_p settings) and its registry default.
"""

from __future__ import annotations

import json

import numpy as np

from analysis.comparator import compare_policy
from common.config import PROJECT_ROOT
from experiments.revision1.provenance import out_dir, stamp, write_json
from experiments.revision1.runner import cells_from_rows, load_rows

R = PROJECT_ROOT / "results"


def _fin(x):
    return float(x) if x is not None and np.isfinite(x) else None


def main() -> int:
    base_all = load_rows(R / "revision1" / "B_sweeps" / "headline_runs.csv")
    base = [r for r in base_all if r["param"] or r["policy"] == "flooding"]
    a = load_rows(R / "revision1" / "agent_runs" / "ref_s0.csv")
    rg = load_rows(R / "revision1" / "agent_runs" / "ref_s0_randomgate.csv")
    sel = lambda rows, tau: [r for r in rows if float(json.loads(r["params_json"]).get("tau", -1)) == tau  # noqa: E731
                             and not json.loads(r["params_json"]).get("deterministic")]
    variants = {"learned gate (tau=0.5)": sel(a, 0.5), "network only (tau=0)": sel(a, 0.0),
                "random gate (matched rate)": rg}
    out: dict = {"variants": {}}
    for name, rows in variants.items():
        res = compare_policy(cells_from_rows(base + rows), "ai_harp")
        cells = {}
        for c in cells_from_rows(rows):
            k = c.key
            pts = sorted(c.curves["ai_harp"].points, key=lambda p: p.value)
            kk = next(x for x in res.agent_cost if (x.scenario, x.density) == (k.scenario, k.density))
            cells[f"{k.scenario}|{k.density:g}"] = {
                "matched_cost": _fin(res.agent_cost[kk]), "regret": _fin(res.regret_per_cell[kk]),
                "margin": _fin(res.margin_per_cell[kk]), "best_rwcr": max(p.quality for p in pts),
                "fallback_rate_by_bias": {p.value: p.extras.get("gate_fallback_rate") for p in pts}}
        out["variants"][name] = {"cells": cells, "cells_matched": 4 - res.cells_failed,
                                 "fixed_best": res.fixed.label}
    # pure weighted_p: its own curve and default, against the other baselines
    wp = {}
    for c in cells_from_rows(base_all):
        others = [r for r in base if r["policy"] != "weighted_p"]
        mine = [r for r in base_all if r["policy"] == "weighted_p" and
                (r["scenario"], float(r["density"])) == (c.key.scenario, c.key.density)]
        curve_res = compare_policy(cells_from_rows(others + [r for r in mine if r["param"]]), "weighted_p")
        kk = next(x for x in curve_res.agent_cost if (x.scenario, x.density) == (c.key.scenario, c.key.density))
        dflt = [r for r in mine if not r["param"]]
        tq = c.resolve_target(0.95, "relative", ("weighted_p",))
        d_rwcr = float(np.mean([float(r["rwcr"]) for r in dflt]))
        wp[f"{c.key.scenario}|{c.key.density:g}"] = {
            "curve_matched_cost": _fin(curve_res.agent_cost[kk]), "curve_regret": _fin(curve_res.regret_per_cell[kk]),
            "default_rwcr": d_rwcr, "default_cost": float(np.mean([float(r["tx_per_at_risk_informed"]) for r in dflt])),
            "default_meets_target": d_rwcr >= tq, "target_without_weighted_p": tq}
    out["pure_weighted_p"] = wp
    out["provenance"] = stamp("C3", {"agent_rows": ["results/revision1/agent_runs/ref_s0.csv",
                                                    "results/revision1/agent_runs/ref_s0_randomgate.csv"],
                                     "baseline_rows": "results/revision1/B_sweeps/headline_runs.csv",
                                     "checkpoint": "checkpoints/campaign/ref/ckpt_final.pt"},
                              seeds=list(range(10)))
    write_json(out_dir("C3") / "c3_fallback.json", out)
    for n, v in out["variants"].items():
        print(n, "matched", v["cells_matched"])
        for k, e in v["cells"].items():
            fr = [round(x, 3) for x in e["fallback_rate_by_bias"].values() if x is not None]
            print(f"   {k:16s} cost {e['matched_cost']} regret {e['regret']} margin {e['margin']} "
                  f"bestRWCR {e['best_rwcr']:.4f} fallback {fr}")
    print("pure weighted_p", json.dumps(wp, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
