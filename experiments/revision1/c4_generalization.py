"""C4: generalization -- held-out topology, hazards and weathers (matched coverage).

::

    python -m experiments.revision1.c4_generalization

Cells (experiments/revision1/agent_eval.py GENERALIZATION): urban_grid d=2/20/80
(held-out topology, fog_bank, clear); rural_highway and urban_nlos d=20 with the
held-out hazards waterlogging and black_ice (clear); and with the held-out weathers
moderate_rain and dense_fog (fog_bank). Baselines: 55 knob settings x 10 eval seeds
(results/revision1/B_sweeps/gen_runs.csv). Agent: reference checkpoint, tau = 0.5,
sampled, biases -2..+3 (results/revision1/agent_runs/ref_s0_gen.csv).

Algorithm 1 per cell. Two fixed references: (i) ``shipped`` = dvcast(n_slots=30),
the best single fixed setting chosen on the headline (training-condition) cells and
applied unchanged; (ii) ``refit`` = the best single fixed setting re-chosen over
these 11 cells (hindsight on held-out conditions).
"""

from __future__ import annotations

import json

import numpy as np

from analysis.comparator import best_fixed_baseline, per_cell_oracle_best
from common.config import PROJECT_ROOT
from experiments.revision1.provenance import out_dir, stamp, write_json
from experiments.revision1.runner import cells_from_rows, load_rows

R = PROJECT_ROOT / "results"
SHIPPED = ("dvcast", "n_slots", 30)


def _fin(x):
    return float(x) if x is not None and np.isfinite(x) else None


def main() -> int:
    base = load_rows(R / "revision1" / "B_sweeps" / "gen_runs.csv")
    agent = load_rows(R / "revision1" / "agent_runs" / "ref_s0_gen.csv")
    cells = cells_from_rows(base + agent)
    oracle = per_cell_oracle_best(cells, exclude=("ai_harp",))
    refit = best_fixed_baseline(cells, exclude=("ai_harp",))
    out = {"refit_fixed_best": refit.label, "shipped_fixed": "dvcast(n_slots=30)", "cells": {}}
    for c in cells:
        tq = c.resolve_target(0.95, "relative", ("ai_harp",))
        mb = oracle.miss_bound[c.key]
        a = c.cost_at_matched("ai_harp", tq, "cost", mb) if "ai_harp" in c.curves else np.nan
        sp = c.find(*SHIPPED)
        shipped = (sp.cost if sp is not None and sp.quality >= tq and c.within_miss_bound(sp, mb)
                   else float("inf"))
        o = oracle.cost(c.key)
        rf = refit.cost(c.key)
        apts = c.curves["ai_harp"].points if "ai_harp" in c.curves else []
        out["cells"][str(c.key)] = {
            "ceiling": c.achievable_quality(("ai_harp",)), "target": tq, "miss_bound": mb,
            "oracle_policy": oracle.label(c.key), "oracle_cost": _fin(o),
            "shipped_cost": _fin(shipped), "refit_cost": _fin(rf),
            "agent_cost": _fin(a), "agent_best_rwcr": max((p.quality for p in apts), default=None),
            "agent_tir_at_best": max(apts, key=lambda p: p.quality).extras.get("tir_median_s") if apts else None,
            "regret": _fin(a / o - 1) if np.isfinite(a) and np.isfinite(o) else None,
            "margin_vs_shipped": _fin(shipped / a - 1) if np.isfinite(a) and np.isfinite(shipped) else None,
            "margin_vs_refit": _fin(rf / a - 1) if np.isfinite(a) and np.isfinite(rf) else None,
            "shipped_meets_target": bool(np.isfinite(shipped)),
            "n_agent_seeds": min((p.n_seeds for p in apts), default=0)}
    out["agent_cells_matched"] = sum(e["agent_cost"] is not None for e in out["cells"].values())
    out["shipped_cells_matched"] = sum(e["shipped_meets_target"] for e in out["cells"].values())
    out["n_cells"] = len(out["cells"])
    out["provenance"] = stamp("C4", {"baseline_rows": "results/revision1/B_sweeps/gen_runs.csv",
                                     "agent_rows": "results/revision1/agent_runs/ref_s0_gen.csv",
                                     "checkpoint": "checkpoints/campaign/ref/ckpt_final.pt", "tau": 0.5},
                              seeds=list(range(10)))
    write_json(out_dir("C4") / "c4_generalization.json", out)
    f = lambda v, p=False: "n/a" if v is None else (f"{100 * v:+.1f}%" if p else f"{v:.3f}")  # noqa: E731
    print(f"refit fixed-best: {refit.label}; agent matched {out['agent_cells_matched']}/{out['n_cells']}, "
          f"shipped dvcast(30) matched {out['shipped_cells_matched']}/{out['n_cells']}")
    for k, e in out["cells"].items():
        print(f"{k:42s} q* {e['target']:.3f} oracle {e['oracle_policy']} {f(e['oracle_cost'])} "
              f"shipped {f(e['shipped_cost'])} refit {f(e['refit_cost'])} agent {f(e['agent_cost'])} "
              f"(best RWCR {e['agent_best_rwcr']:.3f}) regret {f(e['regret'], True)} "
              f"margin(shipped) {f(e['margin_vs_shipped'], True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
