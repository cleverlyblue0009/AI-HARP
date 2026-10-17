"""R1: Algorithm 1 with a TAIL-latency bound -- seed-mean p95 TIR <= D.

::

    python -m experiments.revision2.r1_tail_latency

Same rule as revision-1 B1 (the bound enters the comparator's eligibility test, so the
interpolation's lower anchor must satisfy it too; q* and the miss bound held at their
unbounded values), but on ``tir_p95_s`` and with D in {1, 2, 5 s, derived}. The derived
deadline is B1's D_r (rural 15.39 s, urban_grid 26.65 s, urban_nlos 1.89 s).

Also reported at every reference point (per-cell oracle k*, best fixed setting, agent k*):
the actionable deadline miss rate -- the fraction of at-risk vehicles that still had time
to react at origination but were not warned before their own reaction deadline
(analysis/metrics.py actionable_deadline_miss_rate). Algorithm 1's guard caps it at the
cell's best + 0.05, which is the per-vehicle constraint.

Agent: full model, sampled, tau = 0.5; training seeds 0-2 (headline cells; 3-4 added when
R6 finishes) and seed 0 on the 24-cell grid.
"""

from __future__ import annotations

import numpy as np

from analysis.comparator import best_fixed_baseline, compare_policy, per_cell_oracle_best
from experiments.revision1.b1_latency import bounded, derived_deadlines
from experiments.revision1.provenance import out_dir, stamp, write_json
from experiments.revision2.common import available_seeds, baseline_rows, cells_for, fin

METRIC = "tir_p95_s"
MISS = "actionable_deadline_miss_rate"


def _setting(cell, policy, tq, mb):
    pts = [p for p in cell.curves.get(policy, type("C", (), {"points": []})()).points
           if p.quality >= tq and cell.within_miss_bound(p, mb)] if policy in cell.curves else []
    return min(pts, key=lambda p: p.cost) if pts else None


def analyse(cellset: str) -> dict:
    der = {sc: v["D_s"] for sc, v in derived_deadlines().items()}
    bounds = {"unbounded": float("inf"), "D=1s": 1.0, "D=2s": 2.0, "D=5s": 5.0, "D=derived": der}
    base = baseline_rows(cellset)
    seeds = available_seeds(cellset)
    out = {"seeds": seeds, "bounds": {}}
    for bn, D in bounds.items():
        ref_cells = bounded(cells_for(cellset, None, base=base), D, False, METRIC)
        oracle = per_cell_oracle_best(ref_cells)
        fixed = best_fixed_baseline(ref_cells)
        per_seed = {}
        for s in seeds:
            cs = bounded(cells_for(cellset, s, base=base), D, False, METRIC)
            res = compare_policy(cs, "ai_harp")
            per_seed[s] = {str(k): {"cost": fin(res.agent_cost[k]), "regret": fin(res.regret_per_cell[k]),
                                    "margin": fin(res.margin_per_cell[k])} for k in res.agent_cost}
            for c in cs:
                tq, mb = c.resolve_target(0.95, "relative", ("ai_harp",)), res.oracle.miss_bound[c.key]
                a = _setting(c, "ai_harp", tq, mb)
                per_seed[s][str(c.key)].update(
                    agent_bias=None if a is None else a.value,
                    agent_miss=None if a is None else a.extras.get(MISS),
                    agent_p95=None if a is None else a.extras.get(METRIC))
        cells = {}
        for c in ref_cells:
            tq, mb = c.resolve_target(0.95), oracle.miss_bound[c.key]
            op = oracle.label(c.key)
            o = _setting(c, op, tq, mb) if op != "none" else None
            f = c.find(fixed.policy, fixed.param, fixed.value)
            f_ok = f is not None and np.isfinite(fixed.cost(c.key))
            cells[str(c.key)] = {
                "D_s": c.D, "target": tq, "miss_bound": mb, "best_miss": c.best_miss_rate(),
                "oracle_policy": op, "oracle_setting": None if o is None else o.label,
                "oracle_cost": fin(oracle.cost(c.key)),
                "oracle_miss": None if o is None else o.extras.get(MISS),
                "oracle_p95": None if o is None else o.extras.get(METRIC),
                "fixed_cost": fin(fixed.cost(c.key)), "fixed_miss": f.extras.get(MISS) if f_ok else None,
                "fixed_p95": f.extras.get(METRIC) if f_ok else None,
                "agent": {s: per_seed[s][str(c.key)] for s in seeds}}
        matched = {s: sum(1 for v in per_seed[s].values() if v["cost"] is not None) for s in seeds}
        out["bounds"][bn] = {"fixed_best": fixed.label, "cells": cells, "agent_cells_matched": matched,
                             "fixed_cells_failed": sum(1 for v in fixed.per_cell_cost.values() if not np.isfinite(v)),
                             "cells_no_qualifier": sum(1 for v in cells.values() if v["oracle_cost"] is None)}
    return out


def main() -> int:
    res = {cs: analyse(cs) for cs in ("headline", "grid24")}
    res["provenance"] = stamp("R1", {"metric": METRIC, "bounds": ["unbounded", 1, 2, 5, "derived"],
                                     "target": 0.95, "miss_margin": 0.05, "variant": "q* fixed"},
                              seeds={"eval": list(range(10)), "train": {k: res[k]["seeds"] for k in ("headline", "grid24")}})
    write_json(out_dir("R1", 2) / "r1_tail_latency.json", res)
    f = lambda v, p=False: "–" if v is None else (f"{100 * v:+.1f}%" if p else f"{v:.3f}")  # noqa: E731
    for cs in ("headline", "grid24"):
        print("==", cs, "seeds", res[cs]["seeds"])
        for bn, b in res[cs]["bounds"].items():
            print(f"  {bn:10s} fixed {b['fixed_best']:26s} fixed_fail {b['fixed_cells_failed']} "
                  f"no_qualifier {b['cells_no_qualifier']} agent matched {b['agent_cells_matched']}")
            if cs == "headline":
                for k, e in b["cells"].items():
                    ag = "  ".join(f"s{s}:{f(v['cost'])}({f(v['regret'], True)}/{f(v['margin'], True)})"
                                   for s, v in e["agent"].items())
                    print(f"     {k.replace('/clear/fog_bank', ''):22s} oracle {e['oracle_policy']} {f(e['oracle_cost'])} "
                          f"miss {f(e['oracle_miss'])} p95 {f(e['oracle_p95'])} | fixed {f(e['fixed_cost'])} miss {f(e['fixed_miss'])} | {ag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
