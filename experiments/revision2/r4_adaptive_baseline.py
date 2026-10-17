"""R4: an adaptive classical baseline -- per-cell tuning on TRAINING seeds, no hindsight.

::

    python -m experiments.revision2.sweeps trainsel  # (experiments.revision1.sweeps trainsel)
    python -m experiments.revision2.r4_adaptive_baseline

For each cell: on training seeds 100-109 (results/revision1/B_sweeps/trainsel_runs.csv,
all 55 knob settings) compute q*_train = 0.95 x the best training-seed mean RWCR and the
miss bound (best training-seed actionable miss + 0.05), and pick the cheapest REALISED
setting meeting both (a deployable choice, so no interpolation). Then deploy that one
setting on eval seeds 0-9 and report: its eval cost; whether it meets the eval-side
Algorithm 1 target and guard; regret against the eval-seed per-cell hindsight oracle;
margin against the eval best fixed setting; and the agent (full model, tau=0.5) beside it.
Unbounded, and at the derived deadline (median TIR <= D_r applied on both sides).
"""

from __future__ import annotations

import numpy as np

from analysis.comparator import best_fixed_baseline, compare_policy, per_cell_oracle_best
from experiments.revision1.b1_latency import bounded, derived_deadlines
from experiments.revision1.provenance import out_dir, stamp, write_json
from experiments.revision1.runner import cells_from_rows, load_rows
from experiments.revision2.common import SWEEPS, available_seeds, baseline_rows, cells_for, fin


def choose(train_cells):
    """Per cell: the cheapest realised setting meeting the training-seed target and guard."""
    out = {}
    for c in train_cells:
        tq = c.resolve_target(0.95)
        mb = c.resolve_miss_bound(0.05)
        pts = [p for cu in c.curves.values() for p in cu.points
               if p.quality >= tq and c.within_miss_bound(p, mb)]
        best = min(pts, key=lambda p: p.cost) if pts else None
        out[(c.key.scenario, c.key.density)] = None if best is None else (best.policy, best.param, best.value)
    return out


def analyse(cellset, D):
    train_rows = [r for r in load_rows(SWEEPS / "trainsel_runs.csv") if r["param"] or r["policy"] == "flooding"]
    if cellset == "headline":
        keys = {("rural_highway", 2.0), ("rural_highway", 20.0), ("rural_highway", 80.0), ("urban_nlos", 20.0)}
        train_rows = [r for r in train_rows if (r["scenario"], float(r["density"])) in keys]
    tcells = bounded(cells_from_rows(train_rows), D, False)
    picks = choose(tcells)
    base = baseline_rows(cellset)
    ecells = bounded(cells_for(cellset, None, base=base), D, False)
    oracle = per_cell_oracle_best(ecells)
    fixed = best_fixed_baseline(ecells)
    seeds = available_seeds(cellset)
    agent = {s: compare_policy(bounded(cells_for(cellset, s, base=base), D, False), "ai_harp") for s in seeds}
    cells, n_meet, regs = {}, 0, []
    for c in ecells:
        k = (c.key.scenario, c.key.density)
        tq, mb = c.resolve_target(0.95), oracle.miss_bound[c.key]
        pick = picks.get(k)
        p = c.find(*pick) if pick else None
        meets = bool(p is not None and p.quality >= tq and c.within_miss_bound(p, mb))
        o, f = oracle.cost(c.key), fixed.cost(c.key)
        cost = p.cost if p is not None else np.nan
        e = {"pick": None if pick is None else (f"{pick[0]}({pick[1]}={pick[2]})" if pick[1] else pick[0]),
             "eval_cost": fin(cost), "eval_rwcr": fin(p.quality) if p is not None else None,
             "eval_miss": fin(p.extras.get("actionable_deadline_miss_rate")) if p is not None else None,
             "meets_eval_target_and_guard": meets, "q_star_eval": tq,
             "oracle_policy": oracle.label(c.key), "oracle_cost": fin(o), "fixed_cost": fin(f),
             "regret_vs_oracle": fin(cost / o - 1) if meets and np.isfinite(o) else None,
             "margin_vs_fixed": fin(f / cost - 1) if meets and np.isfinite(f) else None,
             "agent": {}}
        for s, res in agent.items():
            ck = next(x for x in res.agent_cost if (x.scenario, x.density) == k)
            a = res.agent_cost[ck]
            e["agent"][s] = {"cost": fin(a), "regret": fin(res.regret_per_cell[ck]),
                             "vs_adaptive": fin(cost / a - 1) if meets and np.isfinite(a) else None}
        n_meet += meets
        if e["regret_vs_oracle"] is not None:
            regs.append(e["regret_vs_oracle"])
        cells["/".join([k[0], f"d={k[1]:g}"])] = e
    return {"fixed_best": fixed.label, "cells": cells, "adaptive_meets": n_meet, "n_cells": len(cells),
            "adaptive_no_pick": sum(1 for e in cells.values() if e["pick"] is None),
            "adaptive_median_regret": float(np.median(regs)) if regs else None,
            "agent_matched": {s: 4 - r.cells_failed if cellset == "headline" else len(cells) - r.cells_failed
                              for s, r in agent.items()}, "seeds": seeds}


def main() -> int:
    der = {sc: v["D_s"] for sc, v in derived_deadlines().items()}
    out = {cs: {bn: analyse(cs, D) for bn, D in (("unbounded", float("inf")), ("derived", der))}
           for cs in ("headline", "grid24")}
    out["provenance"] = stamp("R4", {"selection": "cheapest realised setting meeting 0.95 x training ceiling "
                                                  "and training miss guard (+0.05), training seeds 100-109",
                                     "train_rows": "results/revision1/B_sweeps/trainsel_runs.csv"},
                              seeds={"selection": list(range(100, 110)), "eval": list(range(10))})
    write_json(out_dir("R4", 2) / "r4_adaptive_baseline.json", out)
    f = lambda v, p=False: "–" if v is None else (f"{100 * v:+.1f}%" if p else f"{v:.3f}")  # noqa: E731
    for cs in ("headline", "grid24"):
        for bn, r in out[cs].items():
            print(f"== {cs} {bn}: adaptive meets eval target in {r['adaptive_meets']}/{r['n_cells']} "
                  f"(no pick {r['adaptive_no_pick']}), median regret {f(r['adaptive_median_regret'], True)}; "
                  f"agent matched {r['agent_matched']}")
            if cs == "headline":
                for k, e in r["cells"].items():
                    ag = " ".join(f"s{s}:{f(v['cost'])}" for s, v in e["agent"].items())
                    print(f"   {k:18s} pick {e['pick']} cost {f(e['eval_cost'])} meets {e['meets_eval_target_and_guard']} "
                          f"regret {f(e['regret_vs_oracle'], True)} | oracle {f(e['oracle_cost'])} | agent {ag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
