"""R9 analysis: Algorithm 1 under the calibrated channel vs the default channel.

Called by ``python -m experiments.revision3.r9_calibrated_channel analyse``. Both channels
go through identical code (comparator on per-seed rows, 55 knob settings, eval seeds 0-9;
agent = full model seeds 0-4, tau 0.5, sampled). Findings compared:

* per-cell oracle scheme and cost; best single fixed setting;
* coverage-saturation spread at rural d=20: best RWCR reached by each of the 7 policy
  families, and the spread (max - min) across families;
* the R4 adaptive picks (chosen on default-channel training seeds; there is no
  calibrated training-seed sweep) evaluated on the eval seeds of each channel;
* agent cells matched and rural d=20 regret per training seed.
"""

from __future__ import annotations

import json

import numpy as np

from analysis.comparator import best_fixed_baseline, compare_policy, per_cell_oracle_best
from experiments.revision1.provenance import out_dir, stamp, write_json
from experiments.revision1.runner import cells_from_rows, load_rows
from experiments.revision2.common import agent_rows, baseline_rows, fin

D = out_dir("R9", 3)
R20 = "rural_highway/d=20/clear/fog_bank"


def _agent_cal(seed):
    out = []
    for r in load_rows(D / f"r9_agent_ref_s{seed}.csv"):
        q = json.loads(r["params_json"])
        if float(q.get("tau", -1)) == 0.5 and not q.get("deterministic"):
            out.append(r)
    return out


def summarise(base, agents, picks):
    cells = cells_from_rows(base)
    oracle = per_cell_oracle_best(cells)
    fixed = best_fixed_baseline(cells)
    out = {"fixed_best": fixed.label, "cells": {}, "agent": {}}
    for c in cells:
        k = str(c.key)
        tq, mb = c.resolve_target(0.95), oracle.miss_bound[c.key]
        fam = {p: max(pt.quality for pt in cu.points) for p, cu in c.curves.items()}
        pick = picks.get((c.key.scenario, c.key.density))
        pt = c.find(*pick) if pick else None
        meets = bool(pt is not None and pt.quality >= tq and c.within_miss_bound(pt, mb))
        o = oracle.cost(c.key)
        out["cells"][k] = {"ceiling": c.achievable_quality(), "target": tq, "miss_bound": mb,
                           "oracle_policy": oracle.label(c.key), "oracle_cost": fin(o),
                           "fixed_cost": fin(fixed.cost(c.key)),
                           "family_best_rwcr": fam, "family_spread": max(fam.values()) - min(fam.values()),
                           "family_spread_excl_p_persistence": (max(v for p, v in fam.items() if p != "p_persistence_03")
                                                                - min(v for p, v in fam.items() if p != "p_persistence_03")),
                           "adaptive_pick": None if pick is None else f"{pick[0]}({pick[1]}={pick[2]})",
                           "adaptive_cost": fin(pt.cost) if pt is not None else None,
                           "adaptive_meets": meets,
                           "adaptive_regret": fin(pt.cost / o - 1) if meets and np.isfinite(o) else None}
    for s, rows in agents.items():
        res = compare_policy(cells_from_rows(base + rows), "ai_harp")
        out["agent"][s] = {"cells_matched": 4 - res.cells_failed,
                           "matched": [str(k) for k, v in res.agent_cost.items() if np.isfinite(v)],
                           "rural_d20_cost": fin(next(v for k, v in res.agent_cost.items() if str(k) == R20)),
                           "rural_d20_regret": fin(next(v for k, v in res.regret_per_cell.items() if str(k) == R20)),
                           "rural_d20_margin": fin(next(v for k, v in res.margin_per_cell.items() if str(k) == R20))}
    return out


def analyse() -> dict:
    r4 = json.loads((D.parent / ".." / "revision2" / "R4" / "r4_adaptive_baseline.json").resolve().read_text())
    picks = {}
    for k, e in r4["headline"]["unbounded"]["cells"].items():
        sc, d = k.split("/d=")
        if e["pick"]:
            pol, rest = e["pick"].split("(", 1) if "(" in e["pick"] else (e["pick"], "")
            par, val = (rest.rstrip(")").split("=") + [""])[:2] if rest else ("", None)
            val = None if val in ("", None) else (int(val) if val.isdigit() else float(val))
            picks[(sc, float(d))] = (pol, par, val)
    default = summarise(baseline_rows("headline"), {s: agent_rows([f"ref_s{s}"], 0.5) for s in range(5)}, picks)
    cal_base = [r for r in load_rows(D / "r9_baseline_runs.csv")]
    calibrated = summarise(cal_base, {s: _agent_cal(s) for s in range(5)}, picks)
    findings = []

    def cmp(name, a, b):
        findings.append({"finding": name, "default": a, "calibrated": b, "same": a == b})

    cmp("best single fixed setting", default["fixed_best"], calibrated["fixed_best"])
    for k in default["cells"]:
        dc, cc = default["cells"][k], calibrated["cells"][k]
        short = k.replace("/clear/fog_bank", "")
        cmp(f"{short}: per-cell oracle scheme", dc["oracle_policy"], cc["oracle_policy"])
        cmp(f"{short}: adaptive pick meets eval target", dc["adaptive_meets"], cc["adaptive_meets"])
    r = R20
    cmp("rural d=20: coverage saturation (family spread excl. p-persistence < 0.01)",
        default["cells"][r]["family_spread_excl_p_persistence"] < 0.01,
        calibrated["cells"][r]["family_spread_excl_p_persistence"] < 0.01)
    cmp("agent cells matched per seed", [default["agent"][s]["cells_matched"] for s in range(5)],
        [calibrated["agent"][s]["cells_matched"] for s in range(5)])
    cmp("agent rural d=20 regret sign per seed",
        [None if default["agent"][s]["rural_d20_regret"] is None else bool(default["agent"][s]["rural_d20_regret"] < 0) for s in range(5)],
        [None if calibrated["agent"][s]["rural_d20_regret"] is None else bool(calibrated["agent"][s]["rural_d20_regret"] < 0) for s in range(5)])
    out = {"default": default, "calibrated": calibrated, "findings": findings,
           "fit": json.loads((D / "r9_fit.json").read_text())["p_max"],
           "provenance": stamp("R9/analyse", {"calibrated_rows": ["results/revision3/R9/r9_baseline_runs.csv",
                                                                  "results/revision3/R9/r9_agent_ref_s{0..4}.csv"]},
                               seeds={"eval": list(range(10)), "train": list(range(5))})}
    write_json(D / "r9_analysis.json", out)
    for f in findings:
        print(f"{'SAME   ' if f['same'] else 'CHANGED'} {f['finding']}: {f['default']} -> {f['calibrated']}")
    return out
