"""Revision 7, Experiment A: policy with and without the decision cap, per simulator (results/revision7/analysis/exp_a.json).

    python -m experiments.revision7.analysis.exp_a

Definitions: results/revision7/ADDENDUM.md (veins_methods: Algorithm 1 inside each simulator, seeds 200-209). No new test;
EVAL-hindsight comparison. The decision rule reads only: capped policy, Veins, rural d=20, at least 3 of 5 training seeds.
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict

import numpy as np

from common.config import PROJECT_ROOT
from experiments.revision1.provenance import stamp, write_json
from experiments.revision5.analysis.common import num, rows
from experiments.revision5.analysis.h3 import veins_rows
from experiments.revision6.analysis.veins_methods import CK, SEEDS, V6, build, cell_eval, custom_rows, veins6_rows

R7 = PROJECT_ROOT / "results" / "revision7"
OUT = R7 / "analysis"
RUNS = [f"ref_s{k}" for k in range(5)]
CELLS = (("rural_highway", 20.0), ("urban_nlos", 20.0))
RULE_CELL = ("rural_highway", 20.0)


def fin(x):
    return x if isinstance(x, float) and math.isfinite(x) else None


def capped_veins() -> list[dict]:
    out = []
    for line in (R7 / "veins" / "capped_rows.jsonl").open(encoding="utf-8"):
        r = json.loads(line)["row"]
        run, b = r["knobs"].rsplit("_b", 1)
        out.append({"scenario": r["scenario"], "density": r["density"], "seed": r["seed"], "rwcr": r["rwcr"],
                    "tx_per_at_risk_informed": r["tx_per_at_risk_informed"], "actionable_deadline_miss_rate": r["actionable_deadline_miss_rate"],
                    "tir_p95_s": r["tir_p95_s"], "policy": run, "value": float(b), "transmissions": r["transmissions"],
                    "decisions": sum(r["action_mix"].values()), "capped_callbacks": r["capped_callbacks"]})
    return out


def capped_custom() -> list[dict]:
    out = []
    for line in (R7 / "runs" / "custom_capped_rows.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        out.append({**r, "policy": r["run"], "value": float(r["bias"])})
    return out


def in_cell(r, c) -> bool:
    return (r["scenario"], float(r["density"])) == c and int(r["seed"]) in SEEDS


def grid(t) -> dict:
    return {f"{k[1]:g}": {"rwcr": float(t.mean("rwcr")[i]), "cost": float(t.mean("cost")[i]), "miss": float(t.mean("miss")[i])}
            for i, k in enumerate(t.settings)}


def arm(cell, label) -> dict:
    e = cell_eval(cell, RUNS)
    return {"arm": label, "policy_per_seed": {r: fin(e["policy"][r]) for r in RUNS}, "seeds_qualifying": e["policy_n_pick"],
            "aggregate": fin(e["policy_aggregate"]), "grid": {r: grid(cell["policy"][r]) for r in RUNS},
            "best_rwcr_per_seed": {r: max(v["rwcr"] for v in grid(cell["policy"][r]).values()) for r in RUNS}}


def activity(rs) -> dict:
    if not rs:
        return {"replays": 0}
    out = {"replays": len(rs)}
    for k in ("transmissions", "decisions", "capped_callbacks"):
        v = [num(r[k]) for r in rs if r.get(k) not in (None, "")]
        out[f"{k}_per_replay"] = float(np.mean(v)) if v else None
    return out


def main(argv=None) -> int:
    import numpy
    print(f"[env] {sys.executable} numpy {numpy.__version__}", flush=True)
    vr, _ = veins_rows()
    rule, unc = veins6_rows(RUNS)
    cap_v, cap_c = capped_veins(), capped_custom()
    V_unc, V_cap = build(vr, rule, unc, RUNS, cells=CELLS), build(vr, rule, cap_v, RUNS, cells=CELLS)
    cc, cr, cp = custom_rows(RUNS)
    C_unc, C_cap = build(cc, cr, cp, RUNS, cells=CELLS), build(cc, cr, cap_c, RUNS, cells=CELLS)

    # activity of the uncapped arms: Veins from the committed revision-6 rows; custom transmissions from the committed rows
    v_act = []
    for line in V6.open(encoding="utf-8"):
        r = json.loads(line)["row"]
        if r["policy"] == "ai_harp":
            v_act.append({"scenario": r["scenario"], "density": r["density"], "seed": r["seed"], "transmissions": r["transmissions"],
                          "decisions": sum(r["action_mix"].values())})
    # the matching committed custom rows, by task_hash
    ref = {r["task_hash"]: r for r in rows("agent__EVAL__headline__default__sumo.csv")}
    same, diff, missing = 0, [], 0
    for r in cap_c:
        q = ref.get(r["task_hash"])
        if q is None:
            missing += 1
        elif all(math.isclose(num(q[k]), num(r[k]), rel_tol=1e-12) or (math.isnan(num(q[k])) and math.isnan(num(r[k])))
                 for k in ("rwcr", "tx_per_at_risk_informed", "actionable_deadline_miss_rate", "tir_p95_s", "transmissions")):
            same += 1
        else:
            diff.append(r["task_hash"])

    cells: dict = defaultdict(dict)
    for c in CELLS:
        ek = CK(c)
        ve, cu = cell_eval(V_unc[c], RUNS), cell_eval(C_unc[c], RUNS)
        fired = sum(r["capped_callbacks"] for r in cap_c if in_cell(r, c))
        c_unc_act = activity([q for q in ref.values() if in_cell(q, c) and q["task_hash"] in {r["task_hash"] for r in cap_c}])
        c_unc_act["decisions_per_replay"] = (activity([r for r in cap_c if in_cell(r, c)])["decisions_per_replay"] if fired == 0 else None)
        cells[ek] = {
            "veins": {"q_star": ve["q_star"], "m_star": ve["m_star"],
                      "best_classical": [ve["best_classical"][0], fin(ve["best_classical"][1])], "best_rule": [ve["best_rule"][0], fin(ve["best_rule"][1])],
                      "uncapped": {**arm(V_unc[c], "uncapped"), **activity([r for r in v_act if in_cell(r, c)])},
                      "capped": {**arm(V_cap[c], "capped"), **activity([r for r in cap_v if in_cell(r, c)])}},
            "custom": {"q_star": cu["q_star"], "m_star": cu["m_star"],
                       "best_classical": [cu["best_classical"][0], fin(cu["best_classical"][1])], "best_rule": [cu["best_rule"][0], fin(cu["best_rule"][1])],
                       "uncapped": {**arm(C_unc[c], "uncapped"), **c_unc_act},
                       "capped": {**arm(C_cap[c], "capped"), **activity([r for r in cap_c if in_cell(r, c)])}}}
    n = cells[CK(RULE_CELL)]["veins"]["capped"]["seeds_qualifying"]
    payload = {"label": "EVAL-hindsight comparison on seeds 200-209; no new test", "runs": RUNS, "seeds": SEEDS, "cells": dict(cells),
               "decision_rule": {"cell": CK(RULE_CELL), "simulator": "veins", "seeds_qualifying_capped": n, "of": len(RUNS), "needed": 3,
                                 "verdict": "sufficient to restore qualification" if n >= 3 else "not sufficient"},
               "custom_capped_vs_committed_rows": {"identical": same, "different": len(diff), "without_committed_row": missing, "different_hashes": diff[:20]}}
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "exp_a.json", {"provenance": stamp("revision7/analysis/exp_a.json", {"runs": RUNS}, seeds=SEEDS), **payload})
    print(OUT / "exp_a.json")
    print(json.dumps(payload["decision_rule"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
