"""H5 and Experiments C/D (results/revision8/ADDENDUM.md sections 6-8), Veins; plus the Holm step over H4 and H5.
Written before any revision-8 Veins replay existed.

    python -m experiments.revision8.analysis.h5 h5       -> results/revision8/analysis/h5.json   (needs h5B_rows; h5A optional)
    python -m experiments.revision8.analysis.h5 expcd    -> results/revision8/analysis/exp_cd.json
    python -m experiments.revision8.analysis.h5 holm     -> results/revision8/analysis/holm.json (needs h4.json, h5.json)

H5 (Tier B primary): frozen Veins q*, m* from results/revision8/veins_thresholds.json (Tier A classical); a training seed
qualifies in a cell if its Algorithm-1 matched cost over its six deltas (Tier B seed means; guard mean miss <= m*) is
finite (revision-6 veins_methods.matched, unchanged). Per cell: rule met iff rand >= 3/5 and plain < 3/5; one-sided
Fisher exact test (rand qualifies more often); p_H5 = min(1, 2 min over the two cells). Supported iff the rule is met in
>= 1 cell and the Holm-adjusted p_H5 < 0.05. No Tier B classical matched cost is computed (qualification test only).
Tier A (descriptive, EVAL-hindsight): the same with the same thresholds (fitted on Tier A).
Experiments C/D: AI-HARP ref_s0-4 on Tier A with the same thresholds; "explains the failure" iff rural d=20 qualification
>= 3/5 (revision-6/7 baseline: 0/5).
"""

from __future__ import annotations

import json
import math
import sys

from common.config import PROJECT_ROOT
from experiments.revision1.provenance import stamp, write_json
from experiments.revision5.analysis.common import holm, tables
from experiments.revision6.analysis.veins_methods import matched

R8 = PROJECT_ROOT / "results" / "revision8"
CELLS = (("rural_highway", 20.0), ("urban_nlos", 20.0))
TIER_A, TIER_B = list(range(200, 210)), list(range(210, 230))
ARMS = {"plain": [f"plain_s{k}" for k in range(5)], "rand": [f"rand_s{k}" for k in range(5)]}
REF = [f"ref_s{k}" for k in range(5)]


def frozen() -> dict:
    c = json.loads((R8 / "veins_thresholds.json").read_text())["cells"]
    return {(sc, d): (c[f"{sc}|{d:g}"]["q_star"], c[f"{sc}|{d:g}"]["m_star"]) for sc, d in CELLS}


def load(mode: str, sep: str) -> list[dict]:
    """Scored rows as revision-5-style dicts; policy = the run (plain_s0, ref_s0, ...), value = delta or bias."""
    out = []
    for line in (R8 / "veins" / f"{mode}_rows.jsonl").open(encoding="utf-8"):
        r = json.loads(line)["row"]
        run, v = r["knobs"].rsplit(sep, 1)
        out.append({"scenario": r["scenario"], "density": r["density"], "seed": r["seed"], "policy": run,
                    "value": float(v), "rwcr": r["rwcr"], "tx_per_at_risk_informed": r["tx_per_at_risk_informed"],
                    "actionable_deadline_miss_rate": r["actionable_deadline_miss_rate"], "tir_p95_s": r["tir_p95_s"]})
    return out


def per_run(rows: list[dict], runs: list[str], seeds: list[int], th: dict) -> dict:
    out = {}
    for run in runs:
        t = tables([r for r in rows if r["policy"] == run], CELLS, seeds, key=lambda r: (r["policy"], r["value"]))
        out[run] = {}
        for c in CELLS:
            q, m = th[c]
            mc = matched(t[c], run, q, m)
            rw, co, mi = (t[c].mean(x) for x in ("rwcr", "cost", "miss"))
            out[run][f"{c[0]} d={c[1]:g}"] = {
                "matched_cost": mc if math.isfinite(mc) else None, "qualifies": bool(math.isfinite(mc)),
                "per_setting": {str(k[1]): {"rwcr": float(rw[i]), "cost": float(co[i]), "miss": float(mi[i])}
                                for i, k in enumerate(t[c].settings)}}
    return out


def fisher_greater(a_q: int, a_n: int, b_q: int, b_n: int) -> float:
    from scipy.stats import fisher_exact
    return float(fisher_exact([[a_q, a_n - a_q], [b_q, b_n - b_q]], alternative="greater")[1])


def h5_tier(mode: str, seeds: list[int], th: dict) -> dict:
    rows = load(mode, "_d")
    res = {arm: per_run(rows, runs, seeds, th) for arm, runs in ARMS.items()}
    cells = {}
    for sc, d in CELLS:
        ck = f"{sc} d={d:g}"
        nq = {arm: sum(res[arm][r][ck]["qualifies"] for r in runs) for arm, runs in ARMS.items()}
        cells[ck] = {"rand_qualifying": nq["rand"], "plain_qualifying": nq["plain"],
                     "rule_met": bool(nq["rand"] >= 3 and nq["plain"] < 3),
                     "p_fisher_one_sided": fisher_greater(nq["rand"], 5, nq["plain"], 5)}
    p = min(1.0, 2 * min(c["p_fisher_one_sided"] for c in cells.values()))
    return {"seeds": seeds, "cells": cells, "rule_met_any_cell": any(c["rule_met"] for c in cells.values()),
            "p_H5": p, "per_run": res}


def expcd(th: dict) -> dict:
    out = {}
    for mode, label in (("expC", "geometric neighbour table"), ("expD", "one-epoch coupling lag removed")):
        p = R8 / "veins" / f"{mode}_rows.jsonl"
        if not p.exists():
            out[mode] = {"status": "not run"}
            continue
        pr = per_run(load(mode, "_b"), REF, TIER_A, th)
        n_r = sum(pr[r]["rural_highway d=20"]["qualifies"] for r in REF)
        n_u = sum(pr[r]["urban_nlos d=20"]["qualifies"] for r in REF)
        out[mode] = {"what": label, "seeds": TIER_A, "rural_d20_qualifying": n_r, "urban_nlos_d20_qualifying": n_u,
                     "baseline_revision6_7_rural_d20": "0/5",
                     "verdict": "explains the failure" if n_r >= 3 else "does not explain it", "per_run": pr}
    return out


def main(argv=None) -> int:
    mode = (argv if argv is not None else sys.argv[1:])[0]
    th = frozen()
    (R8 / "analysis").mkdir(parents=True, exist_ok=True)
    if mode == "h5":
        out = {"hypothesis": "H5", "thresholds": {f"{c[0]} d={c[1]:g}": th[c] for c in CELLS},
               "tier_B_primary": h5_tier("h5B", TIER_B, th)}
        if (R8 / "veins" / "h5A_rows.jsonl").exists():
            out["tier_A_descriptive_hindsight"] = h5_tier("h5A", TIER_A, th)
        name = "h5.json"
    elif mode == "expcd":
        out, name = expcd(th), "exp_cd.json"
    else:
        h4 = json.loads((R8 / "analysis" / "h4.json").read_text())["tiers"]["B"]["vs_adaptive_classical"]
        h5 = json.loads((R8 / "analysis" / "h5.json").read_text())["tier_B_primary"]
        adj = holm({"H4": h4["p_noninferiority"], "H5": h5["p_H5"]})
        out = {"raw": {"H4": h4["p_noninferiority"], "H5": h5["p_H5"]}, "holm_adjusted": adj, "alpha": 0.05,
               "H4_supported": bool(h4["decision"]["rule_met"] and adj["H4"] < 0.05),
               "H5_supported": bool(h5["rule_met_any_cell"] and adj["H5"] < 0.05)}
        name = "holm.json"
    write_json(R8 / "analysis" / name, {"provenance": stamp(f"revision8/analysis/{name}", {"mode": mode}), **out})
    print(R8 / "analysis" / name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
