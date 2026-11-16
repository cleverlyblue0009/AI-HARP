"""H4 (results/revision8/ADDENDUM.md section 5): STRUCT-plain vs the revision-5 adaptive classical comparator, custom
simulator. Written before any STRUCT evaluation row existed.

    python -m experiments.revision8.analysis.h4 [--B 10000]     -> results/revision8/analysis/h4.json

Reuses experiments/revision5/analysis/h2.py (policy_eval, method_eval, statistic) unchanged. Per tier (A = 200-209,
B = 210-229; B primary): every training seed picks its delta on SEL; comparator = cheapest qualifying of the 55 classical
settings on SEL; frozen revision-5 q*, m*; D = infinity. Decision (Tier B): (i) STRUCT-plain qualifying cells >= comparator
qualifying cells; (ii) one-sided 95 % upper bound of T (0.95 bootstrap quantile) < +0.10; p_H4 = share of defined
replicates with T >= 0.10 (1 if T undefined). Hierarchical bootstrap over training seeds, SEL seeds and the tier's EVAL
seeds, B = 10,000, RNG 20261116, every selection recomputed.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from common.config import PROJECT_ROOT
from experiments.revision5.analysis.common import HEADLINE, SEL, finite, rows, tables, thresholds, wilson
from experiments.revision5.analysis.h2 import agent_tables, method_eval, policy_eval, statistic
from experiments.revision1.runner import load_rows

R8 = PROJECT_ROOT / "results" / "revision8"
TIERS = {"A": list(range(200, 210)), "B": list(range(210, 230))}
MARGIN = 0.10
RNG = 20261116
B_DEFAULT = 10_000


def struct_tables(arm: str, seeds: list[int], seedset: str) -> dict[str, dict]:
    """{run: {cell: Table over deltas}}; a row belongs to a run by its theta (results/revision8/train/*/theta.json)."""
    th = {}
    for k in range(10):
        d = json.loads((R8 / "train" / f"{arm}_s{k}" / "theta.json").read_text())
        th[json.dumps([round(v, 12) for v in d["theta"]])] = f"{arm}_s{k}"
    allrows = load_rows(R8 / "runs" / f"struct_{arm}__{seedset}__headline__default__fallback.csv")
    out = {}
    for key, run in th.items():
        def where(r, key=key):
            return json.dumps([round(v, 12) for v in json.loads(r["params_json"])["theta"]]) == key
        out[run] = tables(allrows, HEADLINE, seeds, key=lambda r: (float(r["value"]),), where=where)
    return out


def evaluate(runs, asel, aev, csel, cev, th, si, ei, ri):
    pol = {c: policy_eval(asel, aev, ri, runs, c, *th[c], math.inf, si, ei) for c in HEADLINE}
    com = {c: method_eval(csel, cev, c, *th[c], math.inf, si, ei) for c in HEADLINE}
    return pol, com


def analyse_tier(runs, asel, aev, csel, cev, th, n_eval, nB, rng) -> dict:
    R = list(range(len(runs)))
    pol, com = evaluate(runs, asel, aev, csel, cev, th, None, None, R)
    T, per = statistic(HEADLINE, com, pol)
    q_pol = sum(1 for c in HEADLINE if pol[c]["qualifies"])
    q_com = sum(1 for c in HEADLINE if com[c] is not None and com[c]["qualifies"])
    bs = []
    for _ in range(nB):
        ri = list(rng.integers(0, len(runs), len(runs)))
        si, ei = rng.integers(0, len(SEL), len(SEL)), rng.integers(0, n_eval, n_eval)
        p, c = evaluate(runs, asel, aev, csel, cev, th, si, ei, ri)
        bs.append(statistic(HEADLINE, c, p)[0])
    bs = np.array(bs, dtype=float)
    ok = bs[~np.isnan(bs)]
    res = {"T": finite(T) if math.isfinite(T) else (None if math.isnan(T) else "inf"),
           "per_cell": {f"{c[0]} d={c[1]:g}": (v if math.isfinite(v) else "inf") for c, v in per.items()},
           "cells_both_qualify": len(per), "qualifying_cells": {"struct": q_pol, "comparator": q_com},
           "B": nB, "undefined_replicates": int(np.isnan(bs).sum()), "infinite_replicates": int(np.isinf(bs).sum())}
    if ok.size:
        qf = lambda x: float(np.quantile(ok, x, method="inverted_cdf"))   # noqa: E731
        js = lambda x: x if math.isfinite(x) else ("inf" if x > 0 else "-inf")   # noqa: E731
        res.update(upper95_one_sided=js(qf(0.95)), ci95=[js(qf(0.025)), js(qf(0.975))],
                   p_noninferiority=float((ok >= MARGIN).mean()),
                   p_two_sided_rev5=float(min(1.0, 2 * min((ok <= 0).mean(), (ok >= 0).mean()))),
                   superiority_rev5=bool(qf(0.975) < 0), equivalence_TOST_5pct_rev5=bool(qf(0.05) > -0.05 and qf(0.95) < 0.05))
    else:
        res.update(upper95_one_sided=None, p_noninferiority=1.0)
    if math.isnan(T):                      # undefined: no cell where both qualify -> (ii) fails, p_H4 = 1
        res["p_noninferiority"] = 1.0
    cond_i = q_pol >= q_com
    ub = res.get("upper95_one_sided")
    cond_ii = (not math.isnan(T)) and isinstance(ub, float) and ub < MARGIN
    res["decision"] = {"i_qualification_count": cond_i, "ii_noninferiority_10pct": bool(cond_ii),
                       "rule_met": bool(cond_i and cond_ii)}
    kp = sum(1 for c in HEADLINE for v in pol[c]["per_seed"].values() if v is not None and v["qualifies"])
    res["qualification_per_seed_cell"] = [kp, len(HEADLINE) * len(runs), wilson(kp, len(HEADLINE) * len(runs))]
    res["point_detail"] = {f"{c[0]} d={c[1]:g}": {
        "struct": {k: v for k, v in pol[c].items() if k != "per_seed"} | {"cost": finite(pol[c]["cost"])},
        "struct_per_seed": {r: (None if v is None else {**v, "cost": finite(v["cost"])}) for r, v in pol[c]["per_seed"].items()},
        "comparator": None if com[c] is None else {**com[c], "cost": finite(com[c]["cost"])}} for c in HEADLINE}
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--B", type=int, default=B_DEFAULT)
    ap.add_argument("--arm", default="plain", choices=["plain", "rand"], help="plain = H4; rand = descriptive")
    a = ap.parse_args(argv)
    th = thresholds()
    runs = [f"{a.arm}_s{k}" for k in range(10)]
    asel = struct_tables(a.arm, SEL, "SEL")
    csel = tables(rows("classical__SEL__grid24__default__fallback.csv"), HEADLINE, SEL)
    rsel = tables(rows("rule__SEL__grid24__default__fallback.csv"), HEADLINE, SEL)
    out = {"hypothesis": "H4" if a.arm == "plain" else "descriptive (STRUCT-rand, custom simulator)", "arm": a.arm,
           "margin": MARGIN, "tiers": {}}
    for tier, seeds in TIERS.items():
        aev = struct_tables(a.arm, seeds, "EVAL")
        cev = tables(rows("classical__EVAL__grid24__default__fallback.csv"), HEADLINE, seeds)
        rev = tables(rows("rule__EVAL__grid24__default__fallback.csv"), HEADLINE, seeds)
        rng = np.random.default_rng(RNG)
        out["tiers"][tier] = {"vs_adaptive_classical": analyse_tier(runs, asel, aev, csel, cev, th, len(seeds), a.B, rng),
                              "vs_best_rule_descriptive": analyse_tier(runs, asel, aev, rsel, rev, th, len(seeds), a.B,
                                                                       np.random.default_rng(RNG))}
        # AI-HARP reference on the same tier from the committed revision-5 rows (descriptive)
        ref = [f"ref_s{k}" for k in range(10)]
        hsel = agent_tables(ref, HEADLINE, SEL, ["agent__SEL__grid24__default__fallback.csv",
                                                 "agent__SEL__headline__default__fallback__s5_9.csv"])
        hev = agent_tables(ref, HEADLINE, seeds, ["agent__EVAL__grid24__default__fallback.csv",
                                                  "agent__EVAL__headline__default__fallback__s5_9.csv"])
        pol = {c: policy_eval(hsel, hev, list(range(10)), ref, c, *th[c], math.inf, None, None) for c in HEADLINE}
        out["tiers"][tier]["ai_harp_reference_descriptive"] = {
            f"{c[0]} d={c[1]:g}": {"qualifies": pol[c]["qualifies"], "cost": finite(pol[c]["cost"]),
                                   "n_pick": pol[c]["n_pick"]} for c in HEADLINE}
    from experiments.revision1.provenance import stamp, write_json
    (R8 / "analysis").mkdir(parents=True, exist_ok=True)
    p = R8 / "analysis" / ("h4.json" if a.arm == "plain" else "h4_rand_descriptive.json")
    write_json(p, {"provenance": stamp(f"revision8/analysis/{p.name}", {"B": a.B, "arm": a.arm},
                                       seeds={"SEL": SEL, **TIERS}), **out})
    print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
