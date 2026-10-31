"""H1 -- no single classical family is the cheapest qualifying family in all four headline cells (PREREGISTRATION,
ANALYSIS_SPEC). Writes results/revision5/analysis/h1.json.

    python -m experiments.revision5.analysis.h1 [--B 10000] [--D inf|Dr]
"""

from __future__ import annotations

import argparse
import math

import numpy as np

from analysis.pareto import POLICY_SWEEPS
from experiments.revision5.analysis.common import (B, CAL, D_R, EVAL, HEADLINE, SEED_RNG, SEL, Table, binom_lower_p,
                                                  cal_thresholds, clopper_pearson, finite, rows, select, tables,
                                                  write)

FAMILIES = list(POLICY_SWEEPS)


def family_masks(t: Table) -> dict[str, np.ndarray]:
    pol = np.array([k[0] for k in t.settings])
    return {f: pol == f for f in FAMILIES}


def winners(cal: dict, sel: dict, ev: dict, cells, D_of, ci=None, si=None, ei=None) -> dict:
    """Per cell: q*, m*, each family's pick, its EVAL qualification/cost, and the winner (or None)."""
    out = {}
    for c in cells:
        q, m = cal_thresholds(cal[c], ci)
        masks = family_masks(sel[c])
        fam = {}
        for f, mask in masks.items():
            k = select(sel[c], q, m, D_of(c), si, mask)
            if k < 0:
                fam[f] = {"pick": None, "qualifies": False, "cost": math.inf}
                continue
            key = sel[c].settings[k]
            j = ev[c].settings.index(key)
            rw, mi, co = (float(ev[c].mean(x, ei)[j]) for x in ("rwcr", "miss", "cost"))
            fam[f] = {"pick": list(key), "qualifies": bool(rw >= q and mi <= m), "cost": co, "eval_rwcr": rw,
                      "eval_miss": mi}
        qual = {f: v["cost"] for f, v in fam.items() if v["qualifies"]}
        win = min(qual, key=lambda f: (qual[f], FAMILIES.index(f))) if qual else None
        out[c] = {"q_star": q, "m_star": m, "families": fam, "winner": win}
    return out


def load(D: str):
    cells = HEADLINE
    cal = tables(rows("classical__CAL__grid24__default__fallback.csv"), cells, CAL)
    sel = tables(rows("classical__SEL__grid24__default__fallback.csv"), cells, SEL)
    ev = tables(rows("classical__EVAL__grid24__default__fallback.csv"), cells, EVAL)
    D_of = (lambda c: math.inf) if D == "inf" else (lambda c: D_R[c[0]])
    return cells, cal, sel, ev, D_of


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--B", type=int, default=B)
    ap.add_argument("--D", default="inf", choices=["inf", "Dr"])
    a = ap.parse_args(argv)
    cells, cal, sel, ev, D_of = load(a.D)
    point = winners(cal, sel, ev, cells, D_of)
    rng = np.random.default_rng(SEED_RNG)
    one = 0
    win_counts: dict[str, int] = {}
    for _ in range(a.B):
        ci, si, ei = (rng.integers(0, n, n) for n in (len(CAL), len(SEL), len(EVAL)))
        w = winners(cal, sel, ev, cells, D_of, ci, si, ei)
        ws = [w[c]["winner"] for c in cells]
        if ws[0] is not None and all(x == ws[0] for x in ws):
            one += 1
            win_counts[ws[0]] = win_counts.get(ws[0], 0) + 1
    lo, hi = clopper_pearson(one, a.B)
    p = binom_lower_p(one, a.B, 0.05)
    pw = [point[c]["winner"] for c in cells]
    payload = {
        "hypothesis": "H1: no single classical scheme family is the cheapest qualifying family in all four headline cells",
        "D": a.D, "B": a.B, "rng_seed": SEED_RNG,
        "share_one_family_wins_all_four": one / a.B, "count": one,
        "clopper_pearson_95": [lo, hi], "binomial_p_lower_vs_0.05": p,
        "supported": bool(hi < 0.05),
        "families_that_won_all_four_in_replicates": win_counts,
        "point_estimate": {f"{c[0]} d={c[1]:g}": {"winner": point[c]["winner"], "q_star": point[c]["q_star"],
                                                   "m_star": point[c]["m_star"],
                                                   "families": {f: {**v, "cost": finite(v["cost"])}
                                                                for f, v in point[c]["families"].items()}}
                           for c in cells},
        "point_one_family_wins_all_four": bool(pw[0] is not None and all(x == pw[0] for x in pw)),
    }
    name = "h1.json" if a.D == "inf" else "h1_Dr.json"
    print(write(name, payload, {"B": a.B, "D": a.D}, seeds={"CAL": CAL, "SEL": SEL, "EVAL": EVAL}))
    print({k: payload[k] for k in ("share_one_family_wins_all_four", "clopper_pearson_95", "binomial_p_lower_vs_0.05",
                                   "supported", "point_one_family_wins_all_four")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
