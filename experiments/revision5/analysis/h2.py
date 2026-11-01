"""H2 -- policy vs retuned classical (and vs the best rule baseline), deployable on both sides (PREREGISTRATION,
ANALYSIS_SPEC). Writes results/revision5/analysis/h2.json (headline, seeds 0-9; primary) and h2_grid24.json
(24 cells, seeds 0-4; secondary).

    python -m experiments.revision5.analysis.h2 [--B 10000] [--D inf|Dr] [--grid24]
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np

from common.config import PROJECT_ROOT
from experiments.evaluate_agent import checkpoint_sha
from experiments.revision5.analysis.common import (B, D_R, EVAL, GRID24, HEADLINE, SEED_RNG, SEL, finite, newcombe,
                                                  rows, select, tables, thresholds, wilson, write)
from experiments.revision5.sweep import checkpoint_for


def run_sha(run: str) -> str:
    """checkpoint_sha of a training run: from its checkpoint file if present (it must match the committed map), else
    from the committed map results/revision5/checkpoint_shas.json (checkpoints are not committed)."""
    m = json.loads((PROJECT_ROOT / "results" / "revision5" / "checkpoint_shas.json").read_text(encoding="utf-8"))["shas"]
    p = checkpoint_for(run)
    if p.exists():
        s = checkpoint_sha(p)
        if run in m and m[run] != s:
            raise ValueError(f"{run}: checkpoint sha {s} differs from the committed map ({m[run]})")
        return s
    return m[run]


def agent_tables(runs: list[str], cells, seeds, files: list[str], variant: str = "learned",
                 extra=()) -> dict[str, dict]:
    """{run: {cell: Table over biases}} for each training seed's rows of one gate variant:
    "learned" (tau 0.5), "tau0" (ungated) or "rgate" (random gate at the learned gate's SEL rate)."""
    sha = {run_sha(r): r for r in runs}
    allrows = [r for f in files for r in rows(f)]
    out = {}
    for s, run in sha.items():
        def where(r, s=s):
            q = json.loads(r["params_json"])
            if q.get("checkpoint_sha") != s:
                return False
            if variant == "rgate":
                return "random_gate_rate" in q
            return "random_gate_rate" not in q and float(q.get("tau", -1)) == (0.0 if variant == "tau0" else 0.5)
        out[run] = tables(allrows, cells, seeds, key=lambda r: (float(r["value"]),), where=where, extra=extra)
    return out


def method_eval(sel, ev, c, q, m, D, si, ei):
    """A single-setting method (adaptive classical / rule): its pick, EVAL cost and qualification."""
    k = select(sel[c], q, m, D, si)
    if k < 0:
        return None
    j = ev[c].settings.index(sel[c].settings[k])
    rw, mi, co = (float(ev[c].mean(x, ei)[j]) for x in ("rwcr", "miss", "cost"))
    return {"pick": list(sel[c].settings[k]), "cost": co, "rwcr": rw, "miss": mi, "qualifies": bool(rw >= q and mi <= m)}


def policy_eval(asel, aev, runs_idx, runs, c, q, m, D, si, ei):
    """Policy in a cell: every (resampled) training seed picks its bias on SEL; cost = mean over seeds with a pick."""
    per = {}
    for r in set(runs[i] for i in runs_idx):
        k = select(asel[r][c], q, m, D, si)
        if k < 0:
            per[r] = None
            continue
        j = aev[r][c].settings.index(asel[r][c].settings[k])
        rw, mi, co = (float(aev[r][c].mean(x, ei)[j]) for x in ("rwcr", "miss", "cost"))
        per[r] = {"bias": asel[r][c].settings[k][0], "cost": co, "rwcr": rw, "miss": mi,
                  "qualifies": bool(rw >= q and mi <= m)}
    picks = [per[runs[i]] for i in runs_idx if per[runs[i]] is not None]
    if len(picks) * 2 < len(runs_idx):
        return {"per_seed": per, "qualifies": False, "cost": math.inf, "n_pick": len(picks)}
    rw = float(np.mean([p["rwcr"] for p in picks]))
    mi = float(np.mean([p["miss"] for p in picks]))
    co = float(np.mean([p["cost"] for p in picks]))
    return {"per_seed": per, "qualifies": bool(rw >= q and mi <= m), "cost": co, "rwcr": rw, "miss": mi,
            "n_pick": len(picks)}


def statistic(cells, comp, pol) -> tuple[float, dict]:
    per = {}
    for c in cells:
        a, p = comp[c], pol[c]
        if a is not None and a["qualifies"] and p["qualifies"] and math.isfinite(a["cost"]) and a["cost"] > 0:
            per[c] = p["cost"] / a["cost"] - 1
    return (float(np.mean(list(per.values()))) if per else float("nan")), per


def analyse(cells, runs, asel, aev, csel, cev, rsel, rev, th, D_of, nB, rng):
    R = list(range(len(runs)))
    full_si, full_ei = None, None

    def one(ri, si, ei):
        pol = {c: policy_eval(asel, aev, ri, runs, c, *th[c], D_of(c), si, ei) for c in cells}
        ada = {c: method_eval(csel, cev, c, *th[c], D_of(c), si, ei) for c in cells}
        rul = {c: method_eval(rsel, rev, c, *th[c], D_of(c), si, ei) for c in cells}
        return pol, ada, rul

    pol, ada, rul = one(R, full_si, full_ei)
    T_ada, per_ada = statistic(cells, ada, pol)
    T_rul, per_rul = statistic(cells, rul, pol)
    boots = {"adaptive": [], "rule": []}
    for _ in range(nB):
        ri = list(rng.integers(0, len(runs), len(runs)))
        si, ei = rng.integers(0, len(SEL), len(SEL)), rng.integers(0, len(EVAL), len(EVAL))
        p, a, r = one(ri, si, ei)
        boots["adaptive"].append(statistic(cells, a, p)[0])
        boots["rule"].append(statistic(cells, r, p)[0])

    def summary(T, per, bs, comp):
        bs = np.array(bs, dtype=float)
        # NaN = no cell where both qualify (statistic undefined): excluded and counted. +inf = the policy's mean cost
        # is infinite in a cell (some EVAL seeds inform nobody at risk): KEPT, as the largest values.
        ok = bs[~np.isnan(bs)]
        fin = {c: v for c, v in per.items() if math.isfinite(v)}
        res = {"statistic_pooled_relative_cost_difference": T if math.isfinite(T) else ("inf" if T > 0 else None),
               "per_cell": {f"{c[0]} d={c[1]:g}": (v if math.isfinite(v) else "inf") for c, v in per.items()},
               "nonfinite_cells": [f"{c[0]} d={c[1]:g}" for c in per if c not in fin],
               "sensitivity_excluding_nonfinite_cells": (float(np.mean(list(fin.values()))) if fin else None),
               "cells_both_qualify": len(per), "B": nB,
               "bootstrap_undefined_replicates_nan": int(np.isnan(bs).sum()),
               "bootstrap_infinite_replicates": int(np.isinf(bs).sum())}
        q = lambda x: float(np.quantile(ok, x, method="inverted_cdf"))
        if ok.size:
            lo, hi = q(0.025), q(0.975)
            lo90, hi90 = q(0.05), q(0.95)
            js = lambda x: x if math.isfinite(x) else ("inf" if x > 0 else "-inf")
            res.update(ci95=[js(lo), js(hi)], ci90=[js(lo90), js(hi90)],
                       p_two_sided=float(min(1.0, 2 * min((ok <= 0).mean(), (ok >= 0).mean()))),
                       P_policy_cheaper=float((ok < 0).mean()), superiority=bool(hi < 0),
                       inferiority=bool(lo > 0), equivalence_TOST_5pct=bool(lo90 > -0.05 and hi90 < 0.05))
        # qualification rates (point): policy over (cell, training seed) picks; comparator over cells
        kp = sum(1 for c in cells for v in pol[c]["per_seed"].values() if v is not None and v["qualifies"])
        np_ = len(cells) * len(runs)
        ka = sum(1 for c in cells if comp[c] is not None and comp[c]["qualifies"])
        res["qualification"] = {"policy": [kp, np_, wilson(kp, np_)], "comparator": [ka, len(cells), wilson(ka, len(cells))],
                                "difference_newcombe95": newcombe(kp, np_, ka, len(cells))}
        res["point_detail"] = {f"{c[0]} d={c[1]:g}": {
            "policy": {k: v for k, v in pol[c].items() if k != "per_seed"} | {"cost": finite(pol[c]["cost"])},
            "policy_per_seed": {r: (None if v is None else {**v, "cost": finite(v["cost"])})
                                for r, v in pol[c]["per_seed"].items()},
            "comparator": None if comp[c] is None else {**comp[c], "cost": finite(comp[c]["cost"])}} for c in cells}
        return res

    return {"vs_adaptive_classical": summary(T_ada, per_ada, boots["adaptive"], ada),
            "vs_best_rule": summary(T_rul, per_rul, boots["rule"], rul)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--B", type=int, default=B)
    ap.add_argument("--D", default="inf", choices=["inf", "Dr"])
    ap.add_argument("--grid24", action="store_true", help="secondary: 24 cells, training seeds 0-4")
    a = ap.parse_args(argv)
    cells = GRID24 if a.grid24 else HEADLINE
    runs = [f"ref_s{k}" for k in (range(5) if a.grid24 else range(10))]
    th = thresholds()
    D_of = (lambda c: math.inf) if a.D == "inf" else (lambda c: D_R[c[0]])
    asel = agent_tables(runs, cells, SEL, ["agent__SEL__grid24__default__fallback.csv"] +
                        ([] if a.grid24 else ["agent__SEL__headline__default__fallback__s5_9.csv"]))
    aev = agent_tables(runs, cells, EVAL, ["agent__EVAL__grid24__default__fallback.csv"] +
                       ([] if a.grid24 else ["agent__EVAL__headline__default__fallback__s5_9.csv"]))
    csel = tables(rows("classical__SEL__grid24__default__fallback.csv"), cells, SEL)
    cev = tables(rows("classical__EVAL__grid24__default__fallback.csv"), cells, EVAL)
    rsel = tables(rows("rule__SEL__grid24__default__fallback.csv"), cells, SEL)
    rev = tables(rows("rule__EVAL__grid24__default__fallback.csv"), cells, EVAL)
    res = analyse(cells, runs, asel, aev, csel, cev, rsel, rev, th, D_of, a.B, np.random.default_rng(SEED_RNG))
    payload = {"hypothesis": "H2: policy vs retuned (SEL-selected) adaptive classical, deployable on both sides"
                             + (" -- SECONDARY 24-cell version, seeds 0-4" if a.grid24 else ""),
               "D": a.D, "training_seeds": runs, **res}
    name = ("h2_grid24" if a.grid24 else "h2") + ("" if a.D == "inf" else "_Dr") + ".json"
    print(write(name, payload, {"B": a.B, "D": a.D, "grid24": a.grid24}, seeds={"SEL": SEL, "EVAL": EVAL}))
    for k in ("vs_adaptive_classical", "vs_best_rule"):
        r = res[k]
        print(k, {x: r.get(x) for x in ("statistic_pooled_relative_cost_difference", "ci95", "p_two_sided",
                                         "P_policy_cheaper", "cells_both_qualify", "superiority", "equivalence_TOST_5pct")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
