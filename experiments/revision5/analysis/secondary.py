"""Revision-5 secondary (exploratory) results; no Holm. Writes results/revision5/analysis/secondary.json.

    python -m experiments.revision5.analysis.secondary [--B 2000] [--only ladder,variants,...]

Sections (each "pending" until its committed rows are complete):
  ladder        full model (A) vs rung B (oracle reward) vs rung C (oracle input, non-deployable), seeds 0-4
  variants      full model vs GCN, MLP, no-relevance encoders, seeds 0-4
  gate          learned gate (tau 0.5) vs ungated (tau 0) vs random gate at the learned gate's SEL rate, seeds 0-9
  budget        full model seeds 0-2 at 1,000 / 1,200 / 1,500 / 1,700 / 2,000 updates (D4)
  sumo          policy seeds 0-9 vs adaptive classical and best rule on SUMO mobility (headline cells)
  stress        24 cells on the calibrated channel with its own thresholds (D5): adaptive classical, best rule,
                policy seeds 0-4
  interp        realised (SEL->EVAL selection) vs interpolated (Algorithm 1 crossing on EVAL) costs
  sensitivity   the 18-setting grid: q* factor {0.90, 0.95, 0.98} x miss tolerance {0.02, 0.05, 0.10} x interpolation
                {on, off}, with the revision-2 change flags
Every method/cell entry carries the primary safety columns of its pick on EVAL: actionable miss (and coverage =
1 - miss), stop_deadline_warned_share_actionable, p95 TIR, median TIR (labelled), and qualification under the
absolute miss cap 0.40.
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np

from analysis.pareto import OperatingPoint, PolicyCurve, POLICY_SWEEPS
from experiments.revision5.analysis.common import (CAL, D_R, EVAL, EXTRA_COLS, GRID24, HEADLINE, RUNS, SEED_RNG, SEL,
                                                  cal_thresholds, finite, rows, select, tables, thresholds, write)
from experiments.revision5.analysis.h2 import agent_tables

FAMILIES = list(POLICY_SWEEPS)
CK = lambda c: f"{c[0]} d={c[1]:g}"
SAFETY = ("miss", "stop_deadline_warned_share_actionable", "tir95", "tir_median_s")


def complete(name: str, n: int) -> bool:
    p = RUNS / name
    return p.exists() and sum(1 for _ in p.open(encoding="utf-8")) - 1 >= n


def safety(t, j, ei=None) -> dict:
    d = {k: finite(t.mean(k, ei)[j]) for k in SAFETY if k in t.m}
    d["actionable_coverage"] = None if d.get("miss") is None else 1 - d["miss"]
    return d


def single(sel, ev, c, q, m, D=math.inf) -> dict | None:
    """A single-setting method (adaptive classical / rule): pick on SEL, EVAL cost, qualification, safety."""
    k = select(sel[c], q, m, D)
    if k < 0:
        return None
    j = ev[c].settings.index(sel[c].settings[k])
    rw, mi, co = (float(ev[c].mean(x)[j]) for x in ("rwcr", "miss", "cost"))
    return {"pick": [str(x) for x in sel[c].settings[k]], "cost": finite(co), "rwcr": rw,
            "qualifies": bool(rw >= q and mi <= m), "qualifies_abs_cap_0.40": bool(rw >= q and mi <= 0.40),
            **safety(ev[c], j)}


def learned(asel, aev, runs, c, q, m, B, rng, D=math.inf) -> dict:
    """A learned method over training seeds: per-seed picks; cell aggregate as in H2; bootstrap over training x EVAL
    seeds (selections recomputed on resampled SEL) for the mean cost."""
    per = {}
    for r in runs:
        k = select(asel[r][c], q, m, D)
        if k < 0:
            per[r] = None
            continue
        j = aev[r][c].settings.index(asel[r][c].settings[k])
        rw, mi, co = (float(aev[r][c].mean(x)[j]) for x in ("rwcr", "miss", "cost"))
        per[r] = {"bias": asel[r][c].settings[k][0], "cost": finite(co), "rwcr": rw, "miss": mi,
                  "qualifies": bool(rw >= q and mi <= m), "qualifies_abs_cap_0.40": bool(rw >= q and mi <= 0.40),
                  **safety(aev[r][c], j)}
    picks = [v for v in per.values() if v is not None]
    costs = [v["cost"] if v["cost"] is not None else math.inf for v in picks]
    agg = {"n_seeds": len(runs), "n_pick": len(picks), "n_qualify": sum(v["qualifies"] for v in picks),
           "qualifies": False, "cost_mean": None, "cost_sd": None}
    if picks and 2 * len(picks) >= len(runs):
        rw, mi = np.mean([v["rwcr"] for v in picks]), np.mean([v["miss"] for v in picks])
        agg.update(qualifies=bool(rw >= q and mi <= m), cost_mean=finite(np.mean(costs)),
                   cost_sd=finite(np.std(costs, ddof=1)) if len(costs) > 1 else None)
    boots = []
    for _ in range(B):
        rr = [runs[i] for i in rng.integers(0, len(runs), len(runs))]
        si, ei = rng.integers(0, len(SEL), len(SEL)), rng.integers(0, len(EVAL), len(EVAL))
        cs = []
        for r in rr:
            k = select(asel[r][c], q, m, D, si)
            if k >= 0:
                j = aev[r][c].settings.index(asel[r][c].settings[k])
                cs.append(float(aev[r][c].mean("cost", ei)[j]))
        if cs and 2 * len(cs) >= len(rr):
            boots.append(float(np.mean(cs)))
    if boots:
        b = np.array(boots)
        agg["cost_mean_ci95"] = [finite(np.quantile(b, 0.025, method="inverted_cdf")),
                                 finite(np.quantile(b, 0.975, method="inverted_cdf"))]
        agg["bootstrap_defined"] = len(boots)
    return {"aggregate": agg, "per_seed": per}


def classical_ctx(cells, channel="default", backend="fallback", D=math.inf):
    th = thresholds() if channel == "default" else stress_thresholds()
    tag = f"{channel}__{backend}"
    cs = tables(rows(f"classical__SEL__{'grid24' if backend == 'fallback' else 'headline'}__{tag}.csv"), cells, SEL, extra=EXTRA_COLS)
    ce = tables(rows(f"classical__EVAL__{'grid24' if backend == 'fallback' else 'headline'}__{tag}.csv"), cells, EVAL, extra=EXTRA_COLS)
    rs = tables(rows(f"rule__SEL__{'grid24' if backend == 'fallback' else 'headline'}__{tag}.csv"), cells, SEL, extra=EXTRA_COLS)
    re = tables(rows(f"rule__EVAL__{'grid24' if backend == 'fallback' else 'headline'}__{tag}.csv"), cells, EVAL, extra=EXTRA_COLS)
    return th, cs, ce, rs, re


def stress_thresholds():
    p = RUNS.parent / "thresholds_stress.json"
    th = json.loads(p.read_text())["cells"]
    return {(v["scenario"], float(v["density"])): (v["q_star"], v["m_star"]) for v in th.values()}


def oracle_hindsight(ev, c, q, m):
    """EVAL-hindsight optimistic benchmark: the cheapest classical setting qualifying on EVAL itself."""
    k = select(ev[c], q, m, math.inf)
    return None if k < 0 else {"pick": [str(x) for x in ev[c].settings[k]], "cost": finite(ev[c].mean("cost")[k])}


def compare(methods: dict, cells, th, cs, ce, rs, re, B, rng, focus=("rural_highway", 20.0)):
    """methods: {name: (asel, aev, runs)}. Per cell: adaptive classical, best rule, EVAL-hindsight oracle, each
    learned method; rural d=20 margin (method/adaptive - 1) and regret (adaptive/oracle - 1)."""
    out = {}
    for c in cells:
        q, m = th[c]
        ada, rul, orc = single(cs, ce, c, q, m), single(rs, re, c, q, m), oracle_hindsight(ce, c, q, m)
        cell = {"q_star": q, "m_star": m, "adaptive_classical": ada, "best_rule": rul, "eval_hindsight_oracle": orc,
                "methods": {n: learned(a, e, r, c, q, m, B, rng) for n, (a, e, r) in methods.items()}}
        out[CK(c)] = cell
    f = out.get(CK(focus))
    if f:
        a = f["adaptive_classical"]
        o = f["eval_hindsight_oracle"]
        f["rural_d20_margin_vs_adaptive"] = {
            n: (None if (v["aggregate"]["cost_mean"] is None or not a or not a["cost"]) else v["aggregate"]["cost_mean"] / a["cost"] - 1)
            for n, v in f["methods"].items()}
        f["rural_d20_regret_adaptive_vs_oracle"] = (None if not (a and o and a["cost"] and o["cost"]) else a["cost"] / o["cost"] - 1)
    reached = {n: sum(1 for c in cells if out[CK(c)]["methods"][n]["aggregate"]["qualifies"]) for n in methods}
    return {"cells": out, "cells_reached": reached}


def interp_cost(t, q, m, idx_mask=None) -> float:
    """Algorithm 1 on EVAL means: interpolated crossing at q* among settings with mean miss <= m (realised anchor)."""
    rw, co, mi = t.mean("rwcr"), t.mean("cost"), t.mean("miss")
    sel = range(len(t.settings)) if idx_mask is None else np.flatnonzero(idx_mask)
    pts = [OperatingPoint("x", "", i, float(rw[i]), 0.0, float(co[i]), 0.0, 0, {"miss": float(mi[i])}) for i in sel]
    return PolicyCurve("x", pts).overhead_at(q, eligible=lambda p: p.extras["miss"] <= m)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--only", default="")
    a = ap.parse_args(argv)
    only = set(a.only.split(",")) - {""}
    want = lambda s: not only or s in only
    rng = np.random.default_rng(SEED_RNG)
    out_path = RUNS.parent / "analysis" / "secondary.json"
    res = json.loads(out_path.read_text()) if out_path.exists() else {}
    res.pop("provenance", None)
    H = HEADLINE
    sel_ref = ["agent__SEL__grid24__default__fallback.csv", "agent__SEL__headline__default__fallback__s5_9.csv"]
    ev_ref = ["agent__EVAL__grid24__default__fallback.csv", "agent__EVAL__headline__default__fallback__s5_9.csv"]
    R5 = [f"ref_s{k}" for k in range(5)]
    R10 = [f"ref_s{k}" for k in range(10)]

    def ag(runs, sel_files, ev_files, variant="learned", cells=H):
        return (agent_tables(runs, cells, SEL, sel_files, variant, EXTRA_COLS),
                agent_tables(runs, cells, EVAL, ev_files, variant, EXTRA_COLS), runs)

    th, cs, ce, rs, re = classical_ctx(H)

    if want("ladder"):
        if complete("agent__EVAL__headline__default__fallback__ladderB.csv", 3600) and complete("agent__EVAL__headline__default__fallback__ladderC.csv", 3600):
            m = {"A_full_model": ag(R5, sel_ref[:1], ev_ref[:1]),
                 "B_oracle_reward": ag([f"rungB_s{k}" for k in range(5)], ["agent__SEL__headline__default__fallback__ladderB.csv"], ["agent__EVAL__headline__default__fallback__ladderB.csv"]),
                 "C_oracle_input_nondeployable": ag([f"rungC_s{k}" for k in range(5)], ["agent__SEL__headline__default__fallback__ladderC.csv"], ["agent__EVAL__headline__default__fallback__ladderC.csv"])}
            res["ladder"] = compare(m, H, th, cs, ce, rs, re, a.B, rng)
        else:
            res["ladder"] = {"status": "pending: ladder rows incomplete"}
    if want("variants"):
        if complete("agent__EVAL__headline__default__fallback__variants.csv", 10800):
            vf = (["agent__SEL__headline__default__fallback__variants.csv"], ["agent__EVAL__headline__default__fallback__variants.csv"])
            m = {"full_model": ag(R5, sel_ref[:1], ev_ref[:1]),
                 **{v: ag([f"{v}_s{k}" for k in range(5)], *vf) for v in ("gcn", "mlp", "norel")}}
            res["variants"] = compare(m, H, th, cs, ce, rs, re, a.B, rng)
        else:
            res["variants"] = {"status": "pending: variant rows incomplete"}
    if want("gate"):
        if complete("agent__EVAL__headline__default__fallback__tau0.csv", 7200) and complete("agent__EVAL__headline__default__fallback__rgate.csv", 7200):
            m = {"learned_gate_tau0.5": ag(R10, sel_ref, ev_ref),
                 "ungated_tau0": ag(R10, ["agent__SEL__headline__default__fallback__tau0.csv"], ["agent__EVAL__headline__default__fallback__tau0.csv"], "tau0"),
                 "random_gate_matched_rate": ag(R10, ["agent__SEL__headline__default__fallback__rgate.csv"], ["agent__EVAL__headline__default__fallback__rgate.csv"], "rgate")}
            res["gate"] = compare(m, H, th, cs, ce, rs, re, a.B, rng)
        else:
            res["gate"] = {"status": "pending: gate rows incomplete"}
    if want("budget"):
        if complete("agent__EVAL__headline__default__fallback__budget.csv", 8640):
            bf = (["agent__SEL__headline__default__fallback__budget.csv"], ["agent__EVAL__headline__default__fallback__budget.csv"])
            m = {"u1000": ag(["ref_s0", "ref_s1", "ref_s2"], sel_ref[:1], ev_ref[:1]),
                 **{f"u{u}": ag([f"ref_s{k}_u2000@{u}" for k in range(3)], *bf) for u in (1200, 1500, 1700, 2000)}}
            res["budget"] = compare(m, H, th, cs, ce, rs, re, a.B, rng)
        else:
            res["budget"] = {"status": "pending: budget rows incomplete"}
    if want("sumo"):
        files = ("classical__SEL__headline__default__sumo.csv", 4400), ("rule__SEL__headline__default__sumo.csv", 3120), \
                ("rule__EVAL__headline__default__sumo.csv", 4680), ("agent__EVAL__headline__default__sumo.csv", 7200)
        if all(complete(f, n) for f, n in files):
            ths, css, ces, rss, res_ = classical_ctx(H, backend="sumo")
            m = {"policy": ag(R10, ["agent__SEL__headline__default__sumo.csv"], ["agent__EVAL__headline__default__sumo.csv"])}
            res["sumo"] = {"note": "frozen default-channel CAL thresholds (no SUMO CAL); exploratory",
                           **compare(m, H, ths, css, ces, rss, res_, a.B, rng)}
        else:
            res["sumo"] = {"status": "pending: SUMO rows incomplete"}
    if want("stress"):
        files = ("classical__EVAL__grid24__calibrated__fallback.csv", 39600), ("rule__EVAL__grid24__calibrated__fallback.csv", 28080), \
                ("agent__EVAL__grid24__calibrated__fallback.csv", 21600)
        if all(complete(f, n) for f, n in files):
            tht, cst, cet, rst, ret = classical_ctx(GRID24, channel="calibrated")
            m = {"policy": ag(R5, ["agent__SEL__grid24__calibrated__fallback.csv"], ["agent__EVAL__grid24__calibrated__fallback.csv"], cells=GRID24)}
            res["stress"] = compare(m, GRID24, tht, cst, cet, rst, ret, a.B, rng)
        else:
            res["stress"] = {"status": "pending: stress-test rows incomplete"}
    if want("interp"):
        res["interp"] = interp_table(H, th, cs, ce, rs, re)
    if want("sensitivity"):
        res["sensitivity"] = sensitivity(H, cs, ce)
    print(write("secondary.json", res, {"B": a.B, "sections": sorted(only) or "all"}))
    for k, v in res.items():
        print(k, v.get("status", "done") if isinstance(v, dict) else "done")
    return 0


def interp_table(H, th, cs, ce, rs, re) -> dict:
    """With/without interpolation, per cell and method family (classical families, rule, policy seeds 0-9)."""
    asel = agent_tables([f"ref_s{k}" for k in range(10)], H, SEL, ["agent__SEL__grid24__default__fallback.csv", "agent__SEL__headline__default__fallback__s5_9.csv"])
    aev = agent_tables([f"ref_s{k}" for k in range(10)], H, EVAL, ["agent__EVAL__grid24__default__fallback.csv", "agent__EVAL__headline__default__fallback__s5_9.csv"])
    out = {}
    for c in H:
        q, m = th[c]
        fam = {}
        for f in FAMILIES:
            mask = np.array([k[0] == f for k in cs[c].settings])
            k = select(cs[c], q, m, math.inf, None, mask)
            real = None if k < 0 else finite(ce[c].mean("cost")[ce[c].settings.index(cs[c].settings[k])])
            fam[f] = {"realised_SEL_to_EVAL": real,
                      "interpolated_EVAL_estimate": finite(interp_cost(ce[c], q, m, np.array([k2[0] == f for k2 in ce[c].settings])))}
        pol = {}
        for r in asel:
            k = select(asel[r][c], q, m, math.inf)
            real = None if k < 0 else finite(aev[r][c].mean("cost")[aev[r][c].settings.index(asel[r][c].settings[k])])
            pol[r] = {"realised_SEL_to_EVAL": real, "interpolated_EVAL_estimate": finite(interp_cost(aev[r][c], q, m))}
        k = select(rs[c], q, m, math.inf)
        rule = {"realised_SEL_to_EVAL": None if k < 0 else finite(re[c].mean("cost")[re[c].settings.index(rs[c].settings[k])]),
                "interpolated_EVAL_estimate": finite(interp_cost(re[c], q, m))}
        out[CK(c)] = {"classical_families": fam, "policy_seeds": pol, "rule": rule,
                      "label": "interpolated = estimate, not an implementable configuration"}
    return out


def sensitivity(H, cs, ce) -> dict:
    """18 settings: q* factor x miss tolerance x interpolation; thresholds recomputed from CAL for each. Flags as in
    revision 2: oracle family, adaptive pick (family), cells each policy seed qualifies in, sign of the rural d=20 margin."""
    cal = tables(rows("classical__CAL__grid24__default__fallback.csv"), H, CAL)
    runs = [f"ref_s{k}" for k in range(10)]
    asel = agent_tables(runs, H, SEL, ["agent__SEL__grid24__default__fallback.csv", "agent__SEL__headline__default__fallback__s5_9.csv"])
    aev = agent_tables(runs, H, EVAL, ["agent__EVAL__grid24__default__fallback.csv", "agent__EVAL__headline__default__fallback__s5_9.csv"])
    out = {}
    for fac in (0.90, 0.95, 0.98):
        for tol in (0.02, 0.05, 0.10):
            for interp in (False, True):
                cellres = {}
                for c in H:
                    rw = cal[c].mean("rwcr")
                    q, m = fac * float(np.nanmax(rw)), float(np.nanmin(cal[c].mean("miss"))) + tol
                    if interp:
                        ada = min((interp_cost(ce[c], q, m, np.array([k[0] == f for k in ce[c].settings])), f) for f in FAMILIES)
                        ada_cost, ada_fam = ada if math.isfinite(ada[0]) else (math.inf, None)
                        pol = {r: interp_cost(aev[r][c], q, m) for r in runs}
                        pol_q = {r: math.isfinite(v) for r, v in pol.items()}
                    else:
                        k = select(cs[c], q, m, math.inf)
                        if k < 0:
                            ada_cost, ada_fam = math.inf, None
                        else:
                            j = ce[c].settings.index(cs[c].settings[k])
                            ok = ce[c].mean("rwcr")[j] >= q and ce[c].mean("miss")[j] <= m
                            ada_cost, ada_fam = (float(ce[c].mean("cost")[j]), cs[c].settings[k][0]) if ok else (math.inf, None)
                        pol, pol_q = {}, {}
                        for r in runs:
                            k = select(asel[r][c], q, m, math.inf)
                            if k < 0:
                                pol[r], pol_q[r] = math.inf, False
                                continue
                            j = aev[r][c].settings.index(asel[r][c].settings[k])
                            pol[r] = float(aev[r][c].mean("cost")[j])
                            pol_q[r] = bool(aev[r][c].mean("rwcr")[j] >= q and aev[r][c].mean("miss")[j] <= m)
                    orc = oracle_hindsight(ce, c, q, m)
                    fin = [v for v in pol.values() if math.isfinite(v)]
                    pmean = float(np.mean(fin)) if fin and 2 * len(fin) >= len(runs) else math.inf
                    cellres[CK(c)] = {"q_star": q, "m_star": m, "adaptive_family": ada_fam, "adaptive_cost": finite(ada_cost),
                                      "oracle_family": None if orc is None else orc["pick"][0],
                                      "policy_seeds_qualifying": sorted(r for r, v in pol_q.items() if v),
                                      "policy_mean_cost": finite(pmean),
                                      "margin": finite(pmean / ada_cost - 1) if math.isfinite(pmean) and math.isfinite(ada_cost) and ada_cost else None}
                out[f"q{fac:.2f}_tol{tol:.2f}_{'interp' if interp else 'realised'}"] = cellres
    base = out["q0.95_tol0.05_realised"]
    flags = {}
    for name, cr in out.items():
        f = []
        for c, v in cr.items():
            b = base[c]
            if v["oracle_family"] != b["oracle_family"]:
                f.append(f"{c}: oracle family {b['oracle_family']} -> {v['oracle_family']}")
            if v["adaptive_family"] != b["adaptive_family"]:
                f.append(f"{c}: adaptive pick {b['adaptive_family']} -> {v['adaptive_family']}")
            if v["policy_seeds_qualifying"] != b["policy_seeds_qualifying"]:
                f.append(f"{c}: qualifying policy seeds changed")
        r20 = cr[CK(("rural_highway", 20.0))]["margin"]
        b20 = base[CK(("rural_highway", 20.0))]["margin"]
        if r20 is not None and b20 is not None and (r20 < 0) != (b20 < 0):
            f.append(f"rural d=20 margin sign {b20:+.3f} -> {r20:+.3f}")
        flags[name] = f
    return {"settings": out, "baseline": "q0.95_tol0.05_realised", "flags_vs_baseline": flags}


if __name__ == "__main__":
    raise SystemExit(main())
