"""Revision-5 claims ledger (ANALYSIS_SPEC addendum 2026-10-31T12:05+05:30). Writes results/revision5/analysis/ledger.json.

    python -m experiments.revision5.analysis.ledger

Columns per claim: Set. (default channel, held-out SEL->EVAL), Cal. (24-cell stress test, own thresholds, D5), SUMO
(headline cells, SUMO mobility, frozen default CAL thresholds), Veins (H3 cross-fitted H1; classical claims only),
Sens. (the 18-setting grid), Seeds, Resampling/held-out, Verdict.
  stable         holds in Set. and in every applicable column (and >= 3 training seeds for a policy claim)
  sensitive      holds in Set. but fails, reverses or is undefined in at least one applicable column
  not supported  does not hold in Set. (including a non-significant directional result)
A column whose rows are incomplete reads "not tested". A column in which the claimed comparison is undefined (e.g. the
policy qualifies in no cell) counts as not holding: a stable verdict needs positive evidence in every column.
After DEVIATIONS D6 (H3 still failing) the Veins column is primary for classical claims and every custom-simulator
column is restricted to relative comparisons; the ledger records this, it does not change the verdict rule.
Every value is read from committed analysis outputs or recomputed (point estimates only) from committed rows.
"""

from __future__ import annotations

import json
import math

from experiments.revision5.analysis.common import CAL, EVAL, HEADLINE, OUT, SEL, rows, tables, write
from experiments.revision5.analysis.h1 import winners

CK = lambda c: f"{c[0]} d={c[1]:g}"  # noqa: E731
INF = lambda c: math.inf  # noqa: E731
POLICY_CELLS = {"rural_highway d=2": False, "rural_highway d=20": True, "rural_highway d=80": False,
                "urban_nlos d=20": True}


def _j(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def col(holds, text) -> dict:
    """holds: True / False / None (None = not applicable or not tested; the text says which)."""
    return {"holds": holds, "evidence": text}


def verdict(cols: dict, policy_seeds: int | None = None) -> str:
    if not cols["Set."]["holds"]:
        return "not supported"
    rest = [v for k, v in cols.items() if k != "Set." and v["holds"] is not None]
    if any(v["holds"] is False for v in rest):
        return "sensitive"
    if any(v["evidence"].startswith("not tested") for k, v in cols.items() if k != "Set."):
        return "sensitive"
    if policy_seeds is not None and policy_seeds < 3:
        return "sensitive"
    return "stable"


def point_winners(prefix_cal: str, prefix_sel: str, prefix_ev: str, cells) -> dict:
    cal = tables(rows(prefix_cal), cells, CAL)
    sel = tables(rows(prefix_sel), cells, SEL)
    ev = tables(rows(prefix_ev), cells, EVAL)
    return {CK(c): v["winner"] for c, v in winners(cal, sel, ev, cells, INF).items()}


def main(argv=None) -> int:
    h1, h2, h2g, h3, h3s, holm, sec = (_j(n) for n in ("h1.json", "h2.json", "h2_grid24.json", "h3.json",
                                                         "h3_subepoch.json", "holm.json", "secondary.json"))
    sens = sec["sensitivity"]["settings"]
    set_w = {c: v["winner"] for c, v in h1["point_estimate"].items()}
    cal_w = point_winners("classical__CAL__grid24__calibrated__fallback.csv", "classical__SEL__grid24__calibrated__fallback.csv",
                          "classical__EVAL__grid24__calibrated__fallback.csv", HEADLINE)
    sumo_w = point_winners("classical__CAL__grid24__default__fallback.csv", "classical__SEL__headline__default__sumo.csv",
                           "classical__EVAL__headline__default__sumo.csv", HEADLINE)
    veins_w = {m: h["crossfit_H1"]["veins"]["point_winners"] for m, h in (("epoch", h3), ("sub_epoch", h3s))}
    one = lambda w: (len(set(w.values())) == 1 and None not in w.values())  # noqa: E731

    # ---- classical group ----------------------------------------------------------------------------------------
    grid_one = [k for k, s in sens.items() if one({c: v["oracle_family"] for c, v in s.items()})]
    k1 = {
        "Set.": col(bool(h1["supported"]),
                    f"share of replicates with one winner in all four cells {h1['share_one_family_wins_all_four']} "
                    f"(CP95 {h1['clopper_pearson_95']}); Holm p {holm['holm_adjusted_p']['H1']:.3g}"),
        "Cal.": col(not one(cal_w), f"point winners on the calibrated channel: {cal_w}"),
        "SUMO": col(not one(sumo_w), f"point winners on SUMO mobility (default CAL thresholds): {sumo_w}"),
        "Veins": col(bool(h3["crossfit_H1"]["veins"]["supported"]),
                     f"cross-fitted H1 in Veins supported={h3['crossfit_H1']['veins']['supported']} "
                     f"(share {h3['crossfit_H1']['veins']['share_one_family_wins_all_four']}, "
                     f"CP95 {h3['crossfit_H1']['veins']['clopper_pearson_95']}); point winners {veins_w['epoch']}"),
        "Sens.": col(not grid_one, f"settings in which one EVAL-hindsight family is cheapest in all four cells: "
                                   f"{len(grid_one)} of {len(sens)} {grid_one}"),
        "Seeds": col(None, "n/a (classical)"),
        "Resampling/held-out": col(None, f"hierarchical bootstrap B={h1['B']} over CAL x SEL x EVAL seeds; selection on "
                                         "SEL, scoring on EVAL"),
    }
    k2_cols = {
        "Set.": col(all(set_w.values()), f"H1 point winners: {set_w}"),
        "Cal.": col(cal_w == set_w, f"calibrated channel: {cal_w}"),
        "SUMO": col(sumo_w == set_w, f"SUMO mobility: {sumo_w}"),
        "Veins": col(veins_w["epoch"] == set_w, f"Veins cross-fitted point winners: {veins_w['epoch']} "
                                                "(no qualifying family in three cells)"),
        "Sens.": col(None, "n/a (the grid reports EVAL-hindsight families, not the H1 winner)"),
        "Seeds": col(None, "n/a (classical)"),
        "Resampling/held-out": col(None, "point estimate (per-cell winner identity is not bootstrapped)"),
    }

    # ---- policy group -------------------------------------------------------------------------------------------
    va, vr = h2["vs_adaptive_classical"], h2["vs_best_rule"]
    ga, gr = h2g["vs_adaptive_classical"], h2g["vs_best_rule"]
    sumo = sec["sumo"]["cells"]
    sumo_reach = {c: v["methods"]["policy"]["aggregate"]["qualifies"] for c, v in sumo.items()}
    stress_reach = {c: sec["stress"]["cells"][c]["methods"]["policy"]["aggregate"]["qualifies"] for c in POLICY_CELLS}
    fmt = lambda x: "inf" if x == "inf" else f"{x:+.1%}"  # noqa: E731
    ci = lambda v: f"[{fmt(v['ci95'][0])}, {fmt(v['ci95'][1])}]" if v.get("ci95") else "undefined"  # noqa: E731

    def grid_margin_neg():
        ok = []
        for k, s in sens.items():
            m = [v["margin"] for v in s.values() if v["margin"] is not None]
            if m and all(x < 0 for x in m):
                ok.append(k)
        return ok

    def grid_reach():
        same = []
        for k, s in sens.items():
            r = {c: len(v["policy_seeds_qualifying"]) >= 5 for c, v in s.items()}
            if r == POLICY_CELLS:
                same.append(k)
        return same

    def stress_rel(key):
        """Calibrated stress test (D5), seeds 0-4: per cell where the policy aggregate and the comparator both qualify,
        policy mean cost / comparator cost - 1 (point estimates; the stress test has no pooled bootstrap)."""
        out = {}
        for c, v in sec["stress"]["cells"].items():
            a, p = v[key], v["methods"]["policy"]["aggregate"]
            if a and a["qualifies"] and a["cost"] and p["qualifies"]:
                out[c] = "inf" if p["cost_mean"] is None else p["cost_mean"] / a["cost"] - 1
        return out

    sa, sr = stress_rel("adaptive_classical"), stress_rel("best_rule")
    srt = lambda d: ", ".join(f"{c}: {fmt(x)}" for c, x in d.items())  # noqa: E731
    gm, gq = grid_margin_neg(), grid_reach()
    p1 = {
        "Set.": col(bool(va["superiority"]),
                    f"pooled relative cost difference {fmt(va['statistic_pooled_relative_cost_difference'])} "
                    f"(policy minus adaptive, relative), CI95 {ci(va)}, two-sided p {va['p_two_sided']:.4f}, Holm p "
                    f"{holm['holm_adjusted_p']['H2']:.4f}; P(policy cheaper) {va['P_policy_cheaper']:.4f}"),
        "Cal.": col(bool(sa) and all(x != "inf" and x < 0 for x in sa.values()),
                    f"calibrated stress test, seeds 0-4, cells where both qualify ({len(sa)}): {srt(sa)}. "
                    f"(Default-channel 24-cell H2, h2_grid24.json: {fmt(ga['statistic_pooled_relative_cost_difference'])}, "
                    f"CI95 {ci(ga)})"),
        "SUMO": col(False, f"undefined: the policy qualifies in no SUMO headline cell {sumo_reach}"),
        "Veins": col(None, "n/a (policy not ported to Veins)"),
        "Sens.": col(not not gm and len(gm) == len(sens),
                     f"settings with a negative margin in every cell where it is defined: {len(gm)} of {len(sens)}"),
        "Seeds": col(None, f"{len(h2['training_seeds'])} training seeds"),
        "Resampling/held-out": col(None, f"hierarchical bootstrap B={va['B']} over training x EVAL seeds; "
                                         f"{va['bootstrap_undefined_replicates_nan']} undefined replicates excluded"),
    }
    p2 = {
        "Set.": col(bool(vr["inferiority"]),
                    f"pooled relative cost difference {fmt(vr['statistic_pooled_relative_cost_difference'])}, "
                    f"CI95 {ci(vr)}, two-sided p {vr['p_two_sided']:.2g} (exploratory, not in the Holm family)"),
        "Cal.": col(bool(sr) and all(x == "inf" or x > 0 for x in sr.values()),
                    f"calibrated stress test, seeds 0-4, cells where both qualify ({len(sr)}): {srt(sr)}. "
                    f"(Default-channel 24-cell, h2_grid24.json: {fmt(gr['statistic_pooled_relative_cost_difference'])}, "
                    f"CI95 {ci(gr)})"),
        "SUMO": col(False, f"undefined: the policy qualifies in no SUMO headline cell {sumo_reach}"),
        "Veins": col(None, "n/a (policy not ported to Veins)"),
        "Sens.": col(None, "not tested (the sensitivity grid compares against adaptive classical only)"),
        "Seeds": col(None, f"{len(h2['training_seeds'])} training seeds"),
        "Resampling/held-out": col(None, f"hierarchical bootstrap B={vr['B']}; "
                                         f"{vr['bootstrap_undefined_replicates_nan']} undefined replicates excluded"),
    }
    set_reach = {c: h2["vs_adaptive_classical"]["point_detail"][c]["policy"]["qualifies"] for c in POLICY_CELLS}
    p3 = {
        "Set.": col(set_reach == POLICY_CELLS, f"cell-aggregate qualification (10 seeds): {set_reach}; "
                                               f"seed x cell qualification {va['qualification']['policy'][0]}/"
                                               f"{va['qualification']['policy'][1]} (Wilson95 "
                                               f"{[round(x, 3) for x in va['qualification']['policy'][2]]})"),
        "Cal.": col(stress_reach == POLICY_CELLS, f"calibrated channel, seeds 0-4: {stress_reach}; "
                                                  f"{sec['stress']['cells_reached']['policy']} of 24 cells reached"),
        "SUMO": col(sumo_reach == POLICY_CELLS, f"SUMO mobility, seeds 0-9: {sumo_reach}"),
        "Veins": col(None, "n/a (policy not ported to Veins)"),
        "Sens.": col(len(gq) == len(sens), f"settings in which the cells with >= 5 of 10 qualifying seeds are exactly "
                                           f"rural d=20 and urban d=20: {len(gq)} of {len(sens)}"),
        "Seeds": col(None, f"{len(h2['training_seeds'])} training seeds"),
        "Resampling/held-out": col(None, "Wilson interval over (seed, cell) picks; selection on SEL, scoring on EVAL"),
    }
    n_seeds = len(h2["training_seeds"])
    groups = {
        "classical_schemes": [
            {"claim": "H1: no single classical family is the cheapest qualifying family in all four headline cells",
             "columns": k1, "verdict": verdict(k1)},
            {"claim": "Per-cell H1 winners: slotted_1p (rural d=2), greedy_farthest (rural d=20), weighted_p (rural "
                      "d=80), counter_based (urban d=20)", "columns": k2_cols, "verdict": verdict(k2_cols)},
        ],
        "tested_policy_at_training_budget": [
            {"claim": "H2: the policy is cheaper than the SEL-selected adaptive classical scheme", "columns": p1,
             "verdict": verdict(p1, n_seeds)},
            {"claim": "The policy is costlier than the SEL-selected best rule baseline", "columns": p2,
             "verdict": verdict(p2, n_seeds)},
            {"claim": "The policy qualifies in rural d=20 and urban d=20 and not in rural d=2 or rural d=80", "columns": p3,
             "verdict": verdict(p3, n_seeds)},
        ],
    }
    payload = {
        "rules": __doc__.split("\n\n", 2)[2].strip(),
        "D6_consequence": "H3 fails in both timing modes; the classical audit computed in Veins is primary and "
                          "custom-simulator claims are restricted to relative comparisons (DEVIATIONS D6).",
        "groups": groups,
    }
    print(write("ledger.json", payload, {"sources": ["h1.json", "h2.json", "h2_grid24.json", "h3.json",
                                                     "h3_subepoch.json", "holm.json", "secondary.json"]}))
    for g, cl in groups.items():
        for c in cl:
            print(f"[{g}] {c['verdict']:14s} {c['claim']}")
            for k, v in c["columns"].items():
                print(f"      {k:20s} {str(v['holds']):5s} {v['evidence'][:150]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
