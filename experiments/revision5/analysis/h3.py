"""H3 -- packet-level agreement between the custom simulator (on SUMO) and Veins (PREREGISTRATION; definitions in
DEVIATIONS D3 and ANALYSIS_SPEC). Writes results/revision5/analysis/h3.json.

    python -m experiments.revision5.analysis.h3 [--B 10000]
    python -m experiments.revision5.analysis.h3 --mode sub_epoch     (DEVIATIONS D6 re-run -> h3_subepoch.json)

Inputs (committed): results/revision5/veins/veins_rows.jsonl (Veins replays, scored by analysis.metrics),
results/revision5/veins/custom_h3.jsonl (custom network-layer runs), results/revision5/runs/
classical__EVAL__headline__default__sumo.csv (custom metric rows, seeds 200-229).
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np

from analysis.pareto import OperatingPoint, PolicyCurve, POLICY_SWEEPS
from common.config import PROJECT_ROOT
from experiments.revision5.analysis.common import (B, HEADLINE, SEED_RNG, cal_thresholds, clopper_pearson, finite,
                                                  rows, select, tables, write)

VEINS = PROJECT_ROOT / "results" / "revision5" / "veins"
FAMILY_OF_VEINS = {"flooding": "flooding", "p_persistence": "p_persistence_03", "slotted_1p": "slotted_1p",
                   "weighted_p": "weighted_p", "counter_based": "counter_based", "greedy_farthest": "greedy_farthest",
                   "dvcast": "dvcast"}
FAMILIES = list(POLICY_SWEEPS)
NET = list(range(200, 210))
ALL = list(range(200, 230))
FOLDS = [list(range(200, 210)), list(range(210, 220)), list(range(220, 230))]
NOMINAL = {"rural_highway": 610.4, "urban_nlos": 479.6}


def _num(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def key(fam: str, value) -> tuple:
    return (fam, 0.0 if value in ("", None) else float(value))


def veins_rows() -> tuple[list[dict], dict]:
    out, raw = [], {}
    for line in (VEINS / "veins_rows.jsonl").open(encoding="utf-8"):
        rec = json.loads(line)
        r = rec["row"]
        fam = FAMILY_OF_VEINS[r["policy"]]
        row = {"scenario": r["scenario"], "density": r["density"], "seed": r["seed"], "policy": fam,
               "value": float(r["knob"]), "rwcr": r.get("rwcr"), "tx_per_at_risk_informed": r.get("tx_per_at_risk_informed"),
               "actionable_deadline_miss_rate": r.get("actionable_deadline_miss_rate"), "tir_p95_s": r.get("tir_p95_s"),
               "channel_busy_ratio": r.get("channel_busy_ratio")}
        out.append(row)
        raw[(r["scenario"], float(r["density"]), int(r["seed"]), key(fam, r["knob"]))] = rec["raw"]
    return out, raw


def custom_rows() -> list[dict]:
    return [{**r, "value": 0.0 if r["param"] == "" else float(r["value"])}
            for r in rows("classical__EVAL__headline__default__sumo.csv")]


def tab(rs, seeds):
    return tables(rs, HEADLINE, seeds, key=lambda r: key(r["policy"], r["value"]))


# ---- matched cost (Algorithm 1 within a simulator) -------------------------------------------------------------
def matched_costs(t, idx=None) -> dict[str, float]:
    rw, co, mi = t.mean("rwcr", idx), t.mean("cost", idx), t.mean("miss", idx)
    q = 0.95 * float(np.nanmax(rw))
    guard = float(np.nanmin(mi)) + 0.05
    out = {}
    for f in FAMILIES:
        pts = [OperatingPoint(f, "", k[1], float(rw[i]), 0.0, float(co[i]), 0.0, 0, {"miss": float(mi[i])})
               for i, k in enumerate(t.settings) if k[0] == f]
        out[f] = PolicyCurve(f, pts).overhead_at(q, eligible=lambda p: p.extras["miss"] <= guard)
    return out


def spearman(tv, tc, idx=None) -> tuple[float, int]:
    from scipy.stats import spearmanr
    a, b = [], []
    for c in HEADLINE:
        mv, mc = matched_costs(tv[c], idx), matched_costs(tc[c], idx)
        for f in FAMILIES:
            if math.isfinite(mv[f]) and math.isfinite(mc[f]):
                a.append(mv[f]); b.append(mc[f])
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan"), len(a)
    return float(spearmanr(a, b).statistic), len(a)


# ---- cross-fitted H1 within a simulator -------------------------------------------------------------------------
def crossfit(t30, mask_of, idx_folds) -> dict:
    """idx_folds: per fold, the column indices (into the 30-seed table) of that fold's (resampled) seeds."""
    res = {}
    for c in HEADLINE:
        t = t30[c]
        fam = {}
        for f in FAMILIES:
            ok_all, costs = True, []
            for fi, held in enumerate(idx_folds):
                train = np.concatenate([idx_folds[j] for j in range(3) if j != fi])
                q, m = cal_thresholds(t, train)
                k = select(t, q, m, math.inf, train, mask_of[c][f])
                if k < 0:
                    ok_all = False
                    break
                rw, mi = float(t.mean("rwcr", held)[k]), float(t.mean("miss", held)[k])
                ok_all &= bool(rw >= q and mi <= m)
                costs.append(float(t.mean("cost", held)[k]))
            fam[f] = {"qualifies": ok_all, "cost": float(np.mean(costs)) if ok_all else math.inf}
        qual = {f: v["cost"] for f, v in fam.items() if v["qualifies"]}
        res[c] = {"winner": min(qual, key=lambda f: (qual[f], FAMILIES.index(f))) if qual else None, "families": fam}
    return res


def crossfit_verdict(t30, nB, rng) -> dict:
    mask_of = {c: {f: np.array([k[0] == f for k in t30[c].settings]) for f in FAMILIES} for c in HEADLINE}
    base = [np.arange(10 * i, 10 * i + 10) for i in range(3)]
    point = crossfit(t30, mask_of, base)
    one = 0
    for _ in range(nB):
        folds = [b[rng.integers(0, 10, 10)] for b in base]
        w = [crossfit(t30, mask_of, folds)[c]["winner"] for c in HEADLINE]
        one += int(w[0] is not None and all(x == w[0] for x in w))
    lo, hi = clopper_pearson(one, nB)
    pw = [point[c]["winner"] for c in HEADLINE]
    return {"share_one_family_wins_all_four": one / nB, "clopper_pearson_95": [lo, hi], "supported": bool(hi < 0.05),
            "point_winners": {f"{c[0]} d={c[1]:g}": point[c]["winner"] for c in HEADLINE},
            "point_one_family_wins_all_four": bool(pw[0] is not None and all(x == pw[0] for x in pw)),
            "point_families": {f"{c[0]} d={c[1]:g}": {f: {**v, "cost": finite(v["cost"])} for f, v in point[c]["families"].items()}
                               for c in HEADLINE}}


# ---- network layer -----------------------------------------------------------------------------------------------
def subepoch_records() -> list[dict]:
    return [json.loads(l) for l in (VEINS / "custom_h3_subepoch.jsonl").open(encoding="utf-8")]


def subepoch_rows(recs) -> list[dict]:
    """Custom metric rows of the frozen sub-epoch re-run (D6), in the shape of custom_rows()."""
    return [{**r["metrics"], "scenario": r["scenario"], "density": r["density"], "seed": r["seed"],
             "policy": r["policy"], "param": r["metrics"].get("param", ""),
             "value": 0.0 if r["value"] in ("", None) else float(r["value"])} for r in recs]


def network(vraw, vrows, cust=None) -> dict:
    if cust is None:
        cust = [json.loads(l) for l in (VEINS / "custom_h3.jsonl").open(encoding="utf-8")]
    out = {}
    for sc, d in HEADLINE:
        R = NOMINAL[sc]
        cv = [r for r in cust if r["scenario"] == sc and float(r["density"]) == d and int(r["seed"]) in NET]
        vv = [v for (s2, d2, sd, _k), v in vraw.items() if s2 == sc and d2 == d and sd in NET and v is not None]

        def pdr(recs):
            a, s = {}, {}
            for r in recs:
                for b, n in r["bins_attempts"].items():
                    a[int(b)] = a.get(int(b), 0) + n
                    s[int(b)] = s.get(int(b), 0) + r["bins_success"].get(b, r["bins_success"].get(str(b), 0))
            return {b: (s[b] / a[b], a[b]) for b in sorted(a)}
        pv, pc = pdr(vv), pdr(cv)
        bins = sorted(set(pv) | set(pc))
        rows_b = []
        for b in bins:
            if b * 50.0 >= R:
                continue
            x, y = pv.get(b, (float("nan"), 0)), pc.get(b, (float("nan"), 0))
            diff = abs(x[0] - y[0]) if (x[1] and y[1]) else float("nan")
            rows_b.append({"bin_m": [b * 50, b * 50 + 50], "pdr_veins": x[0], "n_veins": x[1], "pdr_custom": y[0],
                           "n_custom": y[1], "abs_diff": diff})
        diffs = [r["abs_diff"] for r in rows_b]
        pdr_pass = bool(diffs) and all(math.isfinite(x) and x <= 0.05 for x in diffs)
        dv = np.array([x for v in vv for x in v["single_hop_delays_s"]])
        dc = np.array([x for r in cv for x in r["single_hop_delays_s"]])
        med_v, med_c = (float(np.median(dv)) if dv.size else float("nan")), (float(np.median(dc)) if dc.size else float("nan"))
        cbr_all = [_num(r["channel_busy_ratio"]) for r in vrows
                   if r["scenario"] == sc and float(r["density"]) == d and int(r["seed"]) in NET]
        cbr_v = np.array([x for x in cbr_all if math.isfinite(x)])
        cbr_c = np.array([r["beacon_cbr"] + r["dissemination_cbr"] for r in cv])
        mv, mc = (float(np.median(cbr_v)) if cbr_v.size else float("nan")), float(np.median(cbr_c))
        rel = abs(mv - mc) / mc if mc else float("nan")
        att_c = sum(r["n_rx_attempts"] for r in cv)
        out[f"{sc} d={d:g}"] = {
            "nominal_range_m": R,
            "pdr_vs_distance": rows_b, "pdr_max_abs_diff": max([x for x in diffs if math.isfinite(x)], default=None),
            "pdr_criterion_pass": pdr_pass,
            "single_hop_delay_median_s": {"veins": med_v, "custom": med_c, "abs_diff": abs(med_v - med_c),
                                          "n": [int(dv.size), int(dc.size)],
                                          "criterion_pass": bool(abs(med_v - med_c) <= 0.1)},
            "channel_busy_ratio": {"veins_median": mv, "custom_median": mc, "relative_diff": rel,
                                   "n": [int(cbr_v.size), int(cbr_c.size)], "veins_missing": len(cbr_all) - int(cbr_v.size),
                                   "criterion_pass": bool(rel <= 0.2)},
            "in_range_loss": {"veins": 1 - sum(v * n for v, n in pv.values()) / max(sum(n for _, n in pv.values()), 1),
                              "custom": 1 - sum(r["n_rx_success"] for r in cv) / max(att_c, 1),
                              "custom_split": {k: sum(r[k] for r in cv) / max(att_c, 1)
                                               for k in ("n_fail_sensitivity", "n_fail_sinr", "n_fail_beacon")}},
            "n_runs": {"veins": len(vv), "custom": len(cv)},
            "delay_cdf_quantiles": {"q": [0.05, 0.25, 0.5, 0.75, 0.95],
                                    "veins": [float(np.quantile(dv, x)) for x in (0.05, 0.25, 0.5, 0.75, 0.95)] if dv.size else None,
                                    "custom": [float(np.quantile(dc, x)) for x in (0.05, 0.25, 0.5, 0.75, 0.95)] if dc.size else None},
        }
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--B", type=int, default=B)
    ap.add_argument("--mode", choices=("epoch", "sub_epoch"), default="epoch")
    a = ap.parse_args(argv)
    rng = np.random.default_rng(SEED_RNG)
    vr, vraw = veins_rows()
    sub = a.mode == "sub_epoch"
    recs = subepoch_records() if sub else None
    if sub and len(recs) != 6600:
        raise SystemExit(f"sub-epoch re-run incomplete: {len(recs)}/6600 records")
    cr = subepoch_rows(recs) if sub else custom_rows()
    tv10, tc10 = tab(vr, NET), tab(cr, NET)
    rho, npairs = spearman(tv10, tc10)
    boots = []
    for _ in range(a.B):
        idx = rng.integers(0, 10, 10)
        boots.append(spearman(tv10, tc10, idx)[0])
    bs = np.array(boots)
    ok = bs[~np.isnan(bs)]
    lo, hi = (float(np.quantile(ok, 0.025, method="inverted_cdf")), float(np.quantile(ok, 0.975, method="inverted_cdf"))) if ok.size else (float("nan"),) * 2
    p_rho = float((ok <= 0.6).mean()) if ok.size else float("nan")
    matched = {f"{c[0]} d={c[1]:g}": {"veins": {f: finite(v) for f, v in matched_costs(tv10[c]).items()},
                                      "custom": {f: finite(v) for f, v in matched_costs(tc10[c]).items()}} for c in HEADLINE}
    net = network(vraw, vr, [r for r in recs if int(r["seed"]) in NET] if sub else None)
    tv30, tc30 = tab(vr, ALL), tab(cr, ALL)
    h1_veins = crossfit_verdict(tv30, a.B, np.random.default_rng(SEED_RNG + 1))
    h1_custom = crossfit_verdict(tc30, a.B, np.random.default_rng(SEED_RNG + 1))
    crit = {
        "spearman_rho_ge_0.8_and_lower_ge_0.6": bool(math.isfinite(rho) and rho >= 0.8 and lo >= 0.6),
        "pdr_vs_distance_within_0.05_every_bin": all(v["pdr_criterion_pass"] for v in net.values()),
        "channel_busy_ratio_within_20pct": all(v["channel_busy_ratio"]["criterion_pass"] for v in net.values()),
        "median_single_hop_delay_within_100ms": all(v["single_hop_delay_median_s"]["criterion_pass"] for v in net.values()),
        "same_H1_verdict_both_simulators": h1_veins["supported"] == h1_custom["supported"],
    }
    payload = {
        "hypothesis": "H3: packet-level agreement between the custom simulator (SUMO mobility) and Veins",
        "spearman": {"rho": finite(rho), "n_pairs": npairs, "ci95": [lo, hi], "p_rho_le_0.6": p_rho, "B": a.B,
                     "bootstrap_undefined_replicates": int(np.isnan(bs).sum())},
        "matched_costs": matched, "network_layer": net,
        "crossfit_H1": {"veins": h1_veins, "custom_on_sumo": h1_custom},
        "criteria": crit, "all_criteria_pass": all(crit.values()),
        "decision": ("pass: custom simulator validated for these metrics in these cells" if all(crit.values()) else
                     ("fail after the D6 re-run: per the pre-registered consequence the classical audit computed in "
                      "Veins is primary and custom-simulator claims are restricted" if sub else
                      "fail: per the pre-registered rule, add the sub-epoch timing mode, calibrate on the validation "
                      "metrics only, freeze, re-run")),
    }
    if sub:
        payload["mode"] = {"mac_timing": "sub_epoch", "phy_patch": recs[0]["phy_patch"],
                           "source": "DEVIATIONS D6; configs/phy_patch_d6_frozen.json"}
    print(write("h3_subepoch.json" if sub else "h3.json", payload, {"B": a.B, "mode": a.mode},
                seeds={"network": NET, "audit": ALL}))
    print({"rho": rho, "ci95": [lo, hi], "p_rho_le_0.6": p_rho, **crit})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
