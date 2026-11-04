"""Revision 6 Step 6: classical, rules and the learned policy in Veins -- V1-V4 (results/revision6/analysis/veins_methods.json).

    python -m experiments.revision6.analysis.veins_methods [--B 10000] [--runs ref_s0 ... ref_s4]

Definitions: results/revision6/PREREGISTRATION_ADDENDUM.md section 4 (Algorithm 1 inside Veins as D3: per cell q* = 0.95 x
max mean RWCR over the 55 classical settings, m* = min mean actionable miss over them + 0.05; matched cost = the
interpolated crossing at q* over a family's settings with mean miss <= m*, PolicyCurve.overhead_at). Families: the 7
classical families, RAR (27 settings), SCF-slotted (12), each policy training run (6 biases). Tier A, seeds 200-209:
an EVAL-HINDSIGHT comparison (thresholds, crossings and comparison on the same seeds).
Bootstrap: B replicates, RNG 20261102, replay seeds (and training runs) resampled with replacement; q*, m*, every
crossing and every "best" recomputed; undefined replicates counted and excluded. Holm over V1-V3.
Custom side of V4: the committed revision-5 custom-on-SUMO rows (epoch mode), same seeds, same definitions.
Scope (addendum D10): the policy has training seeds 0-4 and no rural d=80, so V2, V3 and the policy's V4 pairs and
qualification cover three cells; classical and rules keep all four, including V1.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

from analysis.pareto import OperatingPoint, PolicyCurve
from common.config import PROJECT_ROOT
from experiments.revision5.analysis.common import holm, rows, tables, wilson, write as write5
from experiments.revision5.analysis.h2 import run_sha
from experiments.revision5.analysis.h3 import FAMILIES, key, veins_rows
from experiments.revision5.sweep import HEADLINE
from experiments.revision6.veins.queue6 import POLICY_CELLS

OUT =PROJECT_ROOT / "results" / "revision6" / "analysis"
V6 = PROJECT_ROOT / "results" / "revision6" / "veins" / "veins6_rows.jsonl"
SEEDS = list(range(200, 210))
RNG_SEED = 20261102
CK = lambda c: f"{c[0]} d={c[1]:g}"  # noqa: E731
RULES = ("rar", "scf_slotted")


def write(name: str, payload: dict, params: dict, seeds=None) -> Path:
    from experiments.revision1.provenance import stamp, write_json
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / name
    write_json(p, {"provenance": stamp(f"revision6/analysis/{name}", params, seeds=seeds), **payload})
    return p


# ---- rows -----------------------------------------------------------------------------------------------------
def veins6_rows(runs) -> tuple[list[dict], list[dict]]:
    """(rule rows, policy rows) from the scored revision-6 replays, in the revision-5 row format."""
    rule, pol = [], []
    for line in V6.open(encoding="utf-8"):
        r = json.loads(line)["row"]
        base = {"scenario": r["scenario"], "density": r["density"], "seed": r["seed"], "rwcr": r["rwcr"],
                "tx_per_at_risk_informed": r["tx_per_at_risk_informed"],
                "actionable_deadline_miss_rate": r["actionable_deadline_miss_rate"], "tir_p95_s": r["tir_p95_s"]}
        if r["policy"] in RULES:
            rule.append({**base, "policy": r["policy"], "value": r["knobs"]})
        elif r["policy"] == "ai_harp":
            run, b = r["knobs"].rsplit("_b", 1)
            if run in runs:
                pol.append({**base, "policy": run, "value": float(b)})
    return rule, pol


def custom_rows(runs) -> tuple[list[dict], list[dict], list[dict]]:
    """Custom-on-SUMO (revision 5, epoch mode): classical, rule and policy rows of the headline cells."""
    cl = [{**r, "value": 0.0 if r["param"] == "" else float(r["value"])}
          for r in rows("classical__EVAL__headline__default__sumo.csv")]
    ru = []
    for r in rows("rule__EVAL__headline__default__sumo.csv"):
        v = json.loads(r["value"])
        kn = ",".join(repr(float(x)) for x in v) if r["policy"] == "rar" else ",".join(str(x) for x in v)
        ru.append({**r, "value": kn})
    sha = {run_sha(r): r for r in runs}
    po = []
    for r in rows("agent__EVAL__headline__default__sumo.csv"):
        q = json.loads(r["params_json"])
        if q.get("checkpoint_sha") in sha and float(q.get("tau", -1)) == 0.5 and "random_gate_rate" not in q:
            po.append({**r, "policy": sha[q["checkpoint_sha"]], "value": float(r["value"])})
    return cl, ru, po


def build(cl, ru, po, runs, cells=HEADLINE, seeds=SEEDS) -> dict:
    """Per cell: classical table, rule table, {run: policy table}."""
    k = lambda r: key(r["policy"], r["value"]) if r["policy"] in FAMILIES else (r["policy"], r["value"])  # noqa: E731
    tc = tables(cl, cells, seeds, key=k)
    tr = tables(ru, cells, seeds, key=lambda r: (r["policy"], r["value"]))
    tp = {run: tables([r for r in po if r["policy"] == run], cells, seeds, key=lambda r: (r["policy"], r["value"]))
          for run in runs}
    return {c: {"classical": tc[c], "rules": tr[c], "policy": {run: tp[run][c] for run in runs}} for c in cells}


# ---- Algorithm 1 ---------------------------------------------------------------------------------------------
def thresholds(t, idx=None) -> tuple[float, float]:
    return 0.95 * float(np.nanmax(t.mean("rwcr", idx))), float(np.nanmin(t.mean("miss", idx))) + 0.05


def matched(t, fam, q, guard, idx=None) -> float:
    rw, co, mi = t.mean("rwcr", idx), t.mean("cost", idx), t.mean("miss", idx)
    pts = [OperatingPoint(fam, "", str(k[1]), float(rw[i]), 0.0, float(co[i]), 0.0, 0, {"miss": float(mi[i])})
           for i, k in enumerate(t.settings) if k[0] == fam]
    return PolicyCurve(fam, pts).overhead_at(q, eligible=lambda p: p.extras["miss"] <= guard) if pts else math.inf


def cell_eval(cell: dict, runs, idx=None) -> dict:
    q, m = thresholds(cell["classical"], idx)
    cls = {f: matched(cell["classical"], f, q, m, idx) for f in FAMILIES}
    rul = {f: matched(cell["rules"], f, q, m, idx) for f in RULES}
    pol = {r: matched(cell["policy"][r], r.split("#")[0], q, m, idx) for r in runs}   # "#j": a resampled run's label
    fin = lambda d: {k: v for k, v in d.items() if math.isfinite(v)}  # noqa: E731
    bc = min(fin(cls).items(), key=lambda kv: kv[1], default=(None, math.inf))
    br = min(fin(rul).items(), key=lambda kv: kv[1], default=(None, math.inf))
    picks = list(fin(pol).values())
    agg = float(np.mean(picks)) if picks and 2 * len(picks) >= len(runs) else math.inf
    return {"q_star": q, "m_star": m, "classical": cls, "rules": rul, "policy": pol, "best_classical": bc,
            "best_rule": br, "policy_aggregate": agg, "policy_n_pick": len(picks)}


def stats(ev: dict) -> dict:
    """V1 count; V2 / V3 pooled relative differences (None when no cell has both)."""
    v1 = sum(1 for e in ev.values() if math.isfinite(e["best_rule"][1]) and math.isfinite(e["best_classical"][1])
             and e["best_rule"][1] < e["best_classical"][1])

    def pooled(ref):
        d = [e["policy_aggregate"] / e[ref][1] - 1 for e in ev.values()
             if math.isfinite(e["policy_aggregate"]) and math.isfinite(e[ref][1]) and e[ref][1] > 0]
        return float(np.mean(d)) if d else None
    return {"V1": v1, "V2": pooled("best_classical"), "V3": pooled("best_rule")}


def custom_matched(cell: dict, runs, idx=None) -> dict:
    return cell_eval(cell, runs, idx)


def main(argv=None) -> int:
    import numpy
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--B", type=int, default=10000)
    ap.add_argument("--runs", nargs="*", default=[f"ref_s{k}" for k in range(5)])
    a = ap.parse_args(argv)
    print(f"[env] {sys.executable} numpy {numpy.__version__}", flush=True)
    runs = a.runs
    vr, _ = veins_rows()
    rr, pr = veins6_rows(runs)
    if any((r["scenario"], float(r["density"])) not in POLICY_CELLS for r in pr):
        raise SystemExit("policy rows outside the D10 cells")
    V = build(vr, rr, pr, runs)
    cc, cr, cp = custom_rows(runs)
    C = build(cc, cr, cp, runs)

    point = {c: cell_eval(V[c], runs) for c in HEADLINE}
    cpoint = {c: cell_eval(C[c], runs) for c in HEADLINE}
    st = stats(point)

    rng = np.random.default_rng(RNG_SEED)
    boot = {"V1": [], "V2": [], "V3": [], "rho": []}
    und = {"V2": 0, "V3": 0, "rho": 0}
    n = len(SEEDS)
    for _ in range(a.B):
        idx = rng.integers(0, n, n)
        rsel = [runs[i] for i in rng.integers(0, len(runs), len(runs))]
        sub = {c: {"classical": V[c]["classical"], "rules": V[c]["rules"],
                   "policy": {f"{r}#{j}": V[c]["policy"][r] for j, r in enumerate(rsel)}} for c in HEADLINE}
        rk = [f"{r}#{j}" for j, r in enumerate(rsel)]
        ev = {}
        for c in HEADLINE:
            e = cell_eval({**sub[c], "policy": {k: sub[c]["policy"][k] for k in rk}}, rk, idx)
            ev[c] = e
        s = stats(ev)
        boot["V1"].append(s["V1"])
        for k in ("V2", "V3"):
            if s[k] is None or not math.isfinite(s[k]):
                und[k] += 1
            else:
                boot[k].append(s[k])
        csub = {c: {"classical": C[c]["classical"], "rules": C[c]["rules"],
                    "policy": {k: C[c]["policy"][k.split("#")[0]] for k in rk}} for c in HEADLINE}
        cev = {c: cell_eval(csub[c], rk, idx) for c in HEADLINE}
        rho = spearman_pairs(ev, cev)[0]
        if rho is None or not math.isfinite(rho):
            und["rho"] += 1
        else:
            boot["rho"].append(rho)

    def ci(v):
        return [float(np.quantile(v, 0.025, method="inverted_cdf")), float(np.quantile(v, 0.975, method="inverted_cdf"))] if v else None

    def two_sided(v):
        if not v:
            return None
        b = np.asarray(v)
        return float(min(1.0, 2 * min(np.mean(b <= 0), np.mean(b >= 0))))

    p = {"V1": float(np.mean(np.asarray(boot["V1"]) < 2)), "V2": two_sided(boot["V2"]), "V3": two_sided(boot["V3"])}
    adj = holm({k: v for k, v in p.items() if v is not None})
    rho, npairs = spearman_pairs(point, cpoint)

    def percell(ref):
        return {CK(c): (None if not (math.isfinite(e["policy_aggregate"]) and math.isfinite(e[ref][1]) and e[ref][1] > 0)
                        else e["policy_aggregate"] / e[ref][1] - 1) for c, e in point.items()}

    fin = lambda x: x if (isinstance(x, float) and math.isfinite(x)) else None  # noqa: E731
    payload = {
        "label": "EVAL-hindsight comparison on Tier A seeds 200-209 (thresholds, crossings and comparison on the same seeds)",
        "runs": runs, "seeds": SEEDS, "B": a.B, "rng_seed": RNG_SEED,
        "policy_cells": [CK(c) for c in POLICY_CELLS],
        "scope": "Addendum D10: policy training seeds 0-4, rural d=80 not run for the policy; V2, V3 and the policy's V4 "
                 "pairs cover three cells, V1 and the classical and rule methods cover four",
        "cells": {CK(c): {"q_star": e["q_star"], "m_star": e["m_star"],
                          "classical": {f: fin(v) for f, v in e["classical"].items()},
                          "best_classical": {"family": e["best_classical"][0], "cost": fin(e["best_classical"][1])},
                          "rules": {f: fin(v) for f, v in e["rules"].items()},
                          "best_rule": {"family": e["best_rule"][0], "cost": fin(e["best_rule"][1])},
                          "policy_per_run": {r: fin(v) for r, v in e["policy"].items()},
                          "policy_n_pick": e["policy_n_pick"], "policy_aggregate": fin(e["policy_aggregate"]),
                          "rule_settings": rule_settings_table(V[c]["rules"], e["q_star"], e["m_star"]),
                          "custom_on_sumo": {"q_star": cpoint[c]["q_star"], "m_star": cpoint[c]["m_star"],
                                             "best_classical": [cpoint[c]["best_classical"][0], fin(cpoint[c]["best_classical"][1])],
                                             "best_rule": [cpoint[c]["best_rule"][0], fin(cpoint[c]["best_rule"][1])],
                                             "policy_aggregate": fin(cpoint[c]["policy_aggregate"])}}
                  for c, e in point.items()},
        "V1": {"statistic_cells_rule_cheaper": st["V1"], "holds": st["V1"] >= 2, "p_fewer_than_2": p["V1"],
               "holm_p": adj.get("V1"), "bootstrap_distribution": {str(k): int(np.sum(np.asarray(boot["V1"]) == k)) for k in range(5)}},
        "V2": {"statistic_pooled_relative_difference": st["V2"], "per_cell": percell("best_classical"),
               "ci95": ci(boot["V2"]), "p_two_sided": p["V2"], "holm_p": adj.get("V2"),
               "P_policy_cheaper": float(np.mean(np.asarray(boot["V2"]) < 0)) if boot["V2"] else None,
               "superiority": bool(boot["V2"] and ci(boot["V2"])[1] < 0), "undefined_replicates": und["V2"]},
        "V3": {"statistic_pooled_relative_difference": st["V3"], "per_cell": percell("best_rule"),
               "ci95": ci(boot["V3"]), "p_two_sided": p["V3"], "holm_p": adj.get("V3"),
               "P_policy_cheaper": float(np.mean(np.asarray(boot["V3"]) < 0)) if boot["V3"] else None,
               "superiority": bool(boot["V3"] and ci(boot["V3"])[1] < 0), "undefined_replicates": und["V3"]},
        "V4": {"spearman_rho": rho, "n_pairs": npairs, "ci95": ci(boot["rho"]), "undefined_replicates": und["rho"],
               "pairs": "(method, cell) with a finite matched cost in both simulators; methods: 7 classical families, "
                        "RAR, SCF-slotted, the policy aggregate"},
        "qualification": qualification(point, runs),
        "cross_fitted_tier_B": "not run (Tier B incomplete)",
    }
    print(write("veins_methods.json", payload, {"B": a.B, "runs": runs}, seeds=SEEDS))
    print(json.dumps({k: payload[k] for k in ("V1", "V2", "V3")} | {"V4": {k: payload["V4"][k] for k in ("spearman_rho", "n_pairs", "ci95")}},
                     indent=1, default=str))
    return 0


def spearman_pairs(ev: dict, cev: dict) -> tuple[float | None, int]:
    from scipy.stats import spearmanr
    a, b = [], []
    for c in ev:
        for f in FAMILIES:
            x, y = ev[c]["classical"][f], cev[c]["classical"][f]
            if math.isfinite(x) and math.isfinite(y):
                a.append(x); b.append(y)
        for f in RULES:
            x, y = ev[c]["rules"][f], cev[c]["rules"][f]
            if math.isfinite(x) and math.isfinite(y):
                a.append(x); b.append(y)
        x, y = ev[c]["policy_aggregate"], cev[c]["policy_aggregate"]
        if math.isfinite(x) and math.isfinite(y):
            a.append(x); b.append(y)
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return None, len(a)
    return float(spearmanr(a, b).statistic), len(a)


def rule_settings_table(t, q, m) -> dict:
    rw, co, mi = t.mean("rwcr"), t.mean("cost"), t.mean("miss")
    return {f"{k[0]}:{k[1]}": {"rwcr": float(rw[i]), "cost": float(co[i]) if math.isfinite(co[i]) else None,
                               "miss": float(mi[i]), "meets_q_and_m": bool(rw[i] >= q and mi[i] <= m)}
            for i, k in enumerate(t.settings)}


def qualification(point: dict, runs) -> dict:
    out = {}
    n, np_ = len(point), len(POLICY_CELLS)
    for name, get, of in (("best_classical", lambda e: math.isfinite(e["best_classical"][1]), n),
                          ("best_rule", lambda e: math.isfinite(e["best_rule"][1]), n),
                          ("policy_aggregate", lambda e: math.isfinite(e["policy_aggregate"]), np_)):
        k = sum(1 for e in point.values() if get(e))
        out[name] = {"cells": k, "of": of, "wilson95": wilson(k, of)}
    k = sum(e["policy_n_pick"] for e in point.values())
    out["policy_run_x_cell_picks"] = {"count": k, "of": np_ * len(runs), "wilson95": wilson(k, np_ * len(runs))}
    return out


if __name__ == "__main__":
    raise SystemExit(main())
