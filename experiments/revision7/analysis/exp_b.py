"""Revision 7, Experiment B: greedy_farthest and DV-CAST with the D9 fix (results/revision7/analysis/exp_b.json).

    python -m experiments.revision7.analysis.exp_b [--B 10000]

Descriptive; the registered analysis stays as it was (results/revision7/ADDENDUM.md). The fixed rows replace those two families'
rows in rural d=2, rural d=20 and urban NLOS d=20; rural d=80 keeps its revision-5 rows, so it stays unfixed. Thresholds are
recomputed and V1-V3 use the veins_methods statistics and bootstrap (RNG 20261102, Holm).
"""

from __future__ import annotations

import argparse
import json
import math
import sys

import numpy as np

from common.config import PROJECT_ROOT
from experiments.revision1.provenance import stamp, write_json
from experiments.revision5.analysis.common import holm
from experiments.revision5.analysis.h3 import FAMILIES, FAMILY_OF_VEINS, veins_rows
from experiments.revision5.sweep import HEADLINE
from experiments.revision6.analysis.veins_methods import CK, RNG_SEED, SEEDS, build, cell_eval, stats, veins6_rows
from experiments.revision7.veins.queue7 import CELLS_B, FAMILIES_B

R7 = PROJECT_ROOT / "results" / "revision7"
OUT = R7 / "analysis"
RUNS = [f"ref_s{k}" for k in range(5)]


def fin(x):
    return x if isinstance(x, float) and math.isfinite(x) else None


def fixed_rows() -> list[dict]:
    out = []
    for line in (R7 / "veins" / "d9_rows.jsonl").open(encoding="utf-8"):
        r = json.loads(line)["row"]
        out.append({"scenario": r["scenario"], "density": r["density"], "seed": r["seed"], "policy": FAMILY_OF_VEINS[r["policy"]],
                    "value": float(r["knobs"]), "rwcr": r["rwcr"], "tx_per_at_risk_informed": r["tx_per_at_risk_informed"],
                    "actionable_deadline_miss_rate": r["actionable_deadline_miss_rate"], "tir_p95_s": r["tir_p95_s"]})
    return out


def settings(t, fam) -> dict:
    return {f"{k[1]:g}": {"rwcr": float(t.mean("rwcr")[i]), "cost": float(t.mean("cost")[i]), "miss": float(t.mean("miss")[i])}
            for i, k in enumerate(t.settings) if k[0] == fam}


def summary(e) -> dict:
    return {"q_star": e["q_star"], "m_star": e["m_star"], "best_classical": [e["best_classical"][0], fin(e["best_classical"][1])],
            "best_rule": [e["best_rule"][0], fin(e["best_rule"][1])], "policy_n_pick": e["policy_n_pick"], "policy_aggregate": fin(e["policy_aggregate"]),
            "classical": {f: fin(v) for f, v in e["classical"].items()}, "rules": {f: fin(v) for f, v in e["rules"].items()}}


def main(argv=None) -> int:
    import numpy
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--B", type=int, default=10000)
    a = ap.parse_args(argv)
    print(f"[env] {sys.executable} numpy {numpy.__version__}", flush=True)
    fams = {FAMILY_OF_VEINS[p] for p in FAMILIES_B}
    vr, _ = veins_rows()
    rule, pol = veins6_rows(RUNS)
    old = build(vr, rule, pol, RUNS)
    kept = [r for r in vr if not (r["policy"] in fams and (r["scenario"], float(r["density"])) in CELLS_B)]
    new = build(kept + fixed_rows(), rule, pol, RUNS)
    p_old = {c: cell_eval(old[c], RUNS) for c in HEADLINE}
    p_new = {c: cell_eval(new[c], RUNS) for c in HEADLINE}
    st_old, st_new = stats(p_old), stats(p_new)

    rng = np.random.default_rng(RNG_SEED)
    boot = {"V1": [], "V2": [], "V3": []}
    und = {"V2": 0, "V3": 0}
    n = len(SEEDS)
    for _ in range(a.B):
        idx = rng.integers(0, n, n)
        rsel = [RUNS[i] for i in rng.integers(0, len(RUNS), len(RUNS))]
        rk = [f"{r}#{j}" for j, r in enumerate(rsel)]
        ev = {c: cell_eval({**new[c], "policy": {f"{r}#{j}": new[c]["policy"][r] for j, r in enumerate(rsel)}}, rk, idx) for c in HEADLINE}
        s = stats(ev)
        boot["V1"].append(s["V1"])
        for k in ("V2", "V3"):
            if s[k] is None or not math.isfinite(s[k]):
                und[k] += 1
            else:
                boot[k].append(s[k])

    def ci(v):
        return [float(np.quantile(v, q, method="inverted_cdf")) for q in (0.025, 0.975)] if v else None

    def two(v):
        return float(min(1.0, 2 * min(np.mean(np.asarray(v) <= 0), np.mean(np.asarray(v) >= 0)))) if v else None

    p = {"V1": float(np.mean(np.asarray(boot["V1"]) < 2)), "V2": two(boot["V2"]), "V3": two(boot["V3"])}
    adj = holm({k: v for k, v in p.items() if v is not None})
    reg = json.loads((PROJECT_ROOT / "results" / "revision6" / "analysis" / "veins_methods.json").read_text(encoding="utf-8"))
    payload = {
        "label": "Descriptive; the registered veins_methods analysis is unchanged. Rural d=80 keeps its revision-5 rows for both families (unfixed).",
        "cells_fixed": [CK(c) for c in CELLS_B], "seeds": SEEDS, "B": a.B, "rng_seed": RNG_SEED,
        "cells": {CK(c): {"with_fix": summary(p_new[c]), "registered": summary(p_old[c]),
                          "best_classical_changes": p_new[c]["best_classical"][0] != p_old[c]["best_classical"][0],
                          "family_settings": {f: {"with_fix": settings(new[c]["classical"], f), "registered": settings(old[c]["classical"], f)}
                                              for f in sorted(fams)}} for c in HEADLINE},
        "tests": {
            "V1": {"registered": reg["V1"]["statistic_cells_rule_cheaper"], "with_fix": st_new["V1"], "recheck_registered_point": st_old["V1"],
                   "p_fewer_than_2": p["V1"], "holm_p": adj.get("V1"),
                   "bootstrap_distribution": {str(k): int(np.sum(np.asarray(boot["V1"]) == k)) for k in range(5)}},
            **{k: {"registered": reg[k]["statistic_pooled_relative_difference"], "with_fix": st_new[k], "recheck_registered_point": st_old[k],
                   "ci95": ci(boot[k]), "p_two_sided": p[k], "holm_p": adj.get(k), "undefined_replicates": und[k]} for k in ("V2", "V3")}},
        "families": list(FAMILIES)}
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "exp_b.json", {"provenance": stamp("revision7/analysis/exp_b.json", {"B": a.B}, seeds=SEEDS), **payload})
    print(OUT / "exp_b.json")
    print(json.dumps(payload["tests"], indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
