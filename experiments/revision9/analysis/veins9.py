"""Revision 9 Veins analyses (results/revision9/ADDENDUM.md): assembly of runner shards, H6, C1 and S1.
Written before any revision-9 Veins replay existed.

    python -m experiments.revision9.analysis.veins9 assemble h6|c1   shards -> results/revision9/veins/<mode>_rows.jsonl
    python -m experiments.revision9.analysis.veins9 h6               -> results/revision9/analysis/h6.json
    python -m experiments.revision9.analysis.veins9 c1               -> results/revision9/analysis/c1.json
    python -m experiments.revision9.analysis.veins9 s1               -> results/revision9/analysis/s1.json

Thresholds: results/revision8/veins_thresholds.json (frozen, unchanged). A training seed qualifies in a cell if its
Algorithm-1 matched cost over its six deltas (seed means; guard mean actionable miss <= m*) is finite (revision-6
veins_methods.matched, unchanged). H6 per-cell rule: STRUCT9-rand >= 3/5 and STRUCT9-plain < 3/5; one-sided Fisher exact
per cell; p_H6 = min(1, 2 min(p_rural, p_urban)); supported iff the rule is met in >= 1 cell and p_H6 < 0.05 (the only
test in its family). C1 and S1 are descriptive.
"""

from __future__ import annotations

import json
import math
import sys

import numpy as np

from common.config import PROJECT_ROOT
from experiments.revision1.provenance import stamp, write_json
from experiments.revision5.analysis.common import tables
from experiments.revision6.analysis.veins_methods import matched

R8, R9 = PROJECT_ROOT / "results" / "revision8", PROJECT_ROOT / "results" / "revision9"
CELLS = (("rural_highway", 20.0), ("urban_nlos", 20.0))
SEEDS_B2 = list(range(220, 230))
ARMS = {"plain": [f"plain_s{k}" for k in range(5)], "rand": [f"rand_s{k}" for k in range(5)]}
EXPECTED = 60


def frozen() -> dict:
    c = json.loads((R8 / "veins_thresholds.json").read_text())["cells"]
    return {(sc, d): (c[f"{sc}|{d:g}"]["q_star"], c[f"{sc}|{d:g}"]["m_star"]) for sc, d in CELLS}


def assemble(mode: str) -> int:
    acc, rej, n = [], {}, 0
    with open(R9 / "veins" / f"{mode}_rows.jsonl", "w", encoding="utf-8") as fo:
        for sc, d in CELLS:
            for s in SEEDS_B2:
                dd = R9 / "veins" / mode / f"{sc}_d{d:g}_s{s}"
                ld = lambda f: json.loads((dd / f).read_text()) if (dd / f).exists() else {}  # noqa: E731
                g, t, m = ld("gate.json"), ld("trace_check.json"), ld("merge.json")
                why = []
                if not (g.get("check1", {}).get("pass") and g.get("check2", {}).get("pass")):
                    why.append(f"gate {g.get('check1', {}).get('pass')}/{g.get('check2', {}).get('pass')}")
                if not t.get("all_pass") or t.get("n") != EXPECTED:
                    why.append(f"trace {t.get('all_pass')} over {t.get('n')}")
                if m.get("merged") != EXPECTED:
                    why.append(f"merged {m.get('merged')}")
                if why:
                    rej[dd.name] = why
                    continue
                rows = [json.loads(l) for l in (dd / "rows.jsonl").open(encoding="utf-8")]
                for r in rows:
                    fo.write(json.dumps(r, default=str) + "\n")
                n += len(rows)
                acc.append(dd.name)
    write_json(R9 / "veins" / f"{mode}_rows_merge.json", {"provenance": stamp(f"revision9/veins/assemble/{mode}", {}),
                                                          "shards_accepted": len(acc), "shards_rejected": rej,
                                                          "rows": n})
    print(f"{mode}: accepted {len(acc)} / {len(CELLS) * len(SEEDS_B2)} shards, {n} rows; rejected {rej}")
    return 0 if not rej else 1


def load(path) -> list[dict]:
    out = []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)["row"]
        run, v = r["knobs"].rsplit("_d", 1)
        out.append({"scenario": r["scenario"], "density": r["density"], "seed": r["seed"], "policy": run,
                     "value": float(v), "rwcr": r["rwcr"], "tx_per_at_risk_informed": r["tx_per_at_risk_informed"],
                     "actionable_deadline_miss_rate": r["actionable_deadline_miss_rate"], "tir_p95_s": r["tir_p95_s"]})
    return out


def per_run(rows, seeds, th) -> dict:
    out = {}
    for arm, runs in ARMS.items():
        out[arm] = {}
        for run in runs:
            t = tables([r for r in rows if r["policy"] == run], CELLS, seeds, key=lambda r: (r["policy"], r["value"]))
            out[arm][run] = {}
            for c in CELLS:
                q, m = th[c]
                mc = matched(t[c], run, q, m)
                rw, co, mi = (t[c].mean(x) for x in ("rwcr", "cost", "miss"))
                b = int(np.nanargmax(rw))
                out[arm][run][f"{c[0]} d={c[1]:g}"] = {
                    "matched_cost": mc if math.isfinite(mc) else None, "qualifies": bool(math.isfinite(mc)),
                    "best_rwcr": float(rw[b]), "miss_at_best_rwcr": float(mi[b]), "cost_at_best_rwcr": float(co[b]),
                    "min_miss": float(np.nanmin(mi)),
                    "per_setting": {str(k[1]): {"rwcr": float(rw[i]), "cost": float(co[i]), "miss": float(mi[i])}
                                    for i, k in enumerate(t[c].settings)}}
    return out


def fisher_greater(aq, an, bq, bn) -> float:
    from scipy.stats import fisher_exact
    return float(fisher_exact([[aq, an - aq], [bq, bn - bq]], alternative="greater")[1])


def rule_table(pr) -> tuple[dict, float, bool]:
    cells = {}
    for sc, d in CELLS:
        ck = f"{sc} d={d:g}"
        nq = {arm: sum(pr[arm][r][ck]["qualifies"] for r in runs) for arm, runs in ARMS.items()}
        cells[ck] = {"rand_qualifying": nq["rand"], "plain_qualifying": nq["plain"],
                     "rule_met": bool(nq["rand"] >= 3 and nq["plain"] < 3),
                     "p_fisher_one_sided": fisher_greater(nq["rand"], 5, nq["plain"], 5)}
    p = min(1.0, 2 * min(c["p_fisher_one_sided"] for c in cells.values()))
    return cells, p, any(c["rule_met"] for c in cells.values())


def h6() -> dict:
    th = frozen()
    pr = per_run(load(R9 / "veins" / "h6_rows.jsonl"), SEEDS_B2, th)
    cells, p, met = rule_table(pr)
    return {"hypothesis": "H6 (post-hoc motivated; only test in its family)", "seeds": SEEDS_B2,
            "thresholds": {f"{c[0]} d={c[1]:g}": th[c] for c in CELLS}, "cells": cells, "rule_met_any_cell": met,
            "p_H6": p, "supported": bool(met and p < 0.05), "per_run": pr}


def c1() -> dict:
    th = frozen()
    rows_b2 = load(R9 / "veins" / "c1_rows.jsonl")
    pr2 = per_run(rows_b2, SEEDS_B2, th)
    cells2, p2, met2 = rule_table(pr2)
    rows_b1 = load(R8 / "veins" / "h5B_rows.jsonl")                         # revision-8 H5 rows, seeds 210-219
    pr_all = per_run(rows_b1 + rows_b2, list(range(210, 230)), th)
    cells_all, p_all, met_all = rule_table(pr_all)
    return {"what": "descriptive completion of revision-8 H5 scope (D12 cut); the registered revision-8 H5 verdict "
                    "(seeds 210-219, results/revision8/analysis/h5.json) stays the result of record",
            "seeds_220_229": {"cells": cells2, "rule_met_any_cell": met2, "p": p2, "per_run": pr2},
            "combined_210_229_descriptive": {"cells": cells_all, "rule_met_any_cell": met_all, "p": p_all,
                                             "per_run": pr_all}}


def s1() -> dict:
    """STRUCT9-rand vs revision-8 STRUCT-rand on the same Veins seeds 220-229, per cell (descriptive)."""
    th = frozen()
    pr9 = per_run(load(R9 / "veins" / "h6_rows.jsonl"), SEEDS_B2, th)["rand"]
    pr8 = per_run(load(R9 / "veins" / "c1_rows.jsonl"), SEEDS_B2, th)["rand"]
    out = {}
    for sc, d in CELLS:
        ck = f"{sc} d={d:g}"
        def summ(pr):
            v = [pr[r][ck] for r in ARMS["rand"]]
            return {"qualifying": sum(x["qualifies"] for x in v),
                    "best_rwcr_per_seed": [x["best_rwcr"] for x in v],
                    "miss_at_best_rwcr_per_seed": [x["miss_at_best_rwcr"] for x in v],
                    "cost_at_best_rwcr_per_seed": [x["cost_at_best_rwcr"] for x in v],
                    "min_miss_per_seed": [x["min_miss"] for x in v],
                    "mean_best_rwcr": float(np.mean([x["best_rwcr"] for x in v])),
                    "mean_miss_at_best_rwcr": float(np.mean([x["miss_at_best_rwcr"] for x in v])),
                    "mean_cost_at_best_rwcr": float(np.mean([x["cost_at_best_rwcr"] for x in v])),
                    "mean_min_miss": float(np.mean([x["min_miss"] for x in v]))}
        out[ck] = {"q_star": th[(sc, d)][0], "m_star": th[(sc, d)][1], "struct9_rand": summ(pr9),
                   "revision8_struct_rand": summ(pr8)}
    return {"what": "S1 (descriptive): does the deadline term lower miss, and at what reach and cost?", "seeds": SEEDS_B2,
            "cells": out}


def main(argv=None) -> int:
    a = list(sys.argv[1:] if argv is None else argv)
    (R9 / "analysis").mkdir(parents=True, exist_ok=True)
    if a[0] == "assemble":
        return assemble(a[1])
    payload = {"h6": h6, "c1": c1, "s1": s1}[a[0]]()
    p = R9 / "analysis" / f"{a[0]}.json"
    write_json(p, {"provenance": stamp(f"revision9/analysis/{a[0]}", {}), **payload})
    print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
