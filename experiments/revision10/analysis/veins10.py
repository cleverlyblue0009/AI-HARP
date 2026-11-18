"""Revision 10 Veins analyses (results/revision10/ADDENDUM.md): shard assembly, H7 (primary), S3, S4, S5.
Written before any replay on seeds 230-249.

    python -m experiments.revision10.analysis.veins10 assemble h7|s3|s5   shards -> results/revision10/veins/<mode>_rows.jsonl
    python -m experiments.revision10.analysis.veins10 h7                  -> results/revision10/analysis/h7.json
    python -m experiments.revision10.analysis.veins10 s3                  -> results/revision10/analysis/s3.json
    python -m experiments.revision10.analysis.veins10 s4                  -> results/revision10/analysis/s4.json
    python -m experiments.revision10.analysis.veins10 s5                  -> results/revision10/analysis/s5.json

Qualification exactly as revision-9 H6 (experiments/revision9/analysis/veins9.py): per training seed, the Algorithm-1
matched cost over its six deltas from the seed means, guard mean actionable miss <= m* (revision-6 veins_methods.matched),
qualifies iff finite. H7: urban NLOS d=20 on R-A only; rule rand >= 3/5 and plain < 3/5; one-sided Fisher exact
(veins9.fisher_greater); supported iff rule met and p < 0.05; single hypothesis, no correction.
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
from experiments.revision9.analysis.veins9 import ARMS, EXPECTED, fisher_greater, load

R8, R9, R10 = (PROJECT_ROOT / "results" / f"revision{k}" for k in (8, 9, 10))
URBAN, RURAL = ("urban_nlos", 20.0), ("rural_highway", 20.0)
SEEDS = json.loads((R10 / "seeds.json").read_text())
RA, RB = SEEDS["R_A"], SEEDS["R_B"]
H6_SEEDS = list(range(220, 230))
MODES = {"h7": (URBAN, RA), "s3": (URBAN, RB), "s5": (RURAL, RA)}
DIGESTS = R10 / "gates.json"


def ck(cell) -> str:
    return f"{cell[0]} d={cell[1]:g}"


def frozen(cell) -> tuple[float, float]:
    c = json.loads((R8 / "veins_thresholds.json").read_text())["cells"][f"{cell[0]}|{cell[1]:g}"]
    return c["q_star"], c["m_star"]


def assemble(mode: str) -> int:
    (sc, d), seeds = MODES[mode]
    acc, rej, n = [], {}, 0
    want = json.loads(DIGESTS.read_text())["bundles"]["per_bundle"]
    (R10 / "veins").mkdir(parents=True, exist_ok=True)
    with open(R10 / "veins" / f"{mode}_rows.jsonl", "w", encoding="utf-8") as fo:
        for s in seeds:
            dd = R10 / "veins" / mode / f"{sc}_d{d:g}_s{s}"
            ld = lambda f: json.loads((dd / f).read_text()) if (dd / f).exists() else {}  # noqa: E731
            g, t, m = ld("gate.json"), ld("trace_check.json"), ld("merge.json")
            why = []
            if not (g.get("check1", {}).get("pass") and g.get("check2", {}).get("pass")):
                why.append(f"gate {g.get('check1', {}).get('pass')}/{g.get('check2', {}).get('pass')}")
            if g.get("bundle", {}).get("content_sha256") != want.get(dd.name, {}).get("content_sha256"):
                why.append("bundle digest differs from gates.json")
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
    write_json(R10 / "veins" / f"{mode}_rows_merge.json", {"provenance": stamp(f"revision10/veins/assemble/{mode}", {}),
                                                           "shards_accepted": len(acc), "shards_rejected": rej,
                                                           "rows": n})
    print(f"{mode}: accepted {len(acc)} / {len(seeds)} shards, {n} rows; rejected {rej}")
    return 0 if not rej else 1


def per_run(rows, seeds, cell) -> dict:
    """{arm: {run: {...}}} for one cell -- the revision-9 per-run summary, one cell at a time."""
    q, m = frozen(cell)
    out = {}
    for arm, runs in ARMS.items():
        out[arm] = {}
        for run in runs:
            t = tables([r for r in rows if r["policy"] == run], (cell,), seeds,
                       key=lambda r: (r["policy"], r["value"]))[cell]
            mc = matched(t, run, q, m)
            rw, co, mi = (t.mean(x) for x in ("rwcr", "cost", "miss"))
            b = int(np.nanargmax(rw))
            out[arm][run] = {
                "matched_cost": mc if math.isfinite(mc) else None, "qualifies": bool(math.isfinite(mc)),
                "best_rwcr": float(rw[b]), "miss_at_best_rwcr": float(mi[b]), "cost_at_best_rwcr": float(co[b]),
                "min_miss": float(np.nanmin(mi)),
                "per_setting": {str(k[1]): {"rwcr": float(rw[i]), "cost": float(co[i]), "miss": float(mi[i])}
                                for i, k in enumerate(t.settings)}}
    return out


def rule(pr) -> dict:
    nq = {arm: sum(pr[arm][r]["qualifies"] for r in runs) for arm, runs in ARMS.items()}
    met = bool(nq["rand"] >= 3 and nq["plain"] < 3)
    p = fisher_greater(nq["rand"], 5, nq["plain"], 5)
    return {"rand_qualifying": nq["rand"], "plain_qualifying": nq["plain"], "rule_met": met, "p_fisher_one_sided": p}


def evaluate(rows, seeds, cell) -> dict:
    pr = per_run(rows, seeds, cell)
    q, m = frozen(cell)
    return {"cell": ck(cell), "seeds": list(seeds), "q_star": q, "m_star": m, **rule(pr), "per_run": pr}


def rows_of(mode: str) -> list[dict]:
    return load(R10 / "veins" / f"{mode}_rows.jsonl")


def h7() -> dict:
    e = evaluate(rows_of("h7"), RA, URBAN)
    return {"hypothesis": "H7 (replication of revision-9 H6 urban cell; single hypothesis, no correction)", **e,
            "p_H7": e["p_fisher_one_sided"], "supported": bool(e["rule_met"] and e["p_fisher_one_sided"] < 0.05),
            "h6_of_record": {"rand_qualifying": 4, "plain_qualifying": 0, "p": 0.023809523809523808,
                             "source": "results/revision9/analysis/h6.json"}}


def s3() -> dict:
    ra, rb = rows_of("h7"), rows_of("s3")
    return {"what": "S3 (descriptive): H7 analysis on R-B, and on R-A u R-B pooled (20 seeds)",
            "R_B": evaluate(rb, RB, URBAN), "pooled_R_A_R_B": evaluate(ra + rb, RA + RB, URBAN)}


def s4() -> dict:
    """Per training seed and delta: R-A means beside the H6 means on 220-229, and each seed's status in both."""
    pa = per_run(rows_of("h7"), RA, URBAN)
    p6 = per_run(load(R9 / "veins" / "h6_rows.jsonl"), H6_SEEDS, URBAN)
    out = {}
    for arm, runs in ARMS.items():
        for run in runs:
            a, b = pa[arm][run], p6[arm][run]
            out[run] = {"qualifies_R_A": a["qualifies"], "qualifies_H6": b["qualifies"],
                        "status_repeats": a["qualifies"] == b["qualifies"],
                        "matched_cost_R_A": a["matched_cost"], "matched_cost_H6": b["matched_cost"],
                        "per_delta": {dl: {"R_A": a["per_setting"][dl], "H6_220_229": b["per_setting"][dl]}
                                      for dl in a["per_setting"]}}
    return {"what": "S4 (descriptive): per training seed and delta, urban NLOS d=20, R-A vs H6 seeds 220-229",
            "cell": ck(URBAN), "m_star": frozen(URBAN)[1], "q_star": frozen(URBAN)[0],
            "status_repeats": sum(v["status_repeats"] for v in out.values()), "of": len(out), "per_run": out}


def s5() -> dict:
    e = evaluate(rows_of("s5"), RA, RURAL)
    q, m = frozen(RURAL)
    pattern = {run: {"best_rwcr": v["best_rwcr"], "miss_at_best_rwcr": v["miss_at_best_rwcr"],
                     "reach_without_timeliness": bool(v["best_rwcr"] >= q and v["miss_at_best_rwcr"] > m)}
               for arm in ARMS for run, v in e["per_run"][arm].items()}
    return {"what": "S5 (descriptive): rural d=20 on R-A, both arms", **e, "reach_without_timeliness": pattern}


def main(argv=None) -> int:
    a = list(sys.argv[1:] if argv is None else argv)
    (R10 / "analysis").mkdir(parents=True, exist_ok=True)
    if a[0] == "assemble":
        return assemble(a[1])
    payload = {"h7": h7, "s3": s3, "s4": s4, "s5": s5}[a[0]]()
    p = R10 / "analysis" / f"{a[0]}.json"
    write_json(p, {"provenance": stamp(f"revision10/analysis/{a[0]}", {}), **payload})
    print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
