"""Revision 8 (ADDENDUM section 4): the STRUCT warm start, chosen on SEL only.

    python -m experiments.revision8.warm_start      -> results/revision8/warm_start.json

Among the settings STRUCT represents exactly -- slotted_1p n_slots in {1, 2, 5, 50} and the 27 RAR settings -- take the
one that qualifies on SEL (mean RWCR >= q*, mean actionable miss <= m*, D = infinity; frozen revision-5 thresholds) in the
most headline cells; ties by mean SEL cost over those cells, then by the lower setting index (slotted 1, 2, 5, 50, then
RAR in protocol.json grid order). Rows: results/revision5/runs/{classical,rule}__SEL__grid24__default__fallback.csv.
"""

from __future__ import annotations

import json

import numpy as np

from agents.structured import theta_rar, theta_slotted
from common.config import PROJECT_ROOT
from experiments.revision5.analysis.common import HEADLINE, SEL, rows, tables, thresholds
from experiments.revision5.sweep import rule_settings


def candidates() -> list[tuple[tuple, list[float]]]:
    out = [(("slotted_1p", "n_slots", str(n)), theta_slotted(n).tolist()) for n in (1, 2, 5, 50)]
    for pol, par, val in rule_settings():
        if pol == "rar":
            a, b, c = json.loads(val)
            out.append((("rar", "abc", val), theta_rar(a, b, c).tolist()))
    return out


def main() -> int:
    th = thresholds()
    cls = tables(rows("classical__SEL__grid24__default__fallback.csv"), HEADLINE, SEL)
    rul = tables(rows("rule__SEL__grid24__default__fallback.csv"), HEADLINE, SEL)

    def lookup(key, c):
        t = cls[c] if key[0] == "slotted_1p" else rul[c]
        for i, k in enumerate(t.settings):
            if (k[0], k[1], str(k[2])) == key or (key[0] == "slotted_1p" and k[0] == key[0] and k[1] == key[1]
                                                  and float(k[2]) == float(key[2])):
                return float(t.mean("rwcr")[i]), float(t.mean("miss")[i]), float(t.mean("cost")[i])
        raise KeyError((key, c))

    table = []
    for idx, (key, theta) in enumerate(candidates()):
        per = {}
        for c in HEADLINE:
            rw, mi, co = lookup(key, c)
            q, m = th[c]
            per[f"{c[0]} d={c[1]:g}"] = {"rwcr": rw, "miss": mi, "cost": co, "qualifies": bool(rw >= q and mi <= m)}
        qc = [v["cost"] for v in per.values() if v["qualifies"]]
        table.append({"index": idx, "setting": list(key), "theta": theta, "n_qualify": len(qc),
                      "mean_cost_qualifying": float(np.mean(qc)) if qc else None, "per_cell": per})
    best = min(table, key=lambda r: (-r["n_qualify"], r["mean_cost_qualifying"] if r["mean_cost_qualifying"]
                                     is not None else np.inf, r["index"]))
    from experiments.revision1.provenance import stamp, write_json
    p = PROJECT_ROOT / "results" / "revision8" / "warm_start.json"
    write_json(p, {"provenance": stamp("revision8/warm_start", {"rule": "ADDENDUM section 4"}, seeds=SEL),
                   "chosen": best["setting"], "theta": best["theta"], "n_qualify": best["n_qualify"],
                   "mean_cost_qualifying": best["mean_cost_qualifying"], "candidates": table})
    print(p, best["setting"], best["n_qualify"], best["mean_cost_qualifying"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
