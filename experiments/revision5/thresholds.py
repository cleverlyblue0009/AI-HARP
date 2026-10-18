"""Freeze q* and m* per cell from the CAL classical sweep (pre-registered; protocol.json target_and_guard).

    python -m experiments.revision5.thresholds

q*(cell) = 0.95 x max over the 55 classical settings of the mean oracle RWCR over CAL seeds;
m*(cell) = min over the same settings of the mean actionable deadline miss rate, + 0.05;
absolute variant: same q*, miss <= 0.40. Refuses to run unless every CAL row exists, and never
overwrites an existing thresholds.json (it is frozen before any SEL/EVAL run).
"""

from __future__ import annotations

import math
from collections import defaultdict

from experiments.revision1.provenance import out_dir, stamp, write_json
from experiments.revision1.runner import load_rows
from experiments.revision1.sweeps import knob_settings
from experiments.revision5.sweep import cells, seeds

SRC = out_dir("runs", 5) / "classical__CAL__grid24__default__fallback.csv"
OUT = out_dir("", 5) / "thresholds.json"


def main() -> int:
    if OUT.exists():
        raise SystemExit(f"{OUT} exists and is frozen; not overwriting")
    cl, sd, ks = cells("grid24"), seeds("CAL"), knob_settings()
    expected = len(cl) * len(sd) * len(ks)
    rows = load_rows(SRC)
    if len(rows) != expected:
        raise SystemExit(f"{SRC.name}: {len(rows)} rows, expected {expected}; CAL sweep not finished")
    acc = defaultdict(lambda: ([], []))
    for r in rows:
        k = (r["scenario"], float(r["density"]), r["policy"], r["param"], r["value"])
        acc[k][0].append(float(r["rwcr"]))
        acc[k][1].append(float(r["actionable_deadline_miss_rate"]))
    out = {}
    for sc, d in cl:
        means = {k[2:]: (sum(q) / len(q), sum(m) / len(m), len(q))
                 for k, (q, m) in acc.items() if k[:2] == (sc, d)}
        if len(means) != len(ks) or any(n != len(sd) for *_, n in means.values()):
            raise SystemExit(f"{sc} d={d}: incomplete settings/seeds")
        bad = [k for k, (q, m, _) in means.items() if math.isnan(q) or math.isnan(m)]
        if bad:
            raise SystemExit(f"{sc} d={d}: NaN means for {bad}")
        (qs, qm), (ms, mm) = (max(((k, v[0]) for k, v in means.items()), key=lambda t: t[1]),
                              min(((k, v[1]) for k, v in means.items()), key=lambda t: t[1]))
        out[f"{sc}|{d:g}"] = {"scenario": sc, "density": d, "ceiling_rwcr": qm, "ceiling_setting": list(qs),
                              "q_star": 0.95 * qm, "best_miss": mm, "best_miss_setting": list(ms),
                              "m_star": mm + 0.05, "m_star_absolute": 0.40}
    write_json(OUT, {"provenance": stamp("revision5/thresholds", {"source": SRC.name, "rows": len(rows)},
                                         seeds=sd, n_runs=len(rows)),
                     "rule": "q* = 0.95 x max mean oracle RWCR; m* = min mean actionable miss + 0.05 (CAL)",
                     "cells": out})
    print(f"wrote {OUT} ({len(out)} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
