"""Revision 8 (ADDENDUM section 6): freeze the Veins q*, m* for H5 from the committed Tier A classical replays.

    python -m experiments.revision8.veins_thresholds      -> results/revision8/veins_thresholds.json

Per headline cell: q* = 0.95 x max mean RWCR over the 55 classical settings, m* = min mean actionable miss + 0.05, on
seeds 200-209 of results/revision5/veins/veins_rows.jsonl (revision-5 rows, as registered in revision 6). Applied
unchanged to the Tier B STRUCT replays. Same definition and data as revision 6, so the values equal revision 6's Tier A.
"""

from __future__ import annotations

import json

import numpy as np

from common.config import PROJECT_ROOT
from experiments.revision5.analysis.common import HEADLINE, tables

SEEDS = list(range(200, 210))
SRC = PROJECT_ROOT / "results" / "revision5" / "veins" / "veins_rows.jsonl"


def main() -> int:
    rows = []
    for line in SRC.open(encoding="utf-8"):
        r = json.loads(line)["row"]
        rows.append({**r, "value": r["knob"]})
    t = tables(rows, HEADLINE, SEEDS, key=lambda r: (r["policy"], float(r["value"])))
    out = {}
    for c in HEADLINE:
        assert len(t[c].settings) == 55, (c, len(t[c].settings))
        rw, mi = t[c].mean("rwcr"), t[c].mean("miss")
        out[f"{c[0]}|{c[1]:g}"] = {"q_star": 0.95 * float(np.nanmax(rw)), "m_star": float(np.nanmin(mi)) + 0.05,
                                   "ceiling_setting": list(t[c].settings[int(np.nanargmax(rw))]),
                                   "n_settings": 55, "seeds": SEEDS}
    from experiments.revision1.provenance import stamp, write_json
    p = PROJECT_ROOT / "results" / "revision8" / "veins_thresholds.json"
    write_json(p, {"provenance": stamp("revision8/veins_thresholds", {"source": str(SRC.name)}, seeds=SEEDS),
                   "rule": "q* = 0.95 x max mean RWCR, m* = min mean actionable miss + 0.05, 55 classical settings, "
                           "Veins Tier A 200-209; frozen, applied to Tier B (ADDENDUM section 6)", "cells": out})
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
