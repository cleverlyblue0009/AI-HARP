"""Revision 9 S2 (descriptive): STRUCT9-plain and STRUCT9-rand in the custom simulator with the revision-8 H4
comparator and analysis code (experiments/revision8/analysis/h4.py analyse_tier, unchanged), Tier B 210-229 (and Tier A
200-209), four headline cells, 5 training seeds per arm (below H4's 10; a cell qualifies if >= 3 of 5 seeds pick on SEL).
Labelled descriptive: no test, no Holm. Written before any STRUCT9 row existed.

    python -m experiments.revision9.analysis.s2 [--B 10000]      -> results/revision9/analysis/s2.json
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from common.config import PROJECT_ROOT
from experiments.revision1.provenance import stamp, write_json
from experiments.revision1.runner import load_rows
from experiments.revision5.analysis.common import HEADLINE, SEL, rows, tables, thresholds
from experiments.revision8.analysis.h4 import RNG, TIERS, analyse_tier

R9 = PROJECT_ROOT / "results" / "revision9"


def struct9_tables(arm: str, seeds, ss: str) -> dict:
    th = {}
    for k in range(5):
        d = json.loads((R9 / "train" / f"{arm}_s{k}" / "theta.json").read_text())
        th[json.dumps([round(v, 12) for v in d["theta"]])] = f"{arm}_s{k}"
    allrows = load_rows(R9 / "runs" / f"struct9_{arm}__{ss}__headline__default__fallback.csv")
    out = {}
    for key, run in th.items():
        out[run] = tables(allrows, HEADLINE, seeds,
                          key=lambda r: (float(r["value"]),),
                          where=lambda r, key=key: json.dumps([round(v, 12) for v in
                                                               json.loads(r["params_json"])["theta"]]) == key)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--B", type=int, default=10_000)
    a = ap.parse_args(argv)
    th = thresholds()
    csel = tables(rows("classical__SEL__grid24__default__fallback.csv"), HEADLINE, SEL)
    out = {"what": "S2, descriptive (5 training seeds per arm; H4 used 10)", "arms": {}}
    for arm in ("plain", "rand"):
        runs = [f"{arm}_s{k}" for k in range(5)]
        asel = struct9_tables(arm, SEL, "SEL")
        out["arms"][arm] = {}
        for tier, seeds in TIERS.items():
            aev = struct9_tables(arm, seeds, "EVAL")
            cev = tables(rows("classical__EVAL__grid24__default__fallback.csv"), HEADLINE, seeds)
            out["arms"][arm][tier] = analyse_tier(runs, asel, aev, csel, cev, th, len(seeds), a.B,
                                                  np.random.default_rng(RNG))
    p = R9 / "analysis" / "s2.json"
    write_json(p, {"provenance": stamp("revision9/analysis/s2", {"B": a.B}), **out})
    print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
