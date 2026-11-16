"""Revision 8: the H4 analysis runs end to end on stand-in data (committed revision-5 RAR rows posing as ten STRUCT
runs x six knob values). This checks the machinery only; it uses no STRUCT result."""

from __future__ import annotations

import json
import math

import numpy as np

from experiments.revision5.analysis.common import HEADLINE, SEL, rows, tables, thresholds


def _standin(seeds, seedset):
    rr = rows(f"rule__{seedset}__grid24__default__fallback.csv")
    out = {}
    for k in range(10):
        a = (0.5, 1.0, 2.0)[k % 3]
        keep = {json.dumps([a, b, c]) for b in (0.0, 0.5, 1.0) for c in (-0.25, 0.0)}
        out[f"plain_s{k}"] = tables(rr, HEADLINE, seeds, key=lambda r: (r["value"],),
                                    where=lambda r, keep=keep: r["policy"] == "rar" and r["value"] in keep)
    return out


def test_h4_machinery_on_standin_data():
    from experiments.revision8.analysis import h4
    th = thresholds()
    runs = [f"plain_s{k}" for k in range(10)]
    asel = _standin(SEL, "SEL")
    seeds = h4.TIERS["B"]
    aev = _standin(seeds, "EVAL")
    csel = tables(rows("classical__SEL__grid24__default__fallback.csv"), HEADLINE, SEL)
    cev = tables(rows("classical__EVAL__grid24__default__fallback.csv"), HEADLINE, seeds)
    res = h4.analyse_tier(runs, asel, aev, csel, cev, th, len(seeds), 30, np.random.default_rng(1))
    assert set(res["decision"]) == {"i_qualification_count", "ii_noninferiority_10pct", "rule_met"}
    assert 0 <= res["qualifying_cells"]["struct"] <= 4 and 0 <= res["qualifying_cells"]["comparator"] <= 4
    assert res["B"] == 30 and 0.0 <= res["p_noninferiority"] <= 1.0
    if res["T"] is None:
        assert res["p_noninferiority"] == 1.0 and not res["decision"]["ii_noninferiority_10pct"]
    else:
        assert isinstance(res["T"], float) and math.isfinite(res["T"]) or res["T"] == "inf"
