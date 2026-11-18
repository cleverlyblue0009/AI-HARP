"""Revision 10: the H7 code, run on the committed revision-9 H6 rows (seeds 220-229), reproduces h6.json's urban cell
exactly (counts, p, per-run matched costs and qualification)."""

import json

from common.config import PROJECT_ROOT
from experiments.revision10.analysis import veins10 as v10
from experiments.revision9.analysis.veins9 import load

R9 = PROJECT_ROOT / "results" / "revision9"


def test_h7_code_reproduces_h6_urban_cell():
    h6 = json.loads((R9 / "analysis" / "h6.json").read_text())
    e = v10.evaluate(load(R9 / "veins" / "h6_rows.jsonl"), list(range(220, 230)), v10.URBAN)
    ref = h6["cells"]["urban_nlos d=20"]
    assert (e["rand_qualifying"], e["plain_qualifying"], e["rule_met"]) == \
        (ref["rand_qualifying"], ref["plain_qualifying"], ref["rule_met"]) == (4, 0, True)
    assert e["p_fisher_one_sided"] == ref["p_fisher_one_sided"]
    assert (e["q_star"], e["m_star"]) == tuple(h6["thresholds"]["urban_nlos d=20"])
    for arm in ("plain", "rand"):
        for run, v in e["per_run"][arm].items():
            r = h6["per_run"][arm][run]["urban_nlos d=20"]
            assert v["qualifies"] == r["qualifies"] and v["matched_cost"] == r["matched_cost"]
            assert v["per_setting"] == r["per_setting"]
            assert (v["best_rwcr"], v["miss_at_best_rwcr"], v["min_miss"]) == \
                (r["best_rwcr"], r["miss_at_best_rwcr"], r["min_miss"])


def test_seed_sets_are_the_preregistered_ones():
    assert v10.RA == list(range(230, 240)) and v10.RB == list(range(240, 250))
    assert v10.MODES["h7"] == (("urban_nlos", 20.0), v10.RA)
