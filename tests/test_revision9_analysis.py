"""Revision 9: the Veins analysis code reproduces revision 8's registered H5 counts from revision 8's committed rows
(stand-in data; checks the machinery only)."""

from __future__ import annotations

import json

from common.config import PROJECT_ROOT


def test_veins9_reproduces_revision8_h5_counts():
    from experiments.revision9.analysis import veins9 as v
    rows = v.load(PROJECT_ROOT / "results" / "revision8" / "veins" / "h5B_rows.jsonl")
    pr = v.per_run(rows, list(range(210, 220)), v.frozen())
    cells, p, met = v.rule_table(pr)
    ref = json.loads((PROJECT_ROOT / "results" / "revision8" / "analysis" / "h5.json").read_text())["tier_B_primary"]
    for ck, c in cells.items():
        r = ref["cells"][ck]
        assert (c["rand_qualifying"], c["plain_qualifying"], c["rule_met"]) == \
            (r["rand_qualifying"], r["plain_qualifying"], r["rule_met"])
        assert c["p_fisher_one_sided"] == r["p_fisher_one_sided"]
    assert p == ref["p_H5"] and met == ref["rule_met_any_cell"]
