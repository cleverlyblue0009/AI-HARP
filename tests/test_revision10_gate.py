"""Revision 10: the bundle content digest ignores only SUMO comment headers and the provenance stamp; the pre-registered
Fisher table (results/revision10/ADDENDUM.md) is what the revision-9 test function returns."""

import json

import numpy as np
import pytest

from experiments.revision10.veins.gate10 import digest
from experiments.revision9.analysis.veins9 import fisher_greater


def _bundle(d, comment="<!-- generated on 2026-11-17 by x -->", prov="t1", pos=1.0):
    d.mkdir()
    (d / "net.net.xml").write_text(f'<?xml version="1.0"?>\n{comment}\n<net><edge id="e" x="{pos}"/></net>\n')
    np.savez_compressed(d / "fcd_ref.npz", t=np.arange(3.0), x=np.array([pos, 2.0, 3.0]))
    (d / "r6.txt").write_text("risk grid 1.0\n")
    b = {"trace_sha256": "abc", "t0_sumo_s": 1.0, "vehicles": [{}], "n_steps": 3, "files": {"net.net.xml": "raw"},
         "veins": {"maxInterfDist_m": 1.0, "playgroundSizeX_m": 1, "playgroundSizeY_m": 1}, "provenance": {"t": prov}}
    (d / "bundle.json").write_text(json.dumps(b))
    return d


def test_digest_ignores_comments_and_provenance_only(tmp_path):
    a = digest(_bundle(tmp_path / "a"))
    b = digest(_bundle(tmp_path / "b", comment="<!-- generated on 2027-01-01 at /other/path -->", prov="t2"))
    c = digest(_bundle(tmp_path / "c", pos=1.5))
    assert a["content_sha256"] == b["content_sha256"]
    assert a["content_sha256"] != c["content_sha256"]


@pytest.mark.parametrize("rand,plain,p", [(3, 0, 0.0833), (4, 0, 0.0238), (5, 0, 0.0040), (5, 1, 0.0238), (4, 1, 0.1032)])
def test_preregistered_fisher_values(rand, plain, p):
    assert round(fisher_greater(rand, 5, plain, 5), 4) == p
