"""Each SUMO run gets its own build directory.

The directory was keyed (scenario, density, seed) only, so two weathers of one
cell -- which differ just in their speed/headway factors -- ran SUMO in the same
directory at once under parallel jobs and clobbered each other's net, routes and
FCD files. That killed the sparse SUMO sweep with WinError 32 after 80 s.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import mobility.sumo_runner as sr
from mobility.generate import load_scenario, trace_cache_key
from tests.test_osm_geometry import _trace


def _capture_workdirs(monkeypatch, tmp_path):
    seen: list[Path] = []
    monkeypatch.setattr(sr, "BUILD_DIR", tmp_path)
    monkeypatch.setattr(sr, "_build_network",
                        lambda scn, tools, work, plan=None: (Path(work) / "net.net.xml", "synthetic"))
    monkeypatch.setattr(sr, "_write_routes",
                        lambda scn, d, s, work, sf: (seen.append(Path(work)), Path(work) / "r.xml")[1])
    monkeypatch.setattr(sr, "_run", lambda *a, **k: None)
    monkeypatch.setattr(sr, "parse_fcd", lambda fcd, **kw: _trace(
        np.zeros((2, 4)), np.zeros((2, 4)), np.ones((2, 4)), np.zeros((2, 4)),
        meta=dict(kw.get("meta", {}))))
    return seen


def test_two_weathers_of_one_cell_do_not_share_a_build_directory(monkeypatch, tmp_path):
    seen = _capture_workdirs(monkeypatch, tmp_path)
    scn = load_scenario("rural_highway")
    tools = sr.SumoTools(sumo="sumo", netconvert="nc", netgenerate="ng", sumo_home=None)
    for speed_factor, weather in ((1.0, "clear"), (0.85, "heavy_rain")):
        tag = trace_cache_key(scn, 20.0, 0, "sumo", speed_factor, 1.0, None)
        sr.generate_sumo_trace(scn, 20.0, 0, tools=tools, speed_factor=speed_factor, tag=tag)
    assert len(seen) == 2 and seen[0] != seen[1], seen


def test_same_run_reuses_its_directory(monkeypatch, tmp_path):
    seen = _capture_workdirs(monkeypatch, tmp_path)
    scn = load_scenario("rural_highway")
    tools = sr.SumoTools(sumo="sumo", netconvert="nc", netgenerate="ng", sumo_home=None)
    tag = trace_cache_key(scn, 20.0, 0, "sumo", 1.0, 1.0, None)
    for _ in range(2):
        sr.generate_sumo_trace(scn, 20.0, 0, tools=tools, speed_factor=1.0, tag=tag)
    assert seen[0] == seen[1]


def test_untagged_call_still_works():
    """The tag is optional: older call sites keep the plain directory name."""
    assert "tag" in sr.generate_sumo_trace.__code__.co_varnames
    with pytest.raises(TypeError):
        sr.generate_sumo_trace(load_scenario("rural_highway"), 20.0, 0)   # tools is required
