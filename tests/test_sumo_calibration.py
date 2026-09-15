"""SUMO grid demand calibration, exercised without SUMO."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import mobility.sumo_runner as sr
from mobility.generate import load_scenario
from tests.test_osm_geometry import _trace


def _fake_pipeline(monkeypatch, tmp_path, kind_scale_to_density):
    """Patch every SUMO-touching step; density achieved = f(demand_scale)."""
    import mobility.osm_geometry as og
    import mobility.sumo_osm as so

    scales: list[float] = []
    monkeypatch.setattr(sr, "BUILD_DIR", tmp_path)
    monkeypatch.setattr(sr, "_build_network",
                        lambda scn, tools, work, plan=None: (Path(work) / "net.net.xml", "synthetic"))
    monkeypatch.setattr(og, "read_net_edges", lambda *a, **k: [])
    monkeypatch.setattr(sr, "_run", lambda *a, **k: None)

    def fake_routes(*a, demand_scale=1.0, **k):
        scales.append(demand_scale)
        return Path(a[3]) / "demand.rou.xml"          # (scenario, density, seed, work, ...)

    monkeypatch.setattr(so, "write_osm_routes", fake_routes)
    monkeypatch.setattr(sr, "_write_routes", lambda *a, **k: (scales.append(1.0), Path(a[3]) / "demand.rou.xml")[1])

    def fake_parse(fcd, **kw):
        n = int(round(kind_scale_to_density(scales[-1])))
        return _trace(np.zeros((3, n)), np.zeros((3, n)), np.ones((3, n)), np.zeros((3, n)),
                      meta=dict(kw.get("meta", {})))

    monkeypatch.setattr(sr, "parse_fcd", fake_parse)
    monkeypatch.setattr(so, "postprocess_trace", lambda tr, scn, src, edges: (tr, 1.0))   # 1 lane-km
    tools = sr.SumoTools(sumo="sumo", netconvert="netconvert", netgenerate="netgenerate", sumo_home=None)
    return scales, tools


def test_overshooting_grid_is_rerun_once_with_scaled_demand(monkeypatch, tmp_path):
    scn = load_scenario("urban_nlos")
    scales, tools = _fake_pipeline(monkeypatch, tmp_path, lambda s: 33.5 * s)
    tr = sr.generate_sumo_trace(scn, 20.0, 0, tools=tools)
    assert len(scales) == 2
    # the fake network holds round(33.5) = 34 vehicles on 1 lane-km -> 34 veh/km/lane
    assert scales[1] == pytest.approx(20.0 / 34.0, rel=1e-6)
    cal = tr.meta["demand_calibration"]
    assert [a["achieved_density_veh_km_lane"] for a in cal] == [34.0, 20.0]


def test_congested_grid_is_interpolated_on_the_third_run(monkeypatch, tmp_path):
    """Density not proportional to demand: 33.5 * s^1.6 undershoots after one step."""
    scales, tools = _fake_pipeline(monkeypatch, tmp_path, lambda s: 33.5 * s ** 1.6)
    tr = sr.generate_sumo_trace(load_scenario("urban_nlos"), 20.0, 0, tools=tools)
    cal = tr.meta["demand_calibration"]
    assert len(scales) == 3 == len(cal)
    assert abs(cal[1]["achieved_density_veh_km_lane"] - 20.0) / 20.0 > sr.CALIBRATION_TOLERANCE
    assert abs(cal[2]["achieved_density_veh_km_lane"] - 20.0) / 20.0 <= sr.CALIBRATION_TOLERANCE


def test_grid_within_tolerance_is_not_rerun(monkeypatch, tmp_path):
    scales, tools = _fake_pipeline(monkeypatch, tmp_path, lambda s: 21.0 * s)
    tr = sr.generate_sumo_trace(load_scenario("urban_nlos"), 20.0, 0, tools=tools)
    assert scales == [1.0] and len(tr.meta["demand_calibration"]) == 1


def test_highway_is_never_rerun(monkeypatch, tmp_path):
    scales, tools = _fake_pipeline(monkeypatch, tmp_path, lambda s: 40.0 * s)
    sr.generate_sumo_trace(load_scenario("rural_highway"), 20.0, 0, tools=tools)
    assert scales == [1.0]


def test_lane_km_override_and_grid_formula():
    grid = load_scenario("urban_nlos")
    g = grid["geometry"]
    assert sr._lane_km(grid) == pytest.approx(g["block_length_m"] * g["grid_rows"] * g["grid_cols"] * 4 / 1000)
    assert sr._lane_km(grid, 12.5) == 12.5
