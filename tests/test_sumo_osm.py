"""OSM demand chains, SUMO trace post-processing and scenario inheritance."""

from __future__ import annotations

import math

import numpy as np
import pytest

from mobility.generate import load_scenario
from mobility.sumo_osm import highway_chains, postprocess_trace
from tests.test_osm_geometry import _rotated_grid, _trace


def _edge(eid, frm, to, p0, p1, etype="highway.primary", lanes=1):
    return {"sumo_id": eid, "from": frm, "to": to, "shape": [p0, p1], "type": etype,
            "lanes": lanes, "length": math.dist(p0, p1)}


def test_us50_style_edges_chain_into_two_carriageways():
    # A -> B -> C -> D and the reverse, as on the imported US-50 network
    pts = {"A": (0, 0), "B": (8000, 8000), "C": (9500, 10000), "D": (10500, 12000)}
    fwd = [_edge("-1", "A", "B", pts["A"], pts["B"]), _edge("-2", "B", "C", pts["B"], pts["C"]),
           _edge("-3", "C", "D", pts["C"], pts["D"])]
    rev = [_edge("3", "D", "C", pts["D"], pts["C"]), _edge("2", "C", "B", pts["C"], pts["B"]),
           _edge("1", "B", "A", pts["B"], pts["A"])]
    chains = highway_chains(fwd + rev)
    assert sorted(chains) == [["-1", "-2", "-3"], ["3", "2", "1"]]


def test_synthetic_sumo_grid_gets_edges_and_its_real_dimensions():
    scn = load_scenario("urban_nlos")
    edges = [{**e, "lanes": 1} for e in _rotated_grid(0.0)]
    tr = _trace(np.zeros((1, 1)), np.zeros((1, 1)), np.ones((1, 1)), np.zeros((1, 1)),
                meta={"kind": "grid"})
    out, lane_km = postprocess_trace(tr, scn, "synthetic", edges)
    assert lane_km is None
    assert out.meta["grid_rows"] == scn["geometry"]["grid_rows"] == 6     # not the 5x5 default
    assert out.meta["block_length_m"] == scn["geometry"]["block_length_m"]
    assert len(out.meta["edges"]) == 24 and out.meta["grid_edge_stats"]["n_edges_kept"] == 24


def test_osm_grid_is_rotated_and_lane_km_comes_from_the_network():
    scn = load_scenario("urban_grid_osm")
    edges = [{**e, "lanes": 2} for e in _rotated_grid(61.0)]
    th = math.radians(61.0)
    tr = _trace([[100 * math.cos(th)]], [[100 * math.sin(th)]], [[1.0]], [[0.0]], meta={"kind": "grid"})
    out, lane_km = postprocess_trace(tr, scn, "osm", edges)
    assert out.x[0, 0] == pytest.approx(100.0, abs=1e-3) and out.y[0, 0] == pytest.approx(0.0, abs=1e-3)
    assert lane_km == pytest.approx(24 * 200 * 2 / 1000.0)
    assert "grid_rows" not in out.meta


def test_osm_highway_is_projected_onto_its_straightest_window():
    scn = load_scenario("rural_highway_osm")
    th = math.radians(45.0)
    u = np.array([math.cos(th), math.sin(th)])
    edges = [_edge("-1", "A", "B", (0.0, 0.0), tuple(12000 * u)),
             _edge("1", "B", "A", tuple(12000 * u), (0.0, 0.0))]
    p = 3000 * u
    tr = _trace([[p[0]]], [[p[1]]], [[20 * u[0]]], [[20 * u[1]]], meta={"kind": "highway"})
    out, lane_km = postprocess_trace(tr, scn, "osm", edges)
    assert out.meta["kind"] == "highway" and out.meta["length_m"] == pytest.approx(10000.0)
    assert out.meta["projection"]["max_polyline_deviation_m"] == pytest.approx(0.0, abs=1e-6)
    assert 0.0 <= out.x[0, 0] <= 10000.0 and out.vx[0, 0] == pytest.approx(20.0, abs=1e-3)
    assert lane_km == pytest.approx(10.0 * 1 * 2)


def test_osm_scenarios_inherit_their_base_and_name_a_phy_profile():
    rural, base = load_scenario("rural_highway_osm"), load_scenario("rural_highway")
    assert rural["name"] == "rural_highway_osm" and rural["phy_profile"] == "rural_highway"
    assert rural["vehicles"] == base["vehicles"]
    assert rural["sumo"]["osm_extract"].endswith("rural_us50_nevada.osm.xml")
    assert rural["sumo"]["step_length"] == base["sumo"]["step_length"]
    assert "extends" not in rural
    urban = load_scenario("urban_grid_osm")
    assert urban["kind"] == "grid" and urban["geometry"]["grid_rotation_deg"] == -61.0


def test_osm_scenario_refuses_the_fallback_backend():
    from mobility.generate import get_trace

    with pytest.raises(RuntimeError, match="SUMO backend"):
        get_trace(load_scenario("rural_highway_osm"), 20.0, 0, backend="fallback")
