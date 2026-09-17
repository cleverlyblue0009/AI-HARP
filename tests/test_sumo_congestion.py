"""Congested synthetic SUMO corridors: plan, network, pre-population, masking."""

from __future__ import annotations

import re

import numpy as np
import pytest

import mobility.sumo_runner as sr
from mobility.generate import load_scenario
from tests.test_osm_geometry import _trace


def test_free_flow_density_needs_no_bottleneck():
    plan = sr.congestion_plan(load_scenario("rural_highway"), 20.0)
    assert not plan.congested and plan.v_eq_ms > plan.v_free_ms


def test_dense_corridor_is_held_near_the_fallbacks_measured_speed():
    plan = sr.congestion_plan(load_scenario("rural_highway"), 80.0)
    assert plan.congested
    assert plan.spacing_m == pytest.approx(12.5)
    # fallback d=80 traffic measured 14.8 km/h; Krauss equilibrium gives ~14 km/h
    assert 10.0 <= plan.v_eq_ms * 3.6 <= 20.0


def test_congested_network_gets_exit_bottlenecks(monkeypatch, tmp_path):
    monkeypatch.setattr(sr, "_run", lambda *a, **k: None)
    tools = sr.SumoTools(sumo="sumo", netconvert="netconvert", netgenerate="netgenerate", sumo_home=None)
    scn = load_scenario("rural_highway")
    plan = sr.congestion_plan(scn, 80.0)
    sr._build_highway_network(scn, tools, tmp_path, plan)
    edges = (tmp_path / "corridor.edg.xml").read_text()
    assert 'id="xf"' in edges and 'id="xb"' in edges
    speeds = set(re.findall(r'speed="([0-9.]+)"', edges))
    assert speeds == {f"{plan.v_eq_ms:.2f}"}          # corridor AND exits held at v_eq
    sr._build_highway_network(scn, tools, tmp_path, sr.congestion_plan(scn, 20.0))
    free = (tmp_path / "corridor.edg.xml").read_text()
    assert 'id="xf"' not in free and f'speed="{plan.v_eq_ms:.2f}"' not in free


def test_congested_routes_prepopulate_the_commanded_density(tmp_path):
    scn = load_scenario("rural_highway")
    routes = sr._write_routes(scn, 80.0, 0, tmp_path, 1.0).read_text()
    n = len(re.findall(r"<vehicle ", routes))
    geo = scn["geometry"]
    expected = 80.0 * geo["length_m"] / 1000.0 * geo["lanes_per_direction"] * 2
    # Slightly under the command by construction: every gap carries a 0.5 m
    # insertion margin and vehicles that would straddle a junction move back to
    # the previous edge (measured: 1,538 of 1,600 = 76.9 veh/km/lane; the
    # fallback's own d=80 traces hold 77.9). Never over.
    assert 0.95 * expected <= n <= expected
    assert 'edges="e0' in routes and "xf" in routes and "xb" in routes
    plan = sr.congestion_plan(scn, 80.0)
    flows = [float(v) for v in re.findall(r'vehsPerHour="([0-9.]+)"', routes)]
    # per direction, summed over vehicle classes: k * v_eq[km/h] * lanes
    assert sum(flows) / 2 == pytest.approx(80.0 * plan.v_eq_ms * 3.6 * geo["lanes_per_direction"], rel=0.01)
    assert f'departSpeed="{plan.v_eq_ms:.2f}"' in routes
    free = sr._write_routes(scn, 20.0, 0, tmp_path, 1.0).read_text()
    assert "<vehicle " not in free and "xf" not in free and 'departSpeed="max"' in free


def test_prepopulated_vehicles_never_overlap_even_behind_trucks(tmp_path):
    """Uniform 12.5 m spacing cannot fit a 12 m truck + 2.5 m minGap: SUMO refused
    those insertions and rural d=80 recorded 22.9 veh/km/lane."""
    scn = load_scenario("rural_highway")
    routes = sr._write_routes(scn, 80.0, 0, tmp_path, 1.0).read_text()
    veh = scn["vehicles"]
    lengths = {n: c["length_m"] for n, c in veh["classes"].items()}
    plan = sr.congestion_plan(scn, 80.0)
    rows = re.findall(r'<vehicle id="p(-?1)_\d+_(\d+)" type="(\w+)" depart="0" departPos="([0-9.]+)" '
                      r'departLane="(\d+)"[^>]*><route edges="([^ "]+)', routes)
    assert rows
    by_edge: dict[tuple, list] = {}
    for d, _, cls, pos, lane, edge in rows:
        by_edge.setdefault((edge, lane), []).append((float(pos), lengths[cls]))
    needed = veh["min_gap_m"] + plan.v_eq_ms * veh["reaction_time_s"]
    for items in by_edge.values():
        items.sort()
        assert all(p >= length - 0.05 for p, length in items)      # wholly on its edge
        for (p0, _), (p1, l1) in zip(items, items[1:]):
            # departPos is the vehicle's front: the leader (p1) occupies
            # [p1 - l1, p1], so the follower's front must be >= minGap + v*tau
            # behind that. Positions are written to 0.1 m.
            assert p1 - p0 >= l1 + needed - 0.1


def test_exit_sections_are_masked_out_of_the_trace():
    x = np.array([[-100.0, 50.0, 10050.0]])
    tr = sr.mask_to_corridor(_trace(x, np.zeros((1, 3)), np.ones((1, 3)), np.zeros((1, 3))), 10000.0)
    assert tr.active.tolist() == [[False, True, False]]
    assert np.isnan(tr.x[0, 0]) and tr.x[0, 1] == 50.0
