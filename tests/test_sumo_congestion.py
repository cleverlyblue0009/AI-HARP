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
    assert f'speed="{plan.v_eq_ms:.2f}"' in edges
    sr._build_highway_network(scn, tools, tmp_path, sr.congestion_plan(scn, 20.0))
    assert 'id="xf"' not in (tmp_path / "corridor.edg.xml").read_text()


def test_congested_routes_prepopulate_the_commanded_density(tmp_path):
    scn = load_scenario("rural_highway")
    routes = sr._write_routes(scn, 80.0, 0, tmp_path, 1.0).read_text()
    n = len(re.findall(r"<vehicle ", routes))
    geo = scn["geometry"]
    expected = 80.0 * geo["length_m"] / 1000.0 * geo["lanes_per_direction"] * 2
    assert n == pytest.approx(expected, rel=0.02)
    assert 'edges="e0' in routes and "xf" in routes and "xb" in routes
    free = sr._write_routes(scn, 20.0, 0, tmp_path, 1.0).read_text()
    assert "<vehicle " not in free and "xf" not in free


def test_exit_sections_are_masked_out_of_the_trace():
    x = np.array([[-100.0, 50.0, 10050.0]])
    tr = sr.mask_to_corridor(_trace(x, np.zeros((1, 3)), np.ones((1, 3)), np.zeros((1, 3))), 10000.0)
    assert tr.active.tolist() == [[False, True, False]]
    assert np.isnan(tr.x[0, 0]) and tr.x[0, 1] == 50.0
