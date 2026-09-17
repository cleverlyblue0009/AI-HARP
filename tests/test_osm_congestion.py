"""Dense traffic on a real (imported) OSM highway."""

from __future__ import annotations

import math
import re

import pytest

import mobility.sumo_runner as sr
from mobility.generate import load_scenario
from mobility.sumo_osm import write_osm_routes
from tests.test_sumo_osm import _edge


class _Tools:
    sumo = "sumo"
    sumo_home = None


def _chain_edges(n_edges=3, edge_len=5000.0, lanes=1):
    """A -> B -> C -> D and the reverse, as on the imported US-50 network."""
    pts = [(i * edge_len, 0.0) for i in range(n_edges + 1)]
    nodes = "ABCD"[: n_edges + 1]
    fwd = [_edge(f"-{i}", nodes[i], nodes[i + 1], pts[i], pts[i + 1], lanes=lanes)
           for i in range(n_edges)]
    rev = [_edge(f"{i}", nodes[i + 1], nodes[i], pts[i + 1], pts[i], lanes=lanes)
           for i in reversed(range(n_edges))]
    return fwd + rev


def _write(tmp_path, density):
    scn = load_scenario("rural_highway_osm")
    plan = sr.congestion_plan(scn, density)
    routes = write_osm_routes(scn, density, 0, tmp_path, 1.0, tmp_path / "net.net.xml",
                              _chain_edges(), _Tools(), 100.0, plan=plan)
    return scn, plan, routes.read_text()


def test_free_flow_osm_highway_is_unchanged(tmp_path):
    scn, plan, text = _write(tmp_path, 20.0)
    assert not plan.congested
    assert "<vehicle " not in text and 'departSpeed="max"' in text
    assert not (tmp_path / sr.VSS_FILE).exists()


def test_congested_osm_highway_is_prepopulated_and_speed_limited(tmp_path):
    scn, plan, text = _write(tmp_path, 80.0)
    assert plan.congested

    # inflow k * v_eq * lanes, inserted at v_eq (not free-flow spacing)
    flows = [float(v) for v in re.findall(r'vehsPerHour="([0-9.]+)"', text)]
    lanes = scn["geometry"]["lanes_per_direction"]
    assert sum(flows) / 2 == pytest.approx(80.0 * plan.v_eq_ms * 3.6 * lanes, rel=0.01)
    assert f'departSpeed="{plan.v_eq_ms:.2f}"' in text

    # the whole 15 km route (2 x 3 edges of 5 km) pre-placed at ~80 veh/km/lane
    n = len(re.findall(r"<vehicle ", text))
    expected = 80.0 * 15.0 * lanes * 2
    assert 0.9 * expected <= n <= expected

    # every pre-placed vehicle sits wholly on its edge
    rows = re.findall(r'<vehicle id="p\d+_\d+_\d+" type="(\w+)"[^>]*departPos="([0-9.]+)"', text)
    lengths = {k: c["length_m"] for k, c in scn["vehicles"]["classes"].items()}
    assert rows and all(float(pos) >= lengths[cls] - 0.05 for cls, pos in rows)

    # route lanes limited to v_eq, since imported edges cannot be rewritten
    vss = (tmp_path / sr.VSS_FILE).read_text()
    assert vss.count("<variableSpeedSign") == 2
    assert f'speed="{plan.v_eq_ms:.2f}"' in vss
    assert all(f'{e}_0' in vss for e in ("-0", "-1", "-2", "0", "1", "2"))


def test_equilibrium_speed_matches_the_synthetic_corridor(tmp_path):
    osm = sr.congestion_plan(load_scenario("rural_highway_osm"), 80.0)
    synthetic = sr.congestion_plan(load_scenario("rural_highway"), 80.0)
    assert math.isclose(osm.v_eq_ms, synthetic.v_eq_ms, rel_tol=1e-9)
