"""Real-map geometry: straight-window projection and grid rotation."""

from __future__ import annotations

import math

import numpy as np
import pytest

from common.config import load_yaml
from mobility.osm_geometry import (
    grid_edge_records, project_highway_trace, project_points, rotate_trace, straightest_window,
)
from mobility.trace import Trace


def _trace(x, y, vx, vy, active=None, meta=None):
    x = np.asarray(x, np.float32)
    T, N = x.shape
    return Trace(dt=0.1, x=x, y=np.asarray(y, np.float32), vx=np.asarray(vx, np.float32),
                 vy=np.asarray(vy, np.float32),
                 heading=np.arctan2(np.asarray(vy, float), np.asarray(vx, float)).astype(np.float32),
                 lane=np.zeros((T, N), np.int16),
                 active=np.ones((T, N), bool) if active is None else np.asarray(active, bool),
                 vehicle_ids=np.array([f"v{i}" for i in range(N)]),
                 vclass=np.array(["car"] * N), length_m=np.full(N, 4.5, np.float32),
                 direction=np.zeros(N, np.int8), scenario="t", backend="sumo",
                 meta=meta or {})


def test_straightest_window_skips_a_bend():
    # 3 km bent section, then 12 km straight along a 45 degree line
    bend = [(0, 0), (1000, 800), (2000, 0), (3000, 800)]
    u = np.array([1, 1]) / math.sqrt(2)
    straight = [np.array(bend[-1]) + u * s for s in np.linspace(500, 12000, 24)]
    a, b, dev = straightest_window(np.vstack([bend, straight]), 10000.0)
    assert dev < 1e-6
    assert np.hypot(*(b - a)) == pytest.approx(10000.0, rel=1e-6)
    assert a[0] >= 3000                        # starts past the bend


def test_projection_recovers_along_road_distance_and_offset():
    a, b = (100.0, 200.0), (100.0 + 600.0, 200.0 + 800.0)   # 1000 m chord, bearing 53.13 deg
    s, lat = project_points(np.array([400.0, 100.0]), np.array([600.0, 200.0]), a, b)
    assert s[0] == pytest.approx(500.0) and lat[0] == pytest.approx(0.0, abs=1e-9)
    assert s[1] == pytest.approx(0.0)
    s2, lat2 = project_points(np.array([100.0 - 8.0]), np.array([200.0 + 6.0]), a, b)
    assert lat2[0] == pytest.approx(10.0) and s2[0] == pytest.approx(0.0, abs=1e-9)


def test_projected_trace_is_a_straight_corridor():
    th = math.radians(40.0)
    u = np.array([math.cos(th), math.sin(th)])
    n = np.array([-u[1], u[0]])
    a = np.array([50.0, 50.0])
    b = a + 2000.0 * u
    # vehicle 0 drives +chord at 20 m/s, vehicle 1 -chord at 15 m/s, 3.5 m to the left;
    # vehicle 2 sits on a side road 200 m off the chord
    s0, s1 = np.array([100.0, 102.0, 104.0]), np.array([1500.0, 1498.5, 1497.0])
    p0 = a + s0[:, None] * u
    p1 = a + s1[:, None] * u + 3.5 * n
    p2 = a + 800 * u + 200 * n + np.zeros((3, 1))
    x = np.stack([p0[:, 0], p1[:, 0], p2[:, 0]], axis=1)
    y = np.stack([p0[:, 1], p1[:, 1], p2[:, 1]], axis=1)
    vx = np.stack([np.full(3, 20 * u[0]), np.full(3, -15 * u[0]), np.zeros(3)], axis=1)
    vy = np.stack([np.full(3, 20 * u[1]), np.full(3, -15 * u[1]), np.zeros(3)], axis=1)
    out = project_highway_trace(_trace(x, y, vx, vy), a, b, lateral_max_m=30.0, max_dev_m=0.0)
    assert np.allclose(out.x[:, 0], s0, atol=1e-3) and np.allclose(out.y[:, 0], 0.0, atol=1e-3)
    assert np.allclose(out.x[:, 1], s1, atol=1e-3) and np.allclose(out.y[:, 1], 3.5, atol=1e-3)
    assert np.allclose(out.vx[:, 0], 20.0, atol=1e-3) and np.allclose(out.vy[:, 0], 0.0, atol=1e-3)
    assert np.allclose(out.vx[:, 1], -15.0, atol=1e-3)
    assert out.direction.tolist()[:2] == [1, -1]
    assert not out.active[:, 2].any()          # side road dropped
    assert out.meta["kind"] == "highway" and out.meta["length_m"] == pytest.approx(2000.0)


def test_vehicles_outside_the_window_are_inactive():
    a, b = (0.0, 0.0), (1000.0, 0.0)
    x = np.array([[-50.0, 500.0, 1100.0]])
    out = project_highway_trace(_trace(x, np.zeros((1, 3)), np.ones((1, 3)), np.zeros((1, 3))), a, b)
    assert out.active.tolist() == [[False, True, False]]


def _rotated_grid(angle_deg):
    """A 3x3 block grid (200 m blocks) rotated by angle_deg, plus one diagonal street."""
    th = math.radians(angle_deg)
    rot = lambda p: (p[0] * math.cos(th) - p[1] * math.sin(th), p[0] * math.sin(th) + p[1] * math.cos(th))
    edges = []
    for i in range(4):
        for j in range(3):
            edges.append(((j * 200, i * 200), ((j + 1) * 200, i * 200)))     # along x
            edges.append(((i * 200, j * 200), (i * 200, (j + 1) * 200)))     # along y
    edges.append(((0, 0), (200, 200)))                                       # diagonal
    return [{"sumo_id": f"e{k}", "shape": [rot(p), rot(q)],
             "length": math.dist(p, q)} for k, (p, q) in enumerate(edges)]


def test_rotated_grid_comes_back_axis_aligned():
    recs, stats = grid_edge_records(_rotated_grid(61.0), theta_deg=-61.0)
    assert stats["n_edges_in"] == 25 and stats["n_edges_kept"] == 24     # diagonal dropped
    assert {r["axis"] for r in recs} == {"x", "y"}
    for r in recs:
        (x0, y0), (x1, y1) = r["p0"], r["p1"]
        if r["axis"] == "x":
            assert y0 == pytest.approx(y1, abs=1e-6)
        else:
            assert x0 == pytest.approx(x1, abs=1e-6)
    assert [r["id"] for r in recs] == list(range(len(recs)))


def test_rotate_trace_matches_edge_rotation():
    tr = _trace([[math.cos(math.radians(61)) * 100]], [[math.sin(math.radians(61)) * 100]],
                [[math.cos(math.radians(61)) * 10]], [[math.sin(math.radians(61)) * 10]])
    out = rotate_trace(tr, -61.0)
    assert out.x[0, 0] == pytest.approx(100.0, abs=1e-3) and out.y[0, 0] == pytest.approx(0.0, abs=1e-3)
    assert out.vx[0, 0] == pytest.approx(10.0, abs=1e-3) and out.heading[0, 0] == pytest.approx(0.0, abs=1e-5)


def test_grid_records_let_a_hazard_be_placed_and_located():
    from hazard.model import hazard_from_config
    from hazard.risk_field import build_risk_field

    recs, _ = grid_edge_records(_rotated_grid(61.0), theta_deg=-61.0)
    tr = _trace(np.zeros((2, 1)), np.zeros((2, 1)), np.ones((2, 1)), np.zeros((2, 1)),
                meta={"kind": "grid", "edges": recs})
    hz_cfg = load_yaml("hazard.yaml")
    hazard = hazard_from_config(hz_cfg, tr.meta, overrides={"type": "fog_bank"})
    assert hazard.edge_id is not None
    rf = build_risk_field(hz_cfg, tr, hazard)
    assert rf.geometry.hazard_xy is not None
