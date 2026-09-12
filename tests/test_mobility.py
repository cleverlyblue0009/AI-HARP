"""Phase 1 tests: the fallback mobility generator and the trace container."""

from __future__ import annotations

import numpy as np
import pytest

from common.seeding import make_rng
from mobility.fallback import generate_fallback_trace
from mobility.generate import load_scenario
from mobility.trace import load_trace, save_trace


@pytest.fixture(scope="module")
def short_scenario() -> dict:
    s = load_scenario("rural_highway")
    s["simulation"]["duration_s"] = 10.0
    s["simulation"]["warmup_s"] = 5.0
    s["geometry"]["length_m"] = 2000.0
    return s


@pytest.fixture(scope="module")
def trace(short_scenario):
    return generate_fallback_trace(short_scenario, 20.0, make_rng(0, "mobility"))


def test_density_is_held_at_the_commanded_value(trace, short_scenario):
    """Density is the independent variable of the sweep; it must not drift."""
    target = 20.0 * (short_scenario["geometry"]["length_m"] / 1000.0) * 2  # both directions
    counts = trace.active.sum(axis=1)
    assert counts.mean() == pytest.approx(target, rel=0.05)
    assert counts.min() > 0.9 * target


@pytest.mark.parametrize("density", [5.0, 20.0, 40.0])
def test_density_scales_the_vehicle_count(short_scenario, density):
    tr = generate_fallback_trace(short_scenario, density, make_rng(0, "mobility"))
    expected = density * (short_scenario["geometry"]["length_m"] / 1000.0) * 2
    assert tr.active.sum(axis=1).mean() == pytest.approx(expected, rel=0.08)


def test_trace_shape_and_resolution(trace, short_scenario):
    assert trace.dt == pytest.approx(0.1)
    assert trace.n_steps == int(short_scenario["simulation"]["duration_s"] / 0.1)
    for arr in (trace.x, trace.y, trace.vx, trace.vy, trace.heading, trace.lane, trace.active):
        assert arr.shape == (trace.n_steps, trace.n_vehicles)


def test_vehicles_stay_inside_the_corridor(trace, short_scenario):
    x = trace.x[trace.active]
    assert x.min() >= -1.0
    assert x.max() <= short_scenario["geometry"]["length_m"] + 1.0


def test_both_carriageways_are_populated(trace):
    assert (trace.direction == 1).sum() > 0
    assert (trace.direction == -1).sum() > 0


def test_velocity_sign_matches_carriageway(trace):
    moving = trace.active & (np.abs(trace.vx) > 0.5)
    d = np.broadcast_to(trace.direction, trace.vx.shape)
    assert np.all(np.sign(trace.vx[moving]) == d[moving])


def test_heading_is_consistent_with_direction(trace):
    fwd = trace.direction == 1
    assert np.allclose(trace.heading[0, fwd], 0.0)
    assert np.allclose(trace.heading[0, ~fwd], np.pi)


def test_speeds_stay_within_the_configured_envelope(trace, short_scenario):
    classes = short_scenario["vehicles"]["classes"]
    dev = short_scenario["vehicles"]["speed_dev"]
    vmax = max(c["speed_kmh_max"] for c in classes.values()) / 3.6 * (1 + 2 * dev)
    speed = np.hypot(trace.vx, trace.vy)[trace.active]
    assert speed.min() >= 0.0
    assert speed.max() <= vmax + 1e-6


def test_trucks_are_present_in_the_configured_share(trace, short_scenario):
    share = short_scenario["vehicles"]["classes"]["truck"]["share"]
    assert (trace.vclass == "truck").mean() == pytest.approx(share, abs=0.10)


def test_car_following_prevents_overlap(trace):
    """No vehicle may occupy the same space as its leader on the same lane."""
    worst = np.inf
    for step in range(0, trace.n_steps, 10):
        for lane in np.unique(trace.lane[step][trace.active[step]]):
            sel = trace.active[step] & (trace.lane[step] == lane)
            xs = np.sort(trace.x[step][sel])
            if xs.size > 1:
                worst = min(worst, float(np.diff(xs).min()))
    # Bumper-to-bumper spacing must exceed a car length; a violation means the
    # Krauss safe-velocity rule is not binding.
    assert worst > 3.0


def test_generation_is_deterministic_under_a_fixed_seed(short_scenario):
    a = generate_fallback_trace(short_scenario, 20.0, make_rng(3, "mobility"))
    b = generate_fallback_trace(short_scenario, 20.0, make_rng(3, "mobility"))
    assert np.array_equal(np.nan_to_num(a.x), np.nan_to_num(b.x))
    assert np.array_equal(a.active, b.active)


def test_different_seeds_give_different_traces(short_scenario):
    a = generate_fallback_trace(short_scenario, 20.0, make_rng(1, "mobility"))
    b = generate_fallback_trace(short_scenario, 20.0, make_rng(2, "mobility"))
    assert not np.array_equal(np.nan_to_num(a.x), np.nan_to_num(b.x))


def test_backend_is_labelled_on_the_trace(trace):
    assert trace.backend == "fallback"


def test_roundtrip_through_npz(trace, tmp_path):
    p = tmp_path / "t.npz"
    save_trace(trace, p)
    back = load_trace(p)
    assert np.array_equal(np.nan_to_num(back.x), np.nan_to_num(trace.x))
    assert np.array_equal(back.active, trace.active)
    assert back.backend == trace.backend
    assert back.meta["density_veh_km_lane"] == trace.meta["density_veh_km_lane"]


def test_to_records_matches_the_brief_layout(trace):
    rec = trace.to_records()
    assert rec.dtype.names == (
        "timestep", "vehicle_id", "x", "y", "vx", "vy", "heading", "lane"
    )
    assert rec.size == int(trace.active.sum())


def test_grid_scenario_generates(short_scenario):
    g = load_scenario("urban_grid")
    g["simulation"]["duration_s"] = 5.0
    g["simulation"]["warmup_s"] = 2.0
    g["geometry"]["grid_rows"] = 3
    g["geometry"]["grid_cols"] = 3
    tr = generate_fallback_trace(g, 20.0, make_rng(0, "mobility"))
    assert tr.meta["kind"] == "grid"
    assert tr.active.sum(axis=1).mean() > 0
    assert tr.routes is not None
