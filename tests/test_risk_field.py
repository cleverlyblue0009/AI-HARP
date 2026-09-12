"""Phase 3 tests: the risk field.

These encode the behavioural claims the paper makes about relevance. If any of
them break, the headline metric (RWCR) no longer means what the paper says it
means, so they are the most important tests in the repo.
"""

from __future__ import annotations

import numpy as np
import pytest

from hazard.model import DecayKind, DirectionRelevance, Hazard, HazardType, hazard_from_config
from hazard.risk_field import APPROACHING, AWAY, INSIDE, HighwayRiskGeometry, RiskField


@pytest.fixture
def risk(hz_cfg) -> RiskField:
    rf = hz_cfg["risk_field"]
    return RiskField(
        geometry=HighwayRiskGeometry(length_m=10_000.0),
        eta_full_s=float(rf["eta_full_s"]),
        eta_decay_tau_s=float(rf["eta_decay_tau_s"]),
        inside_span_weight=float(rf["inside_span_weight"]),
        min_speed_ms=float(rf["min_speed_ms"]),
        severity_gamma=float(rf["severity_gamma"]),
        at_risk_threshold=float(rf["at_risk_threshold"]),
        high_relevance_threshold=float(rf["high_relevance_threshold"]),
        opposing_direction_relevance=float(rf["opposing_direction_relevance"]),
    )


def make_hazard(**kw) -> Hazard:
    base = dict(
        hazard_id="h", htype=HazardType.FOG_BANK, severity0=1.0,
        span_start_m=5000.0, span_end_m=5800.0, onset_time_s=0.0,
        decay=DecayKind.NONE, direction_relevance=DirectionRelevance.BOTH,
        affected_direction=0, blocks_road=False, detection_range_m=150.0,
        safety_deadline_s=6.0,
    )
    base.update(kw)
    return Hazard(**base)


def rel(risk: RiskField, hazard: Hazard, x: float, v: float, direction: int, t: float = 10.0) -> float:
    out = risk.evaluate(
        np.array([x]), np.array([0.0]), np.array([v * direction]), np.array([0.0]),
        np.array([direction]), hazard, t,
    )
    return float(out["relevance"][0])


# ------------------------------------------------------- the headline claims --
def test_truck_40s_upstream_of_severe_fog_has_relevance_near_one(risk):
    """The build brief's own example: a truck 40 s upstream scores ~1."""
    h = make_hazard(severity0=1.0)
    v = 20.0
    x = h.span_start_m - 40.0 * v          # exactly 40 s out
    assert rel(risk, h, x, v, +1) == pytest.approx(1.0, abs=1e-9)


def test_vehicle_moving_away_from_a_landslide_has_relevance_zero(risk):
    """The brief's other example: driving away from a landslide scores ~0."""
    h = make_hazard(htype=HazardType.LANDSLIDE, decay=DecayKind.NONE, severity0=1.0)
    # Sitting upstream of the span but travelling in the -x direction, i.e.
    # away from it.
    assert rel(risk, h, h.span_start_m - 500.0, 25.0, -1) == 0.0


def test_vehicle_that_has_passed_the_hazard_has_relevance_zero(risk):
    h = make_hazard()
    assert rel(risk, h, h.span_end_m + 300.0, 25.0, +1) == 0.0


def test_relevance_is_one_anywhere_inside_the_eta_plateau(risk):
    h = make_hazard(severity0=1.0)
    v = 25.0
    for eta in (1.0, 10.0, 30.0, 59.9):
        assert rel(risk, h, h.span_start_m - eta * v, v, +1) == pytest.approx(1.0, abs=1e-9)


def test_relevance_decays_beyond_the_plateau(risk):
    h = make_hazard(severity0=1.0)
    v = 25.0
    r120 = rel(risk, h, h.span_start_m - 120.0 * v, v, +1)
    r180 = rel(risk, h, h.span_start_m - 180.0 * v, v, +1)
    # 60 s and 120 s past the plateau, with tau = 60 s.
    assert r120 == pytest.approx(np.exp(-1.0), rel=1e-6)
    assert r180 == pytest.approx(np.exp(-2.0), rel=1e-6)
    assert r180 < r120 < 1.0


def test_far_upstream_vehicle_is_relevant_but_not_urgent(risk):
    h = make_hazard(severity0=1.0)
    r = rel(risk, h, h.span_start_m - 300.0 * 25.0, 25.0, +1)
    assert 0.0 < r < 0.05


# ------------------------------------------------------------------ extent --
def test_hazard_is_a_span_not_a_point(risk):
    h = make_hazard(span_start_m=5000.0, span_end_m=5800.0)
    inside = risk.evaluate(
        np.array([5100.0, 5400.0, 5700.0]), np.zeros(3), np.full(3, 20.0), np.zeros(3),
        np.ones(3, dtype=int), h, 10.0,
    )
    assert np.all(inside["state"] == INSIDE)
    assert np.allclose(inside["relevance"], risk.inside_span_weight * h.severity0)


def test_approach_distance_is_measured_to_the_near_edge(risk):
    h = make_hazard(span_start_m=5000.0, span_end_m=5800.0)
    fwd = risk.evaluate(np.array([4000.0]), np.zeros(1), np.array([20.0]), np.zeros(1),
                        np.array([1]), h, 10.0)
    bwd = risk.evaluate(np.array([6800.0]), np.zeros(1), np.array([-20.0]), np.zeros(1),
                        np.array([-1]), h, 10.0)
    assert fwd["distance_m"][0] == pytest.approx(1000.0)   # to span_start
    assert bwd["distance_m"][0] == pytest.approx(1000.0)   # to span_end
    assert fwd["state"][0] == APPROACHING and bwd["state"][0] == APPROACHING


# --------------------------------------------------------------- direction --
def test_directional_hazard_ignores_the_opposing_carriageway(risk):
    h = make_hazard(
        htype=HazardType.CRASH, direction_relevance=DirectionRelevance.DIRECTIONAL,
        affected_direction=+1,
    )
    on = rel(risk, h, h.span_start_m - 500.0, 25.0, +1)
    off = rel(risk, h, h.span_end_m + 500.0, 25.0, -1)
    assert on > 0.9
    assert off == 0.0


def test_both_direction_hazard_affects_both_carriageways(risk):
    h = make_hazard(direction_relevance=DirectionRelevance.BOTH, affected_direction=0)
    assert rel(risk, h, h.span_start_m - 500.0, 25.0, +1) > 0.9
    assert rel(risk, h, h.span_end_m + 500.0, 25.0, -1) > 0.9


# ---------------------------------------------------------------- severity --
def test_relevance_scales_with_severity(risk):
    v, offset = 25.0, 500.0
    r_hi = rel(risk, make_hazard(severity0=1.0), 5000.0 - offset, v, +1)
    r_lo = rel(risk, make_hazard(severity0=0.4), 5000.0 - offset, v, +1)
    assert r_lo == pytest.approx(0.4 * r_hi)


def test_dissipating_fog_stops_generating_relevance(risk):
    h = make_hazard(decay=DecayKind.EXPONENTIAL, tau_s=100.0, severity0=1.0, onset_time_s=0.0)
    early = rel(risk, h, 4500.0, 25.0, +1, t=0.0)
    late = rel(risk, h, 4500.0, 25.0, +1, t=500.0)
    assert early == pytest.approx(1.0)
    assert late < 0.01


def test_landslide_never_decays(risk):
    h = make_hazard(htype=HazardType.LANDSLIDE, decay=DecayKind.NONE, severity0=1.0)
    assert rel(risk, h, 4500.0, 25.0, +1, t=0.0) == rel(risk, h, 4500.0, 25.0, +1, t=10_000.0)


def test_hazard_generates_no_relevance_before_onset(risk):
    h = make_hazard(onset_time_s=50.0)
    assert rel(risk, h, 4500.0, 25.0, +1, t=10.0) == 0.0
    assert rel(risk, h, 4500.0, 25.0, +1, t=60.0) > 0.9


def test_linear_ttl_reaches_exactly_zero(hz_cfg):
    h = make_hazard(decay=DecayKind.LINEAR_TTL, ttl_s=100.0, severity0=0.8)
    assert h.severity_at(0.0) == pytest.approx(0.8)
    assert h.severity_at(50.0) == pytest.approx(0.4)
    assert h.severity_at(100.0) == pytest.approx(0.0)
    assert h.severity_at(200.0) == pytest.approx(0.0)
    assert not h.is_active(150.0)


# -------------------------------------------------------------- edge cases --
def test_stopped_vehicle_does_not_look_irrelevant(risk):
    """A jammed vehicle has ETA -> inf unless the min-speed floor applies."""
    h = make_hazard()
    r = rel(risk, h, h.span_start_m - 30.0, 0.0, +1)
    assert r > 0.0


def test_relevance_is_always_in_unit_interval(risk):
    h = make_hazard(severity0=0.9)
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 10_000, 5000)
    v = rng.uniform(0, 35, 5000)
    d = rng.choice([-1, 1], 5000)
    out = risk.evaluate(x, np.zeros(5000), v * d, np.zeros(5000), d, h, 20.0)
    assert np.all((out["relevance"] >= 0.0) & (out["relevance"] <= 1.0))


def test_at_risk_set_uses_peak_relevance(risk):
    # [T, N]: vehicle 0 is briefly relevant, vehicle 1 never is.
    mat = np.array([[0.0, 0.0], [0.9, 0.01], [0.0, 0.0]])
    peak, at_risk = risk.at_risk_set(mat)
    assert peak.tolist() == [0.9, 0.01]
    assert at_risk.tolist() == [True, False]


def test_state_codes_partition_the_corridor(risk):
    h = make_hazard(span_start_m=5000.0, span_end_m=5800.0)
    x = np.array([1000.0, 5400.0, 9000.0])
    out = risk.evaluate(x, np.zeros(3), np.full(3, 20.0), np.zeros(3), np.ones(3, int), h, 5.0)
    assert out["state"].tolist() == [APPROACHING, INSIDE, AWAY]


def test_hazard_from_config_places_span_around_position_frac(hz_cfg):
    h = hazard_from_config(hz_cfg, {"length_m": 10_000.0, "kind": "highway"})
    frac = hz_cfg["default_instance"]["position_frac"]
    extent = hz_cfg["default_instance"]["extent_m"]
    assert h.span_mid_m == pytest.approx(frac * 10_000.0)
    assert h.span_length_m == pytest.approx(extent)


def test_hazard_rejects_invalid_severity():
    with pytest.raises(ValueError):
        make_hazard(severity0=1.5)


def test_hazard_rejects_inverted_span():
    with pytest.raises(ValueError):
        make_hazard(span_start_m=6000.0, span_end_m=5000.0)
