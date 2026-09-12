"""The oracle risk field: ground-truth at-risk labelling (evaluation only)."""

from __future__ import annotations

import numpy as np
import pytest

from hazard.oracle import (
    NEVER,
    build_oracle_risk_field,
    estimation_agreement,
    oracle_horizon_s,
)
from hazard.risk_field import build_risk_field
from mobility.generate import get_trace
from hazard.model import hazard_from_config


@pytest.fixture(scope="module")
def rural(hz_cfg):
    tr = get_trace("rural_highway", 20, seed=0)
    hz = hazard_from_config(hz_cfg, tr.meta)
    risk = build_risk_field(hz_cfg, tr, hz)
    return tr, hz, risk, build_oracle_risk_field(risk, hz)


def test_horizon_is_derived_from_the_kernel_not_chosen(hz_cfg, rural):
    """H is where the risk field's own kernel decays below its own at-risk
    threshold -- so oracle and causal ask the same question over the same
    window, rather than the oracle having an arbitrary lookahead."""
    _, hz, risk, oracle = rural
    expected = risk.eta_full_s + risk.eta_decay_tau_s * np.log(
        hz.severity0 / risk.at_risk_threshold
    )
    assert oracle_horizon_s(risk, hz) == pytest.approx(expected, rel=1e-6)
    assert oracle.horizon_s > risk.eta_full_s


def test_horizon_is_bounded_not_infinite():
    """An unbounded lookahead would inflate the at-risk denominator exactly as
    the broken causal version did."""
    from hazard.model import DecayKind, DirectionRelevance, Hazard, HazardType

    from hazard.risk_field import HighwayRiskGeometry, RiskField

    risk = RiskField(
        geometry=HighwayRiskGeometry(10_000.0), eta_full_s=60.0, eta_decay_tau_s=60.0,
        inside_span_weight=1.0, min_speed_ms=1.0, severity_gamma=1.0,
        at_risk_threshold=0.05, high_relevance_threshold=0.5,
        opposing_direction_relevance=0.0,
    )
    hz = Hazard(
        hazard_id="h", htype=HazardType.CRASH, severity0=1.0, span_start_m=5000.0,
        span_end_m=5100.0, onset_time_s=0.0, decay=DecayKind.LINEAR_TTL, ttl_s=30.0,
        direction_relevance=DirectionRelevance.BOTH, affected_direction=0,
        blocks_road=True, detection_range_m=150.0, safety_deadline_s=5.0,
    )
    # A hazard that expires in 30 s cannot threaten anyone beyond its own life.
    assert np.isfinite(oracle_horizon_s(risk, hz))


def test_encounter_times_are_finite_only_for_vehicles_that_reach_the_hazard(rural):
    tr, hz, _, oracle = rural
    enc = oracle.encounter_time_s(tr, hz)
    assert enc.shape == (tr.n_vehicles,)
    assert np.isfinite(enc).any(), "no vehicle ever reaches the hazard"
    assert (enc == NEVER).any(), "every vehicle reaches the hazard -- gate not working"


def test_oracle_relevance_is_zero_after_the_encounter(rural):
    """A warning delivered after the vehicle is already in the hazard is worth
    nothing, which is the whole point of 'informed in time'."""
    tr, hz, _, oracle = rural
    enc = oracle.encounter_time_s(tr, hz)
    mat = oracle.relevance_matrix(tr, hz)
    reached = np.flatnonzero(np.isfinite(enc))[:20]
    for v in reached:
        step = int(enc[v] / tr.dt)
        if step + 5 < tr.n_steps:
            assert mat[step + 5, v] == 0.0


def test_oracle_relevance_rises_towards_the_encounter(hz_cfg):
    """Closer to the encounter means more relevant -- holding severity fixed.

    Uses a landslide (persistent, no decay) deliberately: with a decaying fog
    bank, severity falls faster than the ETA kernel rises inside its plateau,
    so relevance legitimately *decreases* as the encounter approaches. That is
    correct behaviour, not a bug, and conflating the two effects is what made
    the first version of this test fail.
    """
    tr = get_trace("rural_highway", 20, seed=0)
    hz = hazard_from_config(hz_cfg, tr.meta, overrides={"type": "landslide"})
    risk = build_risk_field(hz_cfg, tr, hz)
    oracle = build_oracle_risk_field(risk, hz)

    enc = oracle.encounter_time_s(tr, hz)
    mat = oracle.relevance_matrix(tr, hz)
    # Pick a vehicle whose encounter is far enough out that the early sample
    # sits beyond the kernel plateau and is therefore actually rising.
    far = np.flatnonzero(np.isfinite(enc) & (enc > risk.eta_full_s + 40))
    assert far.size, "no vehicle encounters the hazard beyond the ETA plateau"
    v = int(far[0])
    step = int(enc[v] / tr.dt)
    early, late = max(0, step - 900), max(0, step - 5)
    assert mat[late, v] >= mat[early, v] - 1e-9


def test_oracle_relevance_stays_in_unit_interval(rural):
    tr, hz, _, oracle = rural
    mat = oracle.relevance_matrix(tr, hz)
    assert np.all((mat >= 0.0) & (mat <= 1.0))


def test_causal_recall_is_high_on_a_corridor(rural):
    """On a straight road heading determines destiny, so the causal estimate
    should not MISS at-risk vehicles. Precision is lower -- it over-warns."""
    tr, hz, risk, oracle = rural
    ag = estimation_agreement(tr, hz, risk, oracle)
    assert ag["risk_est_recall"] > 0.95
    assert ag["risk_peak_corr"] > 0.7


def test_estimation_agreement_reports_both_set_sizes(rural):
    tr, hz, risk, oracle = rural
    ag = estimation_agreement(tr, hz, risk, oracle)
    assert ag["n_at_risk_oracle"] > 0
    assert ag["n_at_risk_causal"] > 0
    assert 0.0 <= ag["risk_est_precision"] <= 1.0
