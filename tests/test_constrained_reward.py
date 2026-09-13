"""The constrained objective: tree credit, the multiplier, targets, ranking.

The array-level tests are synthetic and torch-free. The last test runs real
short episodes, because the failure this replaces was only visible when whole
policies were scored -- a unit test of the formula would not have caught it.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from agents.constrained_reward import (
    ConstrainedObjective,
    CoverageTargets,
    LagrangeMultiplier,
    ancestor_path,
    causal_coverage,
    tree_credit,
    warned_at_risk,
)

# A chain: 0 (originator) -> 1 -> 2 -> 3.
CHAIN = np.array([-1, 0, 1, 2])


# ------------------------------------------------------------ tree credit ----
def test_ancestor_path_is_every_transmitter_the_warning_needed():
    assert ancestor_path(CHAIN, 3) == [2, 1, 0]
    assert ancestor_path(CHAIN, 1) == [0]
    assert ancestor_path(CHAIN, 0) == []


def test_credit_is_an_equal_shapley_split_along_the_path():
    peak = np.ones(4)
    at_risk = np.ones(4, dtype=bool)
    warned = np.ones(4, dtype=bool)
    c = tree_credit(CHAIN, peak, at_risk, warned)
    # u=1 -> [0]; u=2 -> [1,0] halves; u=3 -> [2,1,0] thirds.
    assert c == pytest.approx([1 + 1 / 2 + 1 / 3, 1 / 2 + 1 / 3, 1 / 3, 0.0])


def test_credit_sums_to_warned_coverage_without_double_counting():
    """No over-counting: total credit equals the warned non-originator mass."""
    peak = np.array([0.9, 0.5, 0.8, 0.3])
    at_risk = np.ones(4, dtype=bool)
    warned = np.ones(4, dtype=bool)
    c = tree_credit(CHAIN, peak, at_risk, warned)
    assert c.sum() == pytest.approx((peak[1:] / peak.mean()).sum())


def test_a_relay_whose_receivers_were_not_warned_earns_nothing():
    peak = np.ones(4)
    at_risk = np.array([True, True, False, False])
    warned = np.array([True, True, False, False])
    c = tree_credit(CHAIN, peak, at_risk, warned)
    assert c[2] == 0.0 and c[3] == 0.0


def test_silence_earns_no_credit():
    """Only the originator transmits: nobody else is warned by a decision."""
    informed_by = np.array([-1, -1, -1, -1])
    c = tree_credit(informed_by, np.ones(4), np.ones(4, dtype=bool),
                    np.array([True, False, False, False]))
    assert c.sum() == 0.0


def test_late_warnings_are_not_coverage():
    """A warning after the vehicle has passed the hazard counts for nothing."""
    rel = np.array([[0.0, 0.9], [0.0, 0.0]])        # vehicle 1 at risk only at t=0
    peak, at_risk, warned = warned_at_risk(rel, np.array([-1, 1]), 0.05)
    assert at_risk[1] and not warned[1]
    assert causal_coverage(peak, at_risk, warned) == 0.0


# -------------------------------------------------------------- multiplier ----
def test_shortfall_raises_lambda_and_surplus_lowers_it():
    m = LagrangeMultiplier(value=2.0, lr=1.0, max_value=50.0)
    assert m.update([0.5, 0.3]) == pytest.approx(2.4)
    assert m.update([-0.2]) == pytest.approx(2.2)


def test_lambda_is_clipped_to_its_bounds():
    m = LagrangeMultiplier(value=1.0, lr=10.0, max_value=5.0)
    m.update([1.0])
    assert m.value == 5.0 and m.saturated
    m.update([-10.0])
    assert m.value == 0.0


def test_non_finite_shortfalls_do_not_move_lambda():
    m = LagrangeMultiplier(value=3.0, lr=1.0, max_value=50.0)
    m.update([float("nan")])
    assert m.value == 3.0


# ----------------------------------------------------------------- targets ----
def test_target_is_a_fraction_of_the_measured_ceiling(tmp_path):
    key = CoverageTargets.key("rural_highway", 2, "clear", "fog_bank")
    p = tmp_path / "t.json"
    p.write_text(json.dumps({"cells": {key: {"ceiling": 0.7}}}))
    t = CoverageTargets.load(p, fraction=0.95, fallback=0.8)
    assert t.target("rural_highway", 2.0, "clear", "fog_bank") == pytest.approx(0.665)


def test_missing_cell_falls_back_visibly(tmp_path):
    t = CoverageTargets.load(tmp_path / "absent.json", fraction=0.95, fallback=0.8)
    assert t.target("urban_nlos", 40, "clear", "crash") == 0.8


def test_objective_reads_from_config():
    from common.config import load_yaml

    o = ConstrainedObjective.from_config(load_yaml("agent.yaml"))
    assert 0 < o.target_fraction <= 1
    assert 0 <= o.lambda_init <= o.lambda_max


# ------------------------------------------------ ranking on real episodes ---
def test_objective_ranks_real_policies_sanely():
    """The check that would have caught the old reward.

    Silence must be escapable below lambda_max, and just past that point an
    efficient scheme must beat flooding.
    """
    from analysis.reward_check import check, score_references
    from common.config import load_yaml

    cfgs = {"phy": load_yaml("phy.yaml"), "hazard": load_yaml("hazard.yaml"),
            "experiment": load_yaml("experiment.yaml")}
    objective = ConstrainedObjective.from_config(load_yaml("agent.yaml"))
    scores = score_references("rural_highway", 20.0, "clear", "fog_bank", 0, objective,
                              cfgs, duration_s=40.0, corridor_length_m=3000.0)
    verdict = check(scores, objective)
    silence = next(s for s in scores if s.label == "always-suppress")
    flooding = next(s for s in scores if s.label == "flooding")
    assert flooding.coverage > silence.coverage
    assert verdict.silence_escapable, f"break-even {verdict.break_even}"
    assert verdict.efficient_beats_flooding
