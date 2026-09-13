"""Phase 7b tests: backend-comparison verdicts, and metric directions they rely on.

No SUMO needed -- these exercise the verdict logic on synthetic seed samples
shaped like the committed rural d=20 comparison.
"""

from __future__ import annotations

import numpy as np
import pytest

from analysis.metrics import METRIC_DIRECTION
from analysis.stats import strongest_baseline
from experiments.backend_validation import BackendComparison


def _samples(means: dict[str, float], half_width: float) -> dict[str, list[float]]:
    return {p: [m - half_width, m, m + half_width] for p, m in means.items()}


def _comparison(metric, fallback, sumo, half_width) -> BackendComparison:
    return BackendComparison(
        metric=metric, fallback=fallback, sumo=sumo,
        fallback_samples=_samples(fallback, half_width),
        sumo_samples=_samples(sumo, half_width),
    )


# The cost numbers measured at rural d=20 (4 km, 3 seeds).
COST_FALLBACK = {"flooding": 2.3305, "slotted_1p": 0.5809,
                 "greedy_farthest": 0.5284, "dvcast": 0.5875}
COST_SUMO = {"flooding": 2.5846, "slotted_1p": 0.6513,
             "greedy_farthest": 0.6430, "dvcast": 0.6180}


def test_cost_metrics_are_lower_is_better():
    """Regression: tx_per_at_risk_informed was missing from METRIC_DIRECTION
    and defaulted to higher-is-better, ranking flooding best on cost."""
    for metric in ("tx_per_at_risk_informed", "airtime_ms",
                   "airtime_per_at_risk_informed_ms", "dissemination_cbr"):
        assert METRIC_DIRECTION.get(metric) == -1, metric


def test_strongest_cost_baseline_is_not_flooding():
    per_policy = {p: [v] for p, v in COST_FALLBACK.items()}
    best, _ = strongest_baseline(per_policy, "tx_per_at_risk_informed")
    assert best == "greedy_farthest"
    assert best != "flooding"


def test_flooding_ranks_last_on_cost():
    c = _comparison("tx_per_at_risk_informed", COST_FALLBACK, COST_SUMO, 0.05)
    assert c.ranking("fallback")[-1] == "flooding"
    assert c.ranking("sumo")[-1] == "flooding"


def test_outlier_that_did_not_move_cannot_make_a_reshuffle_look_real():
    """Regression: with the spread taken over ALL policies, flooding's ~1.8
    gap made a reshuffle among three schemes within 0.06 of each other read
    as 'CHANGED (real)'."""
    c = _comparison("tx_per_at_risk_informed", COST_FALLBACK, COST_SUMO, 0.05)
    assert not c.ordering_preserved
    assert "flooding" not in c.moved()
    assert c.verdict == "changed within noise"


def test_a_genuine_reversal_is_reported_as_real():
    fallback = {"a": 0.2, "b": 0.8, "c": 1.4}
    sumo = {"a": 1.4, "b": 0.8, "c": 0.2}
    c = _comparison("rwcr", fallback, sumo, 0.01)
    assert set(c.moved()) == {"a", "c"}
    assert c.verdict == "CHANGED (real)"


def test_ordered_in_one_backend_but_tied_in_the_other_is_not_a_flip():
    fallback = {"a": 0.50, "b": 0.90}          # clearly separated
    sumo = {"a": 0.71, "b": 0.70}              # swapped, but within noise
    c = _comparison("rwcr", fallback, sumo, 0.05)
    assert not c.ordering_preserved
    assert c.verdict == "changed within noise"


def test_preserved_ordering_is_reported_as_preserved():
    c = _comparison("rwcr", {"a": 0.5, "b": 0.9}, {"a": 0.4, "b": 0.8}, 0.01)
    assert c.ordering_preserved
    assert c.verdict == "PRESERVED"
    assert c.moved() == []
