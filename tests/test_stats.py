"""Phase 7 tests: paired significance testing and effect sizes."""

from __future__ import annotations

import numpy as np
import pytest

from analysis.stats import (
    MIN_PAIRS_FOR_WILCOXON,
    PairedTest,
    compare_against_reference,
    holm_bonferroni,
    paired_test,
    rank_biserial_correlation,
    strongest_baseline,
)


# ------------------------------------------------------------ effect size --
def test_rank_biserial_is_plus_one_when_candidate_wins_every_pair():
    assert rank_biserial_correlation(np.arange(1, 11, dtype=float)) == pytest.approx(1.0)


def test_rank_biserial_is_minus_one_when_candidate_loses_every_pair():
    assert rank_biserial_correlation(-np.arange(1, 11, dtype=float)) == pytest.approx(-1.0)


def test_rank_biserial_is_zero_for_a_symmetric_split():
    d = np.array([1.0, -1.0, 2.0, -2.0, 3.0, -3.0])
    assert rank_biserial_correlation(d) == pytest.approx(0.0)


def test_rank_biserial_ignores_exact_ties():
    assert rank_biserial_correlation(np.array([0.0, 0.0, 1.0, 2.0])) == pytest.approx(1.0)


def test_rank_biserial_is_bounded():
    rng = np.random.default_rng(0)
    for _ in range(200):
        assert -1.0 <= rank_biserial_correlation(rng.normal(size=12)) <= 1.0


def test_effect_size_does_not_grow_with_sample_size():
    """The reason effect sizes are reported at all: with a deterministic
    simulator, a trivial difference reaches significance as n grows, but the
    effect size does not move."""
    rng = np.random.default_rng(1)
    small = rng.normal(0.01, 1.0, 10)
    big = np.concatenate([rng.normal(0.01, 1.0, 1000)])
    r_small = rank_biserial_correlation(small)
    r_big = rank_biserial_correlation(big)
    assert abs(r_big) < 0.3 and abs(r_small) < 0.9


# ------------------------------------------------------------ paired test --
def test_consistent_improvement_is_significant():
    ref = np.linspace(0.5, 0.6, 12)
    cand = ref + 0.05
    t = paired_test(cand, ref, "rwcr")
    assert t.p_value < 0.05
    assert t.rank_biserial == pytest.approx(1.0)
    assert t.candidate_better


def test_direction_is_respected_for_lower_is_better_metrics():
    """A LOWER transmission count is better, so a negative difference means the
    candidate won."""
    ref = np.linspace(100, 120, 12)
    cand = ref - 20
    t = paired_test(cand, ref, "transmissions")
    assert t.median_difference < 0
    assert t.candidate_better


def test_identical_samples_are_not_significant():
    x = np.linspace(0, 1, 10)
    t = paired_test(x, x, "rwcr")
    assert t.p_value == 1.0
    assert "identical" in t.note


def test_too_few_pairs_reports_no_p_value():
    """Below 6 pairs a two-sided Wilcoxon cannot reach p < 0.05 at all, so
    reporting one would be misleading."""
    t = paired_test([1, 2, 3], [0, 1, 2], "rwcr")
    assert not np.isfinite(t.p_value)
    assert "too few" in t.note
    assert t.n_pairs < MIN_PAIRS_FOR_WILCOXON


def test_non_finite_pairs_are_dropped_and_counted():
    """A metric undefined in half the seeds must not look like a clean test."""
    cand = [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0]
    ref = [0.0, 1.0, 2.0, np.inf, 4.0, 5.0, 6.0, 7.0]
    t = paired_test(cand, ref, "rwcr")
    assert t.n_pairs == 6


def test_mismatched_lengths_are_rejected():
    with pytest.raises(ValueError):
        paired_test([1, 2, 3], [1, 2], "rwcr")


# ----------------------------------------------------------------- Holm ----
def test_holm_is_less_conservative_than_bonferroni():
    tests = [PairedTest(f"m{i}", 10, 0.1, 0.1, 0.5, p)
             for i, p in enumerate([0.001, 0.02, 0.03, 0.5])]
    holm_bonferroni(tests, alpha=0.05)
    # Plain Bonferroni would multiply every p by 4; Holm multiplies the
    # smallest by 4, the next by 3, and so on.
    assert tests[0].p_adjusted == pytest.approx(0.004)
    assert tests[1].p_adjusted == pytest.approx(0.06)


def test_holm_adjusted_p_values_are_monotone():
    rng = np.random.default_rng(3)
    tests = [PairedTest(f"m{i}", 10, 0.1, 0.1, 0.5, float(p))
             for i, p in enumerate(sorted(rng.uniform(0, 0.2, 8)))]
    holm_bonferroni(tests)
    adj = [t.p_adjusted for t in sorted(tests, key=lambda t: t.p_value)]
    assert adj == sorted(adj)


def test_holm_stops_at_the_first_failure():
    """Step-down: once one test fails, every later one is non-significant even
    if its own adjusted p-value would have passed.

    Here the third test adjusts to 0.045 x 1 = 0.045 < alpha, which a naive
    per-test rule would call significant. Holm must not, because the second
    test failed at 0.04 x 2 = 0.08.
    """
    tests = [PairedTest(f"m{i}", 10, 0.1, 0.1, 0.5, p)
             for i, p in enumerate([0.001, 0.04, 0.045])]
    holm_bonferroni(tests, alpha=0.05)
    by_p = sorted(tests, key=lambda t: t.p_value)
    assert by_p[0].significant            # 0.001 * 3 = 0.003
    assert not by_p[1].significant        # 0.04  * 2 = 0.08  -> fails, stops
    assert not by_p[2].significant        # would be 0.045 alone, but comes after
    # Monotonicity also lifts the last adjusted p to the running maximum.
    assert by_p[2].p_adjusted >= by_p[1].p_adjusted


def test_holm_handles_tests_without_p_values():
    tests = [PairedTest("a", 3, 0.1, 0.1, 0.5, float("nan")),
             PairedTest("b", 10, 0.1, 0.1, 0.5, 0.001)]
    holm_bonferroni(tests)
    assert not np.isfinite(tests[0].p_adjusted)
    assert tests[1].significant


def test_significance_markers():
    t = PairedTest("m", 10, 0.1, 0.1, 0.5, 0.0001, p_adjusted=0.0005)
    assert t.marker == "***"
    t.p_adjusted = 0.005
    assert t.marker == "**"
    t.p_adjusted = 0.04
    assert t.marker == "*"
    t.p_adjusted = 0.2
    assert t.marker == ""


# ------------------------------------------------------- strongest baseline --
def test_strongest_baseline_respects_metric_direction():
    per_policy = {"a": [0.9, 0.9], "b": [0.5, 0.5]}
    assert strongest_baseline(per_policy, "rwcr")[0] == "a"        # higher better
    assert strongest_baseline(per_policy, "transmissions")[0] == "b"  # lower better


def test_strongest_baseline_ignores_empty_and_nan_policies():
    per_policy = {"a": [np.nan, np.nan], "b": [0.5, 0.6], "c": []}
    assert strongest_baseline(per_policy, "rwcr")[0] == "b"


# ------------------------------------------------------------------ family --
def test_family_comparison_corrects_across_metrics():
    rng = np.random.default_rng(4)
    ref = {m: rng.uniform(0.4, 0.6, 12) for m in ("rwcr", "tir_median_s", "transmissions")}
    cand = {"rwcr": ref["rwcr"] + 0.2,
            "tir_median_s": ref["tir_median_s"] - 0.2,
            "transmissions": ref["transmissions"] + 0.0001}
    tests = compare_against_reference(cand, ref, list(ref))
    assert len(tests) == 3
    assert all(np.isfinite(t.p_adjusted) for t in tests)
    assert any(t.significant for t in tests)
