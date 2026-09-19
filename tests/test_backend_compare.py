"""Backend comparison: ordering agreement, and what counts as a real flip.

The question this module exists to answer is whether a conclusion changes when
the mobility backend changes. The trap it has to avoid is counting a swap
between two near-tied policies as a changed conclusion: RWCR saturates, so the
top two can trade places on a difference the paper would never report.
"""

from __future__ import annotations

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

from analysis.backend_compare import compare, kendall_tau, summarise  # noqa: E402


# ================================================================== tau ======
def test_identical_ordering_is_plus_one():
    assert kendall_tau([1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]) == pytest.approx(1.0)


def test_reversed_ordering_is_minus_one():
    assert kendall_tau([1.0, 2.0, 3.0, 4.0], [40.0, 30.0, 20.0, 10.0]) == pytest.approx(-1.0)


def test_tau_ignores_non_finite_pairs():
    assert kendall_tau([1.0, 2.0, np.nan], [1.0, 2.0, 99.0]) == pytest.approx(1.0)


def test_tau_is_undefined_below_two_points():
    assert np.isnan(kendall_tau([1.0], [2.0]))


# ============================================================== compare ======
def _frame(values: dict[str, float], **over) -> pd.DataFrame:
    """One cell, one split, ten seeds, a mean per policy."""
    rows = []
    for policy, value in values.items():
        for seed in range(10):
            rows.append({
                "scenario": "rural_highway", "density_veh_km_lane": 2.0,
                "hazard_split": "train", "weather_split": "train",
                "policy": policy, "seed": seed, "metrics_version": 3,
                "tx_per_at_risk_informed": value,
                "rwcr": over.get("rwcr", {}).get(policy, 0.9),
                "tir_median_s": 0.3,
                "actionable_deadline_miss_rate": 0.1,
            })
    return pd.DataFrame(rows)


def _write(tmp_path, fallback: pd.DataFrame, sumo: pd.DataFrame):
    f, s = tmp_path / "fb.csv", tmp_path / "su.csv"
    fallback.to_csv(f, index=False)
    sumo.to_csv(s, index=False)
    return f, s


def test_detects_a_changed_winner_and_prices_it(tmp_path):
    """dvcast is cheapest on the fallback, counter_based on SUMO, by a wide margin."""
    f, s = _write(tmp_path,
                  _frame({"dvcast": 1.0, "counter_based": 2.0, "flooding": 3.0}),
                  _frame({"dvcast": 2.0, "counter_based": 1.0, "flooding": 3.0}))
    df, _ = compare(f, s)
    row = df[df["metric"] == "tx_per_at_risk_informed"].iloc[0]
    assert row["fallback_best"] == "dvcast" and row["sumo_best"] == "counter_based"
    assert row["best_changed"]
    # The new winner is 1.0 better on a 2.0 baseline: a real change, not a tie.
    assert row["flip_gain"] == pytest.approx(1.0)
    assert row["flip_gain_rel"] == pytest.approx(0.5)


def test_a_near_tie_swap_is_not_reported_as_a_changed_conclusion(tmp_path):
    """The swap is real but worth 0.1%: noise being relabelled as a result."""
    f, s = _write(tmp_path,
                  _frame({"dvcast": 1.000, "counter_based": 1.001}),
                  _frame({"dvcast": 1.001, "counter_based": 1.000}))
    df, text = compare(f, s)
    row = df[df["metric"] == "tx_per_at_risk_informed"].iloc[0]
    assert row["best_changed"], "the winner did change"
    assert row["flip_gain_rel"] < 0.05
    assert "0/1 material" in text


def test_agreeing_backends_report_no_flip(tmp_path):
    f, s = _write(tmp_path,
                  _frame({"dvcast": 1.0, "counter_based": 2.0}),
                  _frame({"dvcast": 1.5, "counter_based": 3.0}))
    df, _ = compare(f, s)
    row = df[df["metric"] == "tx_per_at_risk_informed"].iloc[0]
    assert not row["best_changed"]
    assert row["kendall_tau"] == pytest.approx(1.0)


def test_only_policies_present_in_both_are_compared(tmp_path):
    """The SUMO sweep carries baselines only, so the agent must not appear.

    Were it included, it would be scored against its own absence on one side
    and would win every cost comparison by default.
    """
    fallback = _frame({"dvcast": 1.0, "counter_based": 2.0, "ai_harp": 0.1})
    sumo = _frame({"dvcast": 1.0, "counter_based": 2.0})
    f, s = _write(tmp_path, fallback, sumo)
    df, _ = compare(f, s)
    assert (df["n_policies"] == 2).all(), "only the two shared baselines"
    assert "ai_harp" not in set(df["fallback_best"]) | set(df["sumo_best"])


def test_a_single_shared_policy_yields_no_comparison(tmp_path):
    """One policy has no ordering, so there is nothing to agree or disagree about."""
    f, s = _write(tmp_path, _frame({"dvcast": 1.0}), _frame({"dvcast": 2.0}))
    df, _ = compare(f, s)
    assert df.empty


def test_splits_are_reported_separately(tmp_path):
    """Pooling a held-out split into the training one is what this must not do."""
    fallback = pd.concat([_frame({"dvcast": 1.0, "counter_based": 2.0}),
                          _frame({"dvcast": 1.0, "counter_based": 2.0})
                          .assign(hazard_split="held_out")])
    sumo = pd.concat([_frame({"dvcast": 1.0, "counter_based": 2.0}),
                      _frame({"dvcast": 3.0, "counter_based": 1.0})
                      .assign(hazard_split="held_out")])
    f, s = _write(tmp_path, fallback, sumo)
    df, text = compare(f, s)
    assert set(df["hazard_split"]) == {"train", "held_out"}
    train = df[(df["hazard_split"] == "train")
               & (df["metric"] == "tx_per_at_risk_informed")].iloc[0]
    held = df[(df["hazard_split"] == "held_out")
              & (df["metric"] == "tx_per_at_risk_informed")].iloc[0]
    assert not train["best_changed"] and held["best_changed"]
    assert "hazard_split=held_out" in text and "hazard_split=train" in text


def test_summary_of_an_empty_comparison_says_so():
    assert "no shared" in summarise(pd.DataFrame())
