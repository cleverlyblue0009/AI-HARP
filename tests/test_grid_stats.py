"""analysis/grid_stats.py on a synthetic runs.csv grid."""

from __future__ import annotations

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

from analysis.grid_stats import fixed_best_baselines, grid_tests  # noqa: E402


def _grid():
    rng = np.random.default_rng(0)
    rows = []
    cells = [("rural_highway", 2.0, "clear", "fog_bank", "train", "train", "train"),
             ("rural_highway", 20.0, "clear", "fog_bank", "train", "train", "train"),
             ("urban_grid", 20.0, "dense_fog", "black_ice", "held_out", "held_out", "held_out")]
    for sc, d, w, h, ts, hs, ws in cells:
        for seed in range(10):
            noise = rng.normal(0, 0.01)
            base = {"scenario": sc, "density": d, "weather": w, "hazard_type": h, "seed": seed,
                    "topology_split": ts, "hazard_split": hs, "weather_split": ws}
            # A: best on training cells; B: best on the held-out cell only.
            qa, qb = (0.80, 0.70) if ts == "train" else (0.50, 0.90)
            rows.append({**base, "policy": "slotted_1p", "tau": np.nan, "rwcr": qa + noise,
                         "tx_per_at_risk_informed": 1.0})
            rows.append({**base, "policy": "dvcast", "tau": np.nan, "rwcr": qb + noise,
                         "tx_per_at_risk_informed": 1.2})
            # agent: clearly better RWCR everywhere, paired noise
            rows.append({**base, "policy": "ai_harp", "tau": "0.5",
                         "rwcr": max(qa, qb) + 0.05 + noise + 0.001 * seed,
                         "tx_per_at_risk_informed": 1.1})
    return pd.DataFrame(rows)


METRICS = ("rwcr", "tx_per_at_risk_informed")


def test_fixed_best_is_chosen_on_training_cells_only():
    fixed = fixed_best_baselines(_grid(), METRICS)
    assert fixed["rwcr"] == "slotted_1p"            # dvcast wins only on the held-out cell
    assert fixed["tx_per_at_risk_informed"] == "slotted_1p"


def test_oracle_best_is_per_cell_and_families_are_split():
    t = grid_tests(_grid(), METRICS)
    rw = t[(t["metric"] == "rwcr") & (t["comparison"] == "oracle_best")]
    held = rw[rw["topology_split"] == "held_out"]
    train = rw[rw["topology_split"] == "train"]
    assert set(held["reference_policy"]) == {"dvcast"}
    assert set(train["reference_policy"]) == {"slotted_1p"}
    # the fixed reference on the held-out cell is still the training-chosen one
    fx = t[(t["metric"] == "rwcr") & (t["comparison"] == "fixed_best") & (t["topology_split"] == "held_out")]
    assert set(fx["reference_policy"]) == {"slotted_1p"}


def test_a_clearly_better_agent_is_significant_with_effect_size():
    t = grid_tests(_grid(), METRICS)
    rw = t[t["metric"] == "rwcr"]
    assert (rw["n_pairs"] == 10).all()
    assert rw["significant"].all() and rw["agent_better"].all()
    assert (rw["rank_biserial"] == 1.0).all()
    assert (rw["p_holm"] >= rw["p_value"]).all()


def test_cost_direction_is_respected():
    t = grid_tests(_grid(), METRICS)
    cost = t[(t["metric"] == "tx_per_at_risk_informed") & (t["comparison"] == "fixed_best")]
    # agent 1.1 vs slotted_1p 1.0: worse (lower is better); identical in every pair -> no win
    assert not cost["agent_better"].any()


def test_holm_family_is_one_cell_not_the_whole_grid():
    """At 10 seeds the floor is p = 2^-9, so only a family of <= 25 can ever clear
    0.05: Holm must correct across the metrics of ONE cell (here 2), not across
    the 3 cells as well (which would give p x 6 and never be significant)."""
    rng = np.random.default_rng(1)
    rows = []
    for d in (2.0, 20.0, 80.0):
        for seed in range(10):
            base = {"scenario": "rural_highway", "density": d, "weather": "clear",
                    "hazard_type": "fog_bank", "seed": seed, "topology_split": "train",
                    "hazard_split": "train", "weather_split": "train"}
            n = rng.normal(0, 1e-4)
            rows.append({**base, "policy": "slotted_1p", "tau": np.nan,
                         "rwcr": 0.70 + n, "tx_per_at_risk_informed": 1.0})
            rows.append({**base, "policy": "ai_harp", "tau": "0.5",
                         "rwcr": 0.80 + n, "tx_per_at_risk_informed": 1.0})
    t = grid_tests(pd.DataFrame(rows), METRICS)
    rw = t[(t["metric"] == "rwcr") & (t["comparison"] == "oracle_best")]
    assert (rw["n_pairs"] == 10).all() and rw["agent_better"].all()
    assert rw["p_value"].min() == pytest.approx(2 ** -9, rel=1e-6)   # the floor at 10 seeds
    # 2 metrics in this fixture's cell -> p x 2, not p x 6 (metrics x cells)
    assert rw["p_holm"].min() == pytest.approx(rw["p_value"].min() * 2, rel=1e-6)
    assert rw["significant"].all()
    # the tied cost metric is its own family and stays non-significant
    cost = t[(t["metric"] == "tx_per_at_risk_informed") & (t["comparison"] == "oracle_best")]
    assert not cost["significant"].any()


def test_no_agent_rows_gives_empty_result():
    g = _grid()
    assert grid_tests(g[g["policy"] != "ai_harp"], METRICS).empty
