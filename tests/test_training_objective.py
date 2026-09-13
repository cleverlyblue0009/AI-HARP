"""Wiring the constrained objective into training."""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from agents.constrained_reward import build_training_objective
from common.config import load_yaml


def _cfg():
    return copy.deepcopy(load_yaml("agent.yaml"))


def test_builder_starts_lambda_at_its_configured_value(tmp_path):
    cfg = _cfg()
    obj = build_training_objective(cfg, tmp_path)
    assert obj.lambda_for("rural_highway", 20, "clear", "fog_bank") == pytest.approx(
        cfg["objective"]["lambda_init"])


def test_builder_refuses_the_broken_weighted_sum_reward(tmp_path):
    cfg = _cfg()
    cfg["objective"]["kind"] = "weighted_sum"
    with pytest.raises(ValueError, match="only 'constrained'"):
        build_training_objective(cfg, tmp_path)


def test_missing_targets_file_uses_the_fallback(tmp_path):
    cfg = _cfg()
    obj = build_training_objective(cfg, tmp_path)          # no targets under tmp_path
    assert obj.target_for("rural_highway", 20, "clear", "fog_bank") == pytest.approx(
        cfg["objective"]["fallback_target"])


def test_measured_targets_are_used_when_present(tmp_path):
    cfg = _cfg()
    rel = cfg["objective"]["targets_path"]
    path = tmp_path / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"cells": {"rural_highway|20|clear|fog_bank": {"ceiling": 0.9}}}))
    obj = build_training_objective(cfg, tmp_path)
    assert obj.target_for("rural_highway", 20.0, "clear", "fog_bank") == pytest.approx(
        cfg["objective"]["target_fraction"] * 0.9)


def test_update_steps_each_group_once_on_its_pooled_shortfall(tmp_path):
    obj = build_training_objective(_cfg(), tmp_path)
    g = obj.group_for("rural_highway", 20, "clear", "fog_bank")
    start = obj.multipliers.value(g)
    # Weather and hazard share the (scenario, density) group.
    g2 = obj.group_for("rural_highway", 20, "heavy_rain", "crash")
    assert g2 == g
    obj.update([(g, 0.5, 0.8), (g2, 0.7, 0.8)])
    expected = start + obj.objective.lambda_lr * (1.6 - 1.2) / 1.6
    assert obj.multipliers.value(g) == pytest.approx(expected)


def test_state_dict_round_trips_through_json(tmp_path):
    obj = build_training_objective(_cfg(), tmp_path)
    g = obj.group_for("urban_nlos", 5, "clear", "fog_bank")
    obj.update([(g, 0.1, 0.3)])
    back = json.loads(json.dumps(obj.state_dict()))
    assert back["multipliers"]["values"][g] == pytest.approx(obj.multipliers.value(g))
    assert back["objective"]["lambda_group_by"] == ["scenario", "density"]


# ------------------------------------------- pooled, per-group dual ascent ---
def test_pooled_shortfall_is_not_masked_by_an_overshooting_low_target_episode():
    """run4 update 35: pooled coverage short, yet the mean of per-episode ratios
    was negative because one tiny-target episode overshot."""
    from agents.constrained_reward import pooled_shortfall

    covs, tgts = [0.30, 0.55, 0.60], [0.12, 0.80, 0.85]
    ratio_mean = np.mean([(t - c) / t for c, t in zip(covs, tgts)])
    assert ratio_mean < 0                                   # the old step lowered lambda
    assert pooled_shortfall(covs, tgts) == pytest.approx((1.77 - 1.45) / 1.77)
    assert pooled_shortfall(covs, tgts) > 0


def test_pooled_shortfall_ignores_non_finite_episodes():
    from agents.constrained_reward import pooled_shortfall

    assert pooled_shortfall([np.nan, 0.4], [0.5, 0.5]) == pytest.approx(0.2)
    assert np.isnan(pooled_shortfall([np.nan], [0.5]))


def test_dense_surplus_does_not_lower_the_sparse_price(tmp_path):
    obj = build_training_objective(_cfg(), tmp_path)
    sparse = obj.group_for("rural_highway", 2, "clear", "fog_bank")
    dense = obj.group_for("rural_highway", 80, "clear", "fog_bank")
    start = obj.multipliers.value(sparse)
    obj.update([(sparse, 0.40, 0.60), (dense, 0.99, 0.87), (dense, 0.99, 0.87)])
    assert obj.multipliers.value(sparse) > start
    assert obj.multipliers.value(dense) < start


def test_groups_absent_from_a_batch_keep_their_price(tmp_path):
    obj = build_training_objective(_cfg(), tmp_path)
    a = obj.group_for("urban_nlos", 20, "clear", "fog_bank")
    b = obj.group_for("urban_nlos", 40, "clear", "fog_bank")
    obj.update([(b, 0.2, 0.3)])
    before = obj.multipliers.value(b)
    obj.update([(a, 0.1, 0.3)])
    assert obj.multipliers.value(b) == before


def test_empty_group_by_is_one_global_pooled_multiplier(tmp_path):
    cfg = _cfg()
    cfg["objective"]["lambda_group_by"] = []
    obj = build_training_objective(cfg, tmp_path)
    assert (obj.group_for("rural_highway", 2, "clear", "fog_bank")
            == obj.group_for("urban_nlos", 80, "heavy_rain", "crash") == "all")


def test_unknown_group_field_is_refused(tmp_path):
    cfg = _cfg()
    cfg["objective"]["lambda_group_by"] = ["scenario", "seed"]
    with pytest.raises(ValueError, match="unknown fields"):
        build_training_objective(cfg, tmp_path)


def test_multiplier_is_capped_and_saturation_is_reported(tmp_path):
    cfg = _cfg()
    cfg["objective"].update(lambda_lr=1000.0, lambda_max=5.0)
    obj = build_training_objective(cfg, tmp_path)
    g = obj.group_for("urban_nlos", 1, "clear", "fog_bank")
    obj.update([(g, 0.0, 0.2)])
    assert obj.multipliers.value(g) == pytest.approx(5.0)
    assert obj.multipliers.saturated_groups == [g]


def test_run_episode_scores_at_the_current_price_without_stepping_it(tmp_path):
    """The multiplier must not move inside an episode, or transitions in one
    PPO batch would be scored at different prices."""
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from agents.ai_harp import AiHarpPolicy
    from agents.train import EpisodeSpec, run_episode

    cfgs = {"phy": load_yaml("phy.yaml"), "hazard": load_yaml("hazard.yaml"),
            "experiment": load_yaml("experiment.yaml")}
    obj = build_training_objective(_cfg(), tmp_path)
    group = obj.group_for("rural_highway", 5.0, "clear", "fog_bank")
    lam = obj.multipliers.value(group)
    policy = AiHarpPolicy()                                 # no network: analytic fallback
    policy.reset(0, np.random.default_rng(0))
    _, info = run_episode(EpisodeSpec("rural_highway", 5.0, "clear", "fog_bank", 100),
                          policy, obj, cfgs)
    assert obj.multipliers.value(group) == lam
    assert info["lambda"] == pytest.approx(lam)
    assert info["lambda_group"] == group
    assert 0.0 <= info["obj_coverage"] <= 1.0
    assert info["obj_target"] == pytest.approx(obj.objective.fallback_target)
    assert np.isfinite(info["obj_shortfall"])


# ------------------------------------------------ no training on fallbacks ---
def _write_table(tmp_path, cfg, keys):
    path = tmp_path / cfg["objective"]["targets_path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"cells": {k: {"ceiling": 0.9} for k in keys}}))


def test_full_runs_refuse_to_start_without_measured_targets(tmp_path):
    """A smoke run on fallback targets priced an urban-like episode (coverage
    0.286) against an infeasible 0.80 and lambda rose every update."""
    with pytest.raises(ValueError, match="no measured coverage"):
        build_training_objective(_cfg(), tmp_path, require_targets=True)


def test_partial_target_tables_are_refused(tmp_path):
    from agents.constrained_reward import training_cell_keys

    cfg = _cfg()
    keys = training_cell_keys(cfg)
    _write_table(tmp_path, cfg, keys[:-1])
    with pytest.raises(ValueError, match=f"1 of {len(keys)}"):
        build_training_objective(cfg, tmp_path, require_targets=True)


def test_complete_target_table_is_accepted(tmp_path):
    from agents.constrained_reward import training_cell_keys

    cfg = _cfg()
    _write_table(tmp_path, cfg, training_cell_keys(cfg))
    obj = build_training_objective(cfg, tmp_path, require_targets=True)
    assert obj.multipliers.init == pytest.approx(cfg["objective"]["lambda_init"])


def test_smoke_runs_may_use_fallback_targets(tmp_path):
    build_training_objective(_cfg(), tmp_path, require_targets=False)


def test_training_cells_match_what_the_target_builder_measures():
    from agents.constrained_reward import CoverageTargets, training_cell_keys
    from experiments.coverage_targets import training_cells

    cfg = _cfg()
    assert training_cell_keys(cfg) == [CoverageTargets.key(*c) for c in training_cells(cfg)]


def test_training_cells_cover_everything_the_sampler_can_draw():
    """Otherwise a sampled cell silently gets the fallback target."""
    pytest.importorskip("torch")
    from agents.constrained_reward import CoverageTargets, training_cell_keys
    from agents.train import sample_episode_specs

    cfg = _cfg()
    keys = set(training_cell_keys(cfg))
    rng = np.random.default_rng(0)
    for progress in (0.1, 0.4, 0.6, 0.9, 1.0):
        for s in sample_episode_specs(cfg, progress, rng, 60):
            assert CoverageTargets.key(s.scenario, s.density, s.weather,
                                       s.hazard_type) in keys
