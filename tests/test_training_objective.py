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
    assert obj.multiplier.value == pytest.approx(cfg["objective"]["lambda_init"])


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


def test_update_steps_lambda_once_from_a_batch_of_shortfalls(tmp_path):
    obj = build_training_objective(_cfg(), tmp_path)
    start = obj.multiplier.value
    obj.update([0.4, 0.2])
    assert obj.multiplier.value == pytest.approx(start + obj.objective.lambda_lr * 0.3)


def test_state_dict_round_trips_through_json(tmp_path):
    obj = build_training_objective(_cfg(), tmp_path)
    back = json.loads(json.dumps(obj.state_dict()))
    assert back["multiplier"]["value"] == pytest.approx(obj.multiplier.value)


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
    lam = obj.multiplier.value
    policy = AiHarpPolicy()                                 # no network: analytic fallback
    policy.reset(0, np.random.default_rng(0))
    _, info = run_episode(EpisodeSpec("rural_highway", 5.0, "clear", "fog_bank", 100),
                          policy, obj, cfgs)
    assert obj.multiplier.value == lam
    assert info["lambda"] == pytest.approx(lam)
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
    assert obj.multiplier.value == pytest.approx(cfg["objective"]["lambda_init"])


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
