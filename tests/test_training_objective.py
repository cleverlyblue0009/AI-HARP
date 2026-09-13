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
