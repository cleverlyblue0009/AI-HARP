"""Staged curriculum, constraint early stopping, worker count and exact resume."""

from __future__ import annotations

import copy
import json
import os

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from agents.constrained_reward import CoverageTargets, training_cell_keys  # noqa: E402
from agents.train import (  # noqa: E402
    ConstraintEarlyStop, resolve_workers, sample_episode_specs, stage_densities,
    train, training_plan,
)
from common.config import load_yaml  # noqa: E402


def _cfg():
    return copy.deepcopy(load_yaml("agent.yaml"))


# ----------------------------------------------------------- curriculum -----
def test_pretrain_samples_only_dense_densities():
    cfg = _cfg()
    d, p = stage_densities(cfg, "pretrain")
    assert d == [20.0, 40.0, 80.0] and np.allclose(p, 1 / 3)


def test_finetune_weights_sparse_three_to_one_over_dense():
    cfg = _cfg()
    d, p = stage_densities(cfg, "finetune")
    sparse = set(float(x) for x in cfg["training"]["curriculum"]["finetune"]["sparse_densities"])
    assert p.sum() == pytest.approx(1.0)
    assert sum(pi for di, pi in zip(d, p) if di in sparse) == pytest.approx(0.75)
    rng = np.random.default_rng(0)
    specs = sample_episode_specs(cfg, 0.0, rng, 8000, stage="finetune")
    assert np.mean([s.density in sparse for s in specs]) == pytest.approx(0.75, abs=0.02)


def test_staged_sampler_only_draws_cells_with_coverage_targets():
    cfg = _cfg()
    keys = set(training_cell_keys(cfg))
    rng = np.random.default_rng(1)
    for stage in ("pretrain", "finetune"):
        for s in sample_episode_specs(cfg, 0.0, rng, 400, stage=stage):
            assert CoverageTargets.key(s.scenario, s.density, s.weather, s.hazard_type) in keys


def test_the_committed_targets_cover_every_staged_cell():
    from pathlib import Path

    from common.config import PROJECT_ROOT

    cfg = _cfg()
    table = json.loads((PROJECT_ROOT / cfg["objective"]["targets_path"]).read_text())["cells"]
    assert set(training_cell_keys(cfg)) <= set(table)


def test_training_plan_default_is_pretrain_then_capped_finetune():
    cfg = _cfg()
    assert training_plan(cfg, None, "all", smoke=False) == [("pretrain", 200), ("finetune", 800)]
    assert training_plan(cfg, 250, "all", smoke=False) == [("pretrain", 200), ("finetune", 50)]
    assert training_plan(cfg, None, "finetune", smoke=False) == [("finetune", 800)]


def test_anneal_mode_is_still_one_stage():
    cfg = _cfg()
    cfg["training"]["curriculum"]["mode"] = "anneal"
    assert training_plan(cfg, 40, "all", smoke=False) == [("anneal", 40)]


# ----------------------------------------------------------- early stop -----
def test_early_stop_needs_a_full_window_of_met_updates():
    es = ConstraintEarlyStop(window=3, required=frozenset({"a", "b"}))
    assert not es.update({"a": -0.1, "b": -0.2})
    assert not es.update({"a": -0.1})
    assert es.update({"b": 0.0})


def test_early_stop_resets_on_any_shortfall():
    es = ConstraintEarlyStop(window=2, required=frozenset({"a"}))
    es.update({"a": -0.1})
    assert not es.update({"a": 0.01, "b": -1.0})
    assert es.run == 0
    assert not es.update({"a": -0.1})
    assert es.update({"a": -0.1})


def test_early_stop_requires_every_group_to_have_been_seen():
    """A rarely drawn sparse group must not pass by simply not being sampled."""
    es = ConstraintEarlyStop(window=2, required=frozenset({"dense", "sparse"}))
    es.update({"dense": -0.1})
    assert not es.update({"dense": -0.1})
    assert es.update({"sparse": -0.1})


def test_early_stop_state_round_trips():
    es = ConstraintEarlyStop(window=5, required=frozenset({"a"}))
    es.update({"a": -1.0})
    es2 = ConstraintEarlyStop(window=5, required=frozenset({"a"}))
    es2.load_state_dict(json.loads(json.dumps(es.state_dict())))
    assert (es2.run, es2.seen) == (es.run, es.seen)


# -------------------------------------------------------------- workers -----
def test_auto_workers_is_cpu_count_minus_one():
    assert resolve_workers("auto") == max(1, (os.cpu_count() or 2) - 1)
    assert resolve_workers(3) == 3 and resolve_workers("5") == 5


# --------------------------------------------------------------- resume -----
def _tiny_cfg():
    """Smoke-sized staged run on cheap densities."""
    cfg = _cfg()
    c = cfg["training"]["curriculum"]
    # Dense enough that two episodes always yield >= 4 transitions; density 1
    # can yield fewer, and an update with too few skips its PPO step.
    c["pretrain"]["densities"] = [5]
    c["finetune"]["sparse_densities"] = [3]
    c["finetune"]["dense_densities"] = [5]
    cfg["training"]["rollout_workers"] = 1
    return cfg


def _cfgs():
    return {"phy": load_yaml("phy.yaml"), "hazard": load_yaml("hazard.yaml"),
            "experiment": load_yaml("experiment.yaml")}


def _strip(history):
    return [{k: v for k, v in r.items() if k != "elapsed_s"} for r in history]


def test_resume_reproduces_an_uninterrupted_run_exactly(tmp_path):
    import torch

    cfg, cfgs = _tiny_cfg(), _cfgs()
    full = train(copy.deepcopy(cfg), cfgs, 3, tmp_path / "full", smoke=True)
    train(copy.deepcopy(cfg), cfgs, 2, tmp_path / "split", smoke=True)
    resumed = train(copy.deepcopy(cfg), cfgs, 3, tmp_path / "split", smoke=True,
                    resume=tmp_path / "split" / "ckpt_latest.pt")
    assert _strip(resumed["history"]) == _strip(full["history"])
    a = torch.load(tmp_path / "full" / "ckpt_final.pt", weights_only=False)["model"]
    b = torch.load(tmp_path / "split" / "ckpt_final.pt", weights_only=False)["model"]
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert [r["stage"] for r in full["history"]] == ["pretrain", "pretrain", "finetune"]
    summary = json.loads((tmp_path / "split" / "run_summary.json").read_text())
    assert summary["total_updates"] == 3 and summary["resumed_from"]


def test_stage_boundary_checkpoint_seeds_a_finetune_run(tmp_path):
    cfg, cfgs = _tiny_cfg(), _cfgs()
    train(copy.deepcopy(cfg), cfgs, None, tmp_path / "pre", smoke=True, stage="pretrain")
    assert (tmp_path / "pre" / "ckpt_pretrain.pt").exists()
    out = train(copy.deepcopy(cfg), cfgs, None, tmp_path / "ft", smoke=True, stage="finetune",
                init_from=tmp_path / "pre" / "ckpt_pretrain.pt")
    assert [r["stage"] for r in out["history"]] == ["finetune"]
    assert out["history"][0]["update"] == 3          # global count continues after 2 pretrain
