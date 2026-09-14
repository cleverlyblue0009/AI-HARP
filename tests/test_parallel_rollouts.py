"""Parallel rollout collection must reproduce serial collection exactly."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

import torch  # noqa: E402

from agents.constrained_reward import build_training_objective  # noqa: E402
from agents.gat_drl import build_network  # noqa: E402
from agents.train import (  # noqa: E402
    EpisodeSpec, _training_policy, collect_rollouts, make_rollout_pool,
)
from common.config import load_yaml  # noqa: E402

SPECS = [EpisodeSpec("rural_highway", 3.0, "clear", "fog_bank", 100),
         EpisodeSpec("rural_highway", 5.0, "clear", "crash", 101)]
SEEDS = [11, 22]


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    cfg = load_yaml("agent.yaml")
    cfgs = {"phy": load_yaml("phy.yaml"), "hazard": load_yaml("hazard.yaml"),
            "experiment": load_yaml("experiment.yaml")}
    torch.manual_seed(0)
    net = build_network(cfg)
    objective = build_training_objective(cfg, tmp_path_factory.mktemp("targets"))
    return cfg, cfgs, net, objective


def _signature(results):
    out = []
    for transitions, info in results:
        out.append((
            [t.action for t in transitions],
            [round(t.log_prob, 6) for t in transitions],
            [round(t.reward, 6) for t in transitions],
            [t.action_mask for t in transitions],
            round(info["obj_coverage"], 9), info["transmissions"], info["lambda_group"],
        ))
    return out


def test_serial_collection_is_reproducible_under_the_same_episode_seeds(setup):
    cfg, cfgs, net, objective = setup
    policy = _training_policy(net, cfg, None)
    a = collect_rollouts(SPECS, SEEDS, net, policy, objective, cfgs)
    b = collect_rollouts(SPECS, SEEDS, net, policy, objective, cfgs)
    assert _signature(a) == _signature(b)
    assert sum(len(t) for t, _ in a) > 0


def test_episode_seeding_leaves_the_callers_torch_stream_untouched(setup):
    """PPO's minibatch shuffling draws from torch's global RNG after rollouts."""
    cfg, cfgs, net, objective = setup
    policy = _training_policy(net, cfg, None)
    torch.manual_seed(123)
    expected = torch.rand(3)
    torch.manual_seed(123)
    collect_rollouts(SPECS[:1], SEEDS[:1], net, policy, objective, cfgs)
    assert torch.equal(torch.rand(3), expected)


def test_worker_pool_reproduces_serial_collection_exactly(setup):
    cfg, cfgs, net, objective = setup
    serial = collect_rollouts(SPECS, SEEDS, net, _training_policy(net, cfg, None), objective, cfgs)
    pool = make_rollout_pool(cfg, cfgs, None, workers=2)
    try:
        parallel = collect_rollouts(SPECS, SEEDS, net, None, objective, cfgs, pool)
    finally:
        pool.shutdown(wait=True)
    assert _signature(parallel) == _signature(serial)


def test_serial_collection_restores_the_callers_thread_count(setup):
    """Rollouts run single-threaded for bit-identical log-probs; the PPO step
    that follows must get its threads back."""
    cfg, cfgs, net, objective = setup
    before = torch.get_num_threads()
    torch.set_num_threads(3)
    try:
        collect_rollouts(SPECS[:1], SEEDS[:1], net, _training_policy(net, cfg, None),
                         objective, cfgs)
        assert torch.get_num_threads() == 3
    finally:
        torch.set_num_threads(before)


def test_one_worker_means_serial():
    assert make_rollout_pool({}, {}, None, workers=1) is None


def test_different_episode_seeds_change_the_sampled_actions(setup):
    cfg, cfgs, net, objective = setup
    policy = _training_policy(net, cfg, None)
    a = collect_rollouts(SPECS[:1], [1], net, policy, objective, cfgs)
    b = collect_rollouts(SPECS[:1], [2], net, policy, objective, cfgs)
    assert [t.action for t in a[0][0]] != [t.action for t in b[0][0]]
