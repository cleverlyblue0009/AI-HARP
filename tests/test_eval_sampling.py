"""Sampled-policy evaluation must be reproducible and order-independent.

User decision after run8: the headline evaluation scores the stochastic
policy the constraint trained (argmax reported alongside). On rural d=80
argmax gave oracle RWCR 0.642 vs 0.814 sampled at the same cost.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

import torch  # noqa: E402

from agents.ai_harp import AiHarpPolicy, _seed_without_advancing  # noqa: E402
from agents.gat_drl import build_network  # noqa: E402
from common.config import load_yaml  # noqa: E402
from tests.test_policies import make_ctx  # noqa: E402


@pytest.fixture(scope="module")
def net():
    torch.manual_seed(0)
    return build_network(load_yaml("agent.yaml"))


def _policy(net, deterministic=False, record=False):
    return AiHarpPolicy(network=net, deterministic=deterministic, record=record)


def _actions(pol, ctxs, seed, batch):
    pol.reset(0, np.random.default_rng(seed))
    acts = pol.decide_batch(ctxs) if batch else [pol.decide(c) for c in ctxs]
    return [(a.kind, a.delay_steps, a.relay_indices) for a in acts]


CTXS = [make_ctx(index=k, sender_distance_m=40.0 + 25 * k) for k in range(24)]


def test_seed_derivation_does_not_advance_the_rng():
    a, b = np.random.default_rng(3), np.random.default_rng(3)
    _seed_without_advancing(a)
    assert a.random() == b.random()


def test_sampled_evaluation_depends_only_on_the_run_seed(net):
    pol = _policy(net)
    first = _actions(pol, CTXS, 11, batch=False)
    torch.manual_seed(999)                     # disturb the global generator
    torch.rand(1000)
    _actions(pol, CTXS, 12, batch=False)       # another run in between
    again = _actions(pol, CTXS, 11, batch=False)
    assert first == again


def test_sampled_evaluation_is_the_same_batched_and_sequential(net):
    assert _actions(_policy(net), CTXS, 5, batch=True) == _actions(_policy(net), CTXS, 5, batch=False)


def test_sampled_differs_from_argmax(net):
    sampled = [_actions(_policy(net), CTXS, s, batch=True) for s in range(4)]
    argmax = _actions(_policy(net, deterministic=True), CTXS, 0, batch=True)
    assert any(s != argmax for s in sampled)


def test_training_path_keeps_the_global_generator(net):
    """Recording (training) must not switch generators: run8's scheme stays."""
    pol = _policy(net, record=True)
    pol.reset(0, np.random.default_rng(0))
    assert pol._eval_gen is None
    det = _policy(net, deterministic=True)
    det.reset(0, np.random.default_rng(0))
    assert det._eval_gen is None
