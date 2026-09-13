"""Phase 7 tests: evaluating a trained agent through the paired baseline path."""

from __future__ import annotations

import numpy as np
import pytest

from agents.base import ActionType
from tests.test_policies import make_ctx

pytest.importorskip("torch")
pytest.importorskip("torch_geometric")


class _FakeNet:
    """Deterministic stand-in: slightly prefers broadcast_now."""

    def __init__(self) -> None:
        p = np.full(9, 0.105)
        p[1] = 0.16                       # broadcast_now
        self.probs = p / p.sum()

    def act(self, data, deterministic: bool = False):
        return {"action": int(np.argmax(self.probs)), "probs": self.probs.copy(),
                "relay_order": [1, 2, 3], "log_prob": 0.0, "value": 0.0, "entropy": 2.1}


def _policy(bias: float):
    from agents.ai_harp import AiHarpPolicy
    from agents.confidence import ConfidenceGate

    return AiHarpPolicy(network=_FakeNet(), gate=ConfidenceGate(tau=0.0, enabled=False),
                        deterministic=True, suppression_bias=bias)


def test_zero_bias_executes_the_networks_choice():
    assert _policy(0.0).decide(make_ctx()).kind is ActionType.BROADCAST


def test_positive_bias_suppresses():
    """The bias is the agent's knob for tracing an operating curve: positive
    trades coverage for fewer transmissions."""
    assert _policy(2.0).decide(make_ctx()).kind is ActionType.SUPPRESS


def test_negative_bias_never_suppresses_when_the_network_was_close():
    a = _policy(-2.0).decide(make_ctx())
    assert a.kind in (ActionType.BROADCAST, ActionType.RELAY)


def test_bias_is_recorded_in_policy_params():
    """It enters the config hash, so each operating point is a distinct run."""
    assert _policy(1.5).params["suppression_bias"] == 1.5


def _save_checkpoint(tmp_path, embed_stats: bool = True):
    import torch

    from agents.gat_drl import build_network
    from agents.graph import FeatureNormaliser, build_decision_graph
    from common.config import load_yaml

    cfg = load_yaml("agent.yaml")
    net = build_network(cfg)
    norm = FeatureNormaliser.fit(
        [build_decision_graph(make_ctx(relevance=float(r))) for r in np.linspace(0, 1, 6)],
        provenance={"mode": "full", "scenarios": cfg["training"]["train_scenarios"]},
    )
    stats = norm.save(tmp_path / "stats.json")
    import json

    state = {"update": 7, "model": net.state_dict(), "config": cfg}
    if embed_stats:
        state["normaliser_stats"] = json.loads(stats.read_text())
    path = tmp_path / "ckpt.pt"
    torch.save(state, path)
    return path, norm


def test_checkpoint_builds_through_the_registry(tmp_path):
    """A trained agent travels through RunSpec as a checkpoint path, exactly
    like a baseline's knob -- no special-cased comparison path."""
    from agents.registry import build_policy

    path, _ = _save_checkpoint(tmp_path)
    pol = build_policy("ai_harp", checkpoint=str(path), tau=0.3, suppression_bias=0.5)
    assert pol.name == "ai_harp"
    assert pol.network is not None
    assert pol.gate.enabled and pol.gate.tau == pytest.approx(0.3)
    assert pol.decide(make_ctx()).kind in set(ActionType)


def test_checkpoint_is_evaluated_with_its_own_embedded_statistics(tmp_path):
    from agents.ai_harp import AiHarpPolicy

    path, norm = _save_checkpoint(tmp_path)
    pol = AiHarpPolicy.from_checkpoint(path)
    assert np.allclose(pol.normaliser.node_mean, norm.node_mean)
    assert np.allclose(pol.normaliser.node_std, norm.node_std)


def test_evaluation_gate_is_on_by_default(tmp_path):
    """Training disables the gate; evaluation must not inherit that."""
    from agents.ai_harp import AiHarpPolicy
    from common.config import load_yaml

    path, _ = _save_checkpoint(tmp_path)
    pol = AiHarpPolicy.from_checkpoint(path)
    assert pol.gate.enabled
    assert pol.gate.tau == pytest.approx(load_yaml("agent.yaml")["confidence_gate"]["tau"])


def test_checkpoint_sha_changes_with_content(tmp_path):
    """A retrained checkpoint at the same path must not share a config hash."""
    from experiments.evaluate_agent import checkpoint_sha

    a, b = tmp_path / "a.pt", tmp_path / "b.pt"
    a.write_bytes(b"one")
    b.write_bytes(b"two")
    assert checkpoint_sha(a) != checkpoint_sha(b)


def test_run_single_reports_policy_statistics():
    """The gate's fallback rate is a first-class metric; it must reach the row."""
    from experiments.run_sim import RunSpec, run_single

    m, _ = run_single(RunSpec(density_veh_km_lane=5, seed=0, policy="ai_harp",
                              duration_s=20.0, corridor_length_m=2000.0))
    assert "gate_fallback_rate" in m
    assert "relay_action_frac" in m
