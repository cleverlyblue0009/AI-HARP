"""Phase 5 tests: the learned agent's action semantics.

Pinned because each was a measured failure. On rural d=40 an untrained agent
(the policy PPO starts from) had 90% of its suppress decisions followed by a
re-ask on a later duplicate and a transmission, and its defer never cancelled.
PPO saw suppress as 8 points worse than transmitting and eliminated it.
"""

from __future__ import annotations

import numpy as np
import pytest

from agents.ai_harp import ACTION_NAMES, AiHarpPolicy, action_to_engine
from agents.base import ActionType
from agents.confidence import ConfidenceGate
from agents.graph import build_decision_graph
from tests.test_policies import make_ctx

pytest.importorskip("torch")
pytest.importorskip("torch_geometric")


class _CountingNet:
    """Always picks one action; counts how often it is consulted."""

    def __init__(self, action: str) -> None:
        self.index = ACTION_NAMES.index(action)
        self.calls = 0

    def act(self, data, deterministic: bool = False):
        self.calls += 1
        p = np.zeros(len(ACTION_NAMES))
        p[self.index] = 1.0
        return {"action": self.index, "probs": p, "relay_order": [1, 2, 3],
                "log_prob": 0.0, "value": 0.0, "entropy": 0.0}


class _NearUniformNet(_CountingNet):
    def act(self, data, deterministic: bool = False):
        self.calls += 1
        p = np.full(len(ACTION_NAMES), 1.0 / len(ACTION_NAMES))
        return {"action": 1, "probs": p, "relay_order": [1, 2, 3],
                "log_prob": 0.0, "value": 0.0, "entropy": 2.19}


def _policy(net, gate=None):
    return AiHarpPolicy(network=net, gate=gate or ConfidenceGate(tau=0.0, enabled=False),
                        deterministic=True)


# ----------------------------------------------------- (b) suppress is final --
def test_network_suppress_is_final_for_that_message():
    net = _CountingNet("suppress")
    pol = _policy(net)
    assert pol.decide(make_ctx()).kind is ActionType.SUPPRESS
    for dups in (1, 2, 5):
        assert pol.decide(make_ctx(duplicate_count=dups)).kind is ActionType.SUPPRESS
    assert net.calls == 1, "a vehicle that suppressed was re-queried on a duplicate"


def test_settled_vehicle_records_no_further_transitions():
    """Re-asks were what attached a transmitter's reward to suppress decisions."""
    net = _CountingNet("suppress")
    pol = _policy(net)
    pol.record = True
    pol.decide(make_ctx())
    pol.decide(make_ctx(duplicate_count=1))
    assert len(pol.transitions) == 1


def test_carrier_is_still_re_evaluated():
    """Store-carry-forward needs re-evaluation; only suppression is final."""
    net = _CountingNet("carry_and_forward")
    pol = _policy(net)
    assert pol.decide(make_ctx()).kind is ActionType.CARRY
    pol.decide(make_ctx(duplicate_count=1))
    assert net.calls == 2


def test_settled_state_is_per_vehicle():
    net = _CountingNet("suppress")
    pol = _policy(net)
    pol.decide(make_ctx(index=3))
    pol.decide(make_ctx(index=4, duplicate_count=1))
    assert net.calls == 2


def test_reset_clears_settled_vehicles():
    net = _CountingNet("suppress")
    pol = _policy(net)
    pol.decide(make_ctx())
    pol.reset(0, np.random.default_rng(0))
    pol.decide(make_ctx(duplicate_count=1))
    assert net.calls == 2


def test_fallback_suppress_is_final_too():
    """At evaluation the gate hands decisions to weighted_p. Re-drawing it on
    every duplicate would inflate the fallback's cost the same way."""
    for seed in range(50):
        net = _NearUniformNet("suppress")
        pol = _policy(net, gate=ConfidenceGate(tau=0.5))
        first = pol.decide(make_ctx(sender_distance_m=100.0, rng=np.random.default_rng(seed)))
        if first.kind is ActionType.SUPPRESS:
            again = pol.decide(make_ctx(duplicate_count=1, rng=np.random.default_rng(seed + 1)))
            assert again.kind is ActionType.SUPPRESS
            assert net.calls == 1
            return
    pytest.fail("weighted_p never suppressed in 50 seeds; test cannot exercise the path")


# ------------------------------------------------------- (c) defer cancels ---
def test_agent_defer_cancels_on_duplicates_like_slotted_baselines():
    g = build_decision_graph(make_ctx())
    for k in (1, 2, 3):
        a = action_to_engine(ACTION_NAMES.index(f"defer_{k}"), g, [1, 2, 3])
        assert a.kind is ActionType.DEFER
        assert a.cancel_on_duplicates == 1


def test_defer_cancel_threshold_comes_from_config():
    from common.config import load_yaml

    assert load_yaml("agent.yaml")["action_space"]["defer_cancel_on_duplicates"] == 1


def test_policy_semantics_match_the_config_file():
    """The class attributes are the executable form of these config keys; a
    drift between them would change agent behaviour without any config edit."""
    from common.config import load_yaml

    space = load_yaml("agent.yaml")["action_space"]
    assert AiHarpPolicy.defer_cancel_on_duplicates == space["defer_cancel_on_duplicates"]
    assert AiHarpPolicy.suppress_is_final == space["suppress_is_final"]


def test_decide_passes_the_cancel_threshold_through():
    """End to end through decide(), not just the mapping function."""
    net = _CountingNet("defer_2")
    a = _policy(net).decide(make_ctx())
    assert a.kind is ActionType.DEFER
    assert a.delay_steps == 2 and a.cancel_on_duplicates == 1
