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
    """Always picks one action (or suppress, if that one is masked); counts calls."""

    def __init__(self, action: str) -> None:
        self.index = ACTION_NAMES.index(action)
        self.calls = 0
        self.masks: list = []

    def act(self, data, deterministic: bool = False, action_mask=None, uniform=None):
        self.calls += 1
        self.masks.append(action_mask)
        choice = self.index
        if action_mask is not None and not action_mask[choice]:
            choice = ACTION_NAMES.index("suppress")
        p = np.zeros(len(ACTION_NAMES))
        p[choice] = 1.0
        return {"action": choice, "probs": p, "relay_order": [1, 2, 3],
                "log_prob": 0.0, "value": 0.0, "entropy": 0.0}


class _NearUniformNet(_CountingNet):
    def act(self, data, deterministic: bool = False, action_mask=None, uniform=None):
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


# ------------------------------------------------ (d) carry is capped -------
def test_carry_is_masked_after_the_cap():
    """run6 collapsed onto carry: 92-94% of decisions, 16-20 per informed vehicle."""
    net = _CountingNet("carry_and_forward")
    pol = _policy(net)
    cap = AiHarpPolicy.max_carries_per_vehicle
    for _ in range(cap):
        assert pol.decide(make_ctx(trigger=_timer())).kind is ActionType.CARRY
    capped = pol.decide(make_ctx(trigger=_timer()))
    assert capped.kind is not ActionType.CARRY
    assert net.masks[:cap] == [None] * cap
    assert net.masks[cap] is not None and not net.masks[cap][ACTION_NAMES.index("carry_and_forward")]


def _timer():
    from agents.base import Trigger

    return Trigger.TIMER


def test_carry_cap_is_per_vehicle_and_reset_clears_it():
    net = _CountingNet("carry_and_forward")
    pol = _policy(net)
    for _ in range(AiHarpPolicy.max_carries_per_vehicle):
        pol.decide(make_ctx(index=3, trigger=_timer()))
    assert pol.decide(make_ctx(index=4, trigger=_timer())).kind is ActionType.CARRY
    pol.reset(0, np.random.default_rng(0))
    assert pol.decide(make_ctx(index=3, trigger=_timer())).kind is ActionType.CARRY


def test_capped_decision_records_its_mask_for_the_learner():
    net = _CountingNet("carry_and_forward")
    pol = _policy(net)
    pol.record = True
    for _ in range(AiHarpPolicy.max_carries_per_vehicle + 1):
        pol.decide(make_ctx(trigger=_timer()))
    masks = [t.action_mask for t in pol.transitions]
    assert all(m is None for m in masks[:-1])
    assert masks[-1] is not None and masks[-1][ACTION_NAMES.index("carry_and_forward")] is False


def test_carry_cap_matches_the_config_file():
    from common.config import load_yaml

    space = load_yaml("agent.yaml")["action_space"]
    assert AiHarpPolicy.max_carries_per_vehicle == space["max_carries_per_vehicle"]


def _real_net_and_graph():
    from agents.gat_drl import build_network
    from common.config import load_yaml

    net = build_network(load_yaml("agent.yaml"))
    net.eval()
    return net, build_decision_graph(make_ctx()).to_pyg()


def test_a_real_network_never_samples_a_masked_action():
    import torch

    torch.manual_seed(0)
    net, data = _real_net_and_graph()
    mask = np.ones(len(ACTION_NAMES), dtype=bool)
    mask[ACTION_NAMES.index("carry_and_forward")] = False
    for _ in range(200):
        out = net.act(data, deterministic=False, action_mask=mask)
        assert out["action"] != ACTION_NAMES.index("carry_and_forward")
        assert np.isfinite(out["log_prob"]) and np.isfinite(out["entropy"])
    assert out["probs"][ACTION_NAMES.index("carry_and_forward")] == pytest.approx(0.0, abs=1e-9)
    assert net.act(data, deterministic=True, action_mask=mask)["action"] != \
        ACTION_NAMES.index("carry_and_forward")


def test_evaluate_actions_reproduces_the_masked_log_prob():
    """PPO's ratio must be exactly 1 for an unchanged network under the mask."""
    import torch
    from torch_geometric.data import Batch

    torch.manual_seed(1)
    net, data = _real_net_and_graph()
    mask = np.ones(len(ACTION_NAMES), dtype=bool)
    mask[ACTION_NAMES.index("carry_and_forward")] = False
    out = net.act(data, deterministic=False, action_mask=mask)
    lp, _, ent = net.evaluate_actions(Batch.from_data_list([data]),
                                      torch.tensor([out["action"]]),
                                      torch.as_tensor(mask[None, :]))
    assert float(lp[0]) == pytest.approx(out["log_prob"], abs=1e-5)
    assert float(ent[0]) == pytest.approx(out["entropy"], abs=1e-5)
    unmasked_lp, _, _ = net.evaluate_actions(Batch.from_data_list([data]),
                                             torch.tensor([out["action"]]))
    assert float(unmasked_lp[0]) < float(lp[0])       # masking renormalises


def test_default_defer_delays_are_one_two_three_epochs():
    g = build_decision_graph(make_ctx())
    for k in (1, 2, 3):
        assert action_to_engine(ACTION_NAMES.index(f"defer_{k}"), g, [1, 2, 3]).delay_steps == k


def test_defer_epochs_match_the_config_file():
    from common.config import load_yaml

    space = load_yaml("agent.yaml")["action_space"]
    assert AiHarpPolicy.defer_epochs == tuple(space["defer_epochs"])


def test_long_wait_ablation_maps_defer_to_configured_epochs():
    net = _CountingNet("defer_3")
    pol = AiHarpPolicy(network=net, gate=ConfidenceGate(tau=0.0, enabled=False),
                       deterministic=True, defer_epochs=(5, 20, 50))
    a = pol.decide(make_ctx())
    assert a.kind is ActionType.DEFER and a.delay_steps == 50
    assert a.cancel_on_duplicates == 1
    assert _policy(_CountingNet("defer_3")).decide(make_ctx()).delay_steps == 3


def test_defer_epochs_are_validated():
    with pytest.raises(ValueError):
        AiHarpPolicy(defer_epochs=(5, 20))
    with pytest.raises(ValueError):
        AiHarpPolicy(defer_epochs=(0, 2, 3))


def test_training_policy_reads_defer_epochs_from_config():
    from agents.train import _training_policy
    from common.config import load_yaml

    cfg = load_yaml("agent.yaml")
    cfg["action_space"]["defer_epochs"] = [5, 20, 50]
    assert _training_policy(None, cfg, None).defer_epochs == (5, 20, 50)
    assert _training_policy(None, load_yaml("agent.yaml"), None).defer_epochs == (1, 2, 3)


def test_decide_passes_the_cancel_threshold_through():
    """End to end through decide(), not just the mapping function."""
    net = _CountingNet("defer_2")
    a = _policy(net).decide(make_ctx())
    assert a.kind is ActionType.DEFER
    assert a.delay_steps == 2 and a.cancel_on_duplicates == 1
