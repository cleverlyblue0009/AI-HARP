"""Phase 5 tests: confidence gate, action mapping, reward, and (if torch is
available) the GATv2 network.

The gate, the action mapping and the reward are torch-free by design, so most
of this runs on any interpreter. Network tests are skipped without torch.
"""

from __future__ import annotations

import numpy as np
import pytest

from agents.ai_harp import ACTION_NAMES, RewardWeights, action_to_engine
from agents.base import ActionType, Trigger
from agents.confidence import (
    ConfidenceGate,
    ensemble_confidence,
    normalised_entropy_confidence,
)
from agents.graph import build_decision_graph
from tests.test_policies import empty_ctx, make_ctx

torch = pytest.importorskip  # used per-test below


# ============================================================ confidence ====
def test_one_hot_distribution_is_maximally_confident():
    p = np.zeros(9)
    p[3] = 1.0
    assert normalised_entropy_confidence(p) == pytest.approx(1.0)


def test_uniform_distribution_is_minimally_confident():
    assert normalised_entropy_confidence(np.full(9, 1 / 9)) == pytest.approx(0.0, abs=1e-9)


def test_confidence_is_monotone_in_peakedness():
    sharp = np.array([0.9, 0.05, 0.05])
    flat = np.array([0.4, 0.3, 0.3])
    assert normalised_entropy_confidence(sharp) > normalised_entropy_confidence(flat)


def test_confidence_is_bounded():
    rng = np.random.default_rng(0)
    for _ in range(200):
        p = rng.dirichlet(np.ones(9))
        assert 0.0 <= normalised_entropy_confidence(p) <= 1.0


def test_unnormalised_input_is_handled():
    assert normalised_entropy_confidence(np.array([2.0, 0.0, 0.0])) == pytest.approx(1.0)


def test_ensemble_agreement_is_confident():
    members = [np.array([0.9, 0.05, 0.05])] * 5
    assert ensemble_confidence(members) > 0.9


def test_ensemble_disagreement_is_not_confident():
    members = [np.array([1.0, 0.0, 0.0])] * 3 + [np.array([0.0, 1.0, 0.0])] * 3
    assert ensemble_confidence(members) < 0.2


# ----------------------------------------------------------------- gate ----
def test_gate_falls_back_below_tau():
    gate = ConfidenceGate(tau=0.5)
    d = gate.evaluate(np.full(9, 1 / 9))        # confidence 0
    assert d.used_fallback and not d.used_learned


def test_gate_allows_the_learned_action_above_tau():
    gate = ConfidenceGate(tau=0.5)
    p = np.zeros(9)
    p[0] = 1.0
    assert gate.evaluate(p).used_learned


def test_tau_zero_never_falls_back():
    """tau = 0 is the 'gate off' ablation and needs no separate code path."""
    gate = ConfidenceGate(tau=0.0)
    for _ in range(20):
        assert not gate.evaluate(np.full(9, 1 / 9)).used_fallback
    assert gate.fallback_rate == 0.0


def test_disabled_gate_never_falls_back():
    gate = ConfidenceGate(tau=0.9, enabled=False)
    assert not gate.evaluate(np.full(9, 1 / 9)).used_fallback


def test_higher_tau_never_falls_back_less():
    """Monotonicity is what makes the tau sweep interpretable."""
    rng = np.random.default_rng(1)
    probs = [rng.dirichlet(np.ones(9)) for _ in range(300)]
    rates = []
    for tau in (0.0, 0.1, 0.3, 0.5, 0.7, 0.9):
        gate = ConfidenceGate(tau=tau)
        for p in probs:
            gate.evaluate(p)
        rates.append(gate.fallback_rate)
    assert rates == sorted(rates)


def test_gate_reports_fallback_rate_as_a_metric():
    gate = ConfidenceGate(tau=0.5)
    gate.evaluate(np.full(9, 1 / 9))
    sharp = np.zeros(9)
    sharp[0] = 1.0
    gate.evaluate(sharp)
    assert gate.fallback_rate == pytest.approx(0.5)
    s = gate.stats()
    assert "gate_fallback_rate" in s and "gate_confidence_mean" in s


def test_gate_rejects_bad_configuration():
    with pytest.raises(ValueError):
        ConfidenceGate(tau=1.5)
    with pytest.raises(ValueError):
        ConfidenceGate(method="magic")


def test_gate_reads_from_yaml():
    from common.config import load_yaml

    gate = ConfidenceGate.from_config(load_yaml("agent.yaml"))
    assert 0.0 <= gate.tau <= 1.0
    assert gate.method in ("entropy", "ensemble")


# ======================================================== action mapping ====
def test_every_action_maps_to_something_the_engine_accepts():
    g = build_decision_graph(make_ctx())
    order = [3, 2, 1]
    for i in range(len(ACTION_NAMES)):
        assert action_to_engine(i, g, order).kind in set(ActionType)


def test_relay_actions_designate_by_attention_rank():
    """relay_top_k must designate the k-th most ATTENDED neighbour, which is
    what makes the attention heatmap show the quantity that drove the decision."""
    g = build_decision_graph(make_ctx())
    order = [3, 1, 2]                      # node indices, best-attended first
    a1 = action_to_engine(ACTION_NAMES.index("relay_top_1"), g, order)
    a2 = action_to_engine(ACTION_NAMES.index("relay_top_2"), g, order)
    assert a1.kind is ActionType.RELAY
    assert a1.relay_indices == (g.relay_target(2),)   # node 3 -> neighbour idx 2
    assert a2.relay_indices == (g.relay_target(0),)   # node 1 -> neighbour idx 0


def test_relay_degrades_to_broadcast_rather_than_silence():
    """A relay action that quietly became a no-op would be invisible failure."""
    g = build_decision_graph(empty_ctx())
    a = action_to_engine(ACTION_NAMES.index("relay_top_3"), g, [])
    assert a.kind is ActionType.BROADCAST


def test_defer_actions_carry_their_epoch_count():
    g = build_decision_graph(make_ctx())
    for k in (1, 2, 3):
        a = action_to_engine(ACTION_NAMES.index(f"defer_{k}"), g, [1, 2, 3])
        assert a.kind is ActionType.DEFER and a.delay_steps == k


def test_carry_action_uses_the_configured_interval():
    g = build_decision_graph(make_ctx())
    a = action_to_engine(ACTION_NAMES.index("carry_and_forward"), g, [], carry_epochs=7)
    assert a.kind is ActionType.CARRY and a.delay_steps == 7


def test_action_names_match_the_config():
    from common.config import load_yaml

    assert tuple(load_yaml("agent.yaml")["action_space"]["actions"]) == ACTION_NAMES


# ============================================================== policy ======
def test_agent_without_a_network_is_the_analytic_fallback():
    """Makes the policy runnable (and the registry importable) with no torch."""
    from agents.registry import build_policy

    p = build_policy("ai_harp")
    a = p.decide(make_ctx())
    assert a.kind in set(ActionType)


def test_agent_never_relays_the_same_message_twice():
    from agents.registry import build_policy

    p = build_policy("ai_harp")
    assert p.decide(make_ctx(own_tx_count=1)).kind is ActionType.SUPPRESS


def test_agent_originator_always_transmits():
    from agents.registry import build_policy

    p = build_policy("ai_harp")
    a = p.decide(make_ctx(trigger=Trigger.ORIGINATE, sender_index=None,
                          sender_dx=0.0, sender_dy=0.0))
    assert a.kind in (ActionType.BROADCAST, ActionType.RELAY)


# ============================================================== reward ======
def test_reward_weights_load_from_yaml():
    from common.config import load_yaml

    w = RewardWeights.from_config(load_yaml("agent.yaml"))
    assert w.w1_relevance_informed > 0 and w.w2_transmission_cost > 0


def test_configured_weights_do_not_make_flooding_optimal():
    """Regression guard on the calibration.

    The total price of a transmission is w2 + w3 * E[collisions caused]. If it
    drops below mean_relevance / flooding_cost, blind flooding becomes
    reward-optimal and the 'learned' policy is just a flood.
    """
    from common.config import load_yaml

    w = RewardWeights.from_config(load_yaml("agent.yaml"))
    collisions_at_target = 2.0        # measured, rural d=20 at the target point
    total = w.w2_transmission_cost + w.w3_collision * collisions_at_target
    flood_optimal = 0.75 / 2.11
    assert total > flood_optimal * 2, (
        f"total tx price {total:.2f} is too close to the flood-optimal "
        f"{flood_optimal:.2f}; the agent will learn to flood"
    )


def test_configured_weights_do_not_force_silence():
    """The other failure mode: an effective cost so high that never
    transmitting is optimal. This is what an uncalibrated w3 caused."""
    from common.config import load_yaml

    w = RewardWeights.from_config(load_yaml("agent.yaml"))
    total = w.w2_transmission_cost + w.w3_collision * 2.0
    implied_break_even = 0.75 / total
    assert 0.2 < implied_break_even < 1.5, (
        f"implied break-even {implied_break_even:.3f} tx/informed is far from "
        "the measured baseline cluster (~0.41); the agent will not aim there"
    )


def test_collisions_are_attributed_per_transmitter_not_as_a_mean():
    """The bug this guards: a network-mean collision term is a constant
    multiple of tx_count, carries no per-vehicle signal, and silently
    multiplied the calibrated transmission cost by 5.5x."""
    from experiments.run_sim import RunSpec, run_single

    _, res = run_single(
        RunSpec(density_veh_km_lane=20, seed=0, policy="flooding"), return_result=True
    )
    cc = res.collisions_caused
    assert cc is not None
    assert cc.sum() == pytest.approx(res.n_fail_sinr, rel=1e-6)
    tx = res.tx_count > 0
    assert cc[tx].std() > 0.5, "no per-vehicle variation in attributed collisions"


# ============================================================== network =====
def test_network_forward_and_batching():
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    import torch as _t

    from agents.gat_drl import ActorCritic, EncoderConfig, graphs_to_batch

    _t.manual_seed(0)
    net = ActorCritic(EncoderConfig())
    g = build_decision_graph(make_ctx())
    out = net.act(g.to_pyg())
    assert 0 <= out["action"] < len(ACTION_NAMES)
    assert np.isclose(out["probs"].sum(), 1.0)

    batch = graphs_to_batch([build_decision_graph(make_ctx()) for _ in range(5)])
    res = net(batch)
    assert tuple(res["logits"].shape) == (5, len(ACTION_NAMES))
    assert tuple(res["value"].shape) == (5,)


def test_attention_produces_a_relay_ranking():
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    import torch as _t

    from agents.gat_drl import ActorCritic, EncoderConfig

    _t.manual_seed(0)
    out = ActorCritic(EncoderConfig()).act(build_decision_graph(make_ctx()).to_pyg())
    order = out["relay_order"]
    assert len(order) == 3
    assert set(order) == {1, 2, 3}, "ranking must cover the neighbours, not the holder"


def test_encoder_variants_all_run():
    """The ablation swaps a config field, not a code path."""
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from agents.gat_drl import ActorCritic, EncoderConfig

    g = build_decision_graph(make_ctx()).to_pyg()
    for kind in ("gatv2", "gcn", "mlp"):
        out = ActorCritic(EncoderConfig(kind=kind)).act(g)
        assert 0 <= out["action"] < len(ACTION_NAMES)


def test_dueling_dqn_decomposition_is_identifiable():
    """Q = V + A - mean(A). Without the mean subtraction V and A drift by an
    arbitrary constant and the learned value is meaningless."""
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    import torch as _t

    from agents.gat_drl import DuelingQNetwork, EncoderConfig

    _t.manual_seed(0)
    net = DuelingQNetwork(EncoderConfig())
    out = net(build_decision_graph(make_ctx()).to_pyg())
    q, v = out["q"][0], out["value"][0]
    assert _t.allclose(q.mean(), v, atol=1e-5)


def test_gae_is_computed_per_vehicle_not_across_vehicles():
    """Transitions from different vehicles are not consecutive states of one
    trajectory; bootstrapping across them would propagate value between
    unrelated situations."""
    pytest.importorskip("torch")
    from types import SimpleNamespace

    from agents.train import compute_gae

    t = [
        SimpleNamespace(vehicle=1, step=0, reward=1.0, value=0.0),
        SimpleNamespace(vehicle=2, step=1, reward=100.0, value=0.0),
        SimpleNamespace(vehicle=1, step=2, reward=1.0, value=0.0),
    ]
    adv, _ = compute_gae(t, gamma=0.99, lam=0.95)
    # Vehicle 1's first decision must not absorb vehicle 2's huge reward.
    assert adv[0] < 3.0


# ============================================================ seed split ====
def _train_cfgs():
    from common.config import load_yaml

    return load_yaml("agent.yaml"), {"experiment": load_yaml("experiment.yaml")}


def test_committed_training_seeds_are_disjoint_from_evaluation():
    """Training on seeds 0-9 would score the agent on traffic it trained on."""
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from agents.train import check_seed_split, training_seed_pool

    cfg, cfgs = _train_cfgs()
    check_seed_split(cfg, cfgs)                     # must not raise
    pool = set(training_seed_pool(cfg).tolist())
    assert not pool & set(cfgs["experiment"]["compare"]["seeds"])
    assert not pool & set(cfg["graph"]["normalisation"]["holdout_seeds"])


def test_seed_split_guard_rejects_an_overlapping_pool():
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    import copy

    from agents.train import check_seed_split

    cfg, cfgs = _train_cfgs()
    bad = copy.deepcopy(cfg)
    bad["training"]["train_seed_pool"] = {"start": 5, "count": 10}   # hits 5-9
    with pytest.raises(ValueError, match="evaluation"):
        check_seed_split(bad, cfgs)


def test_episode_seeds_come_from_the_bounded_pool():
    """An unbounded draw made every episode a trace-cache miss."""
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from agents.train import sample_episode_specs, training_seed_pool

    cfg, _ = _train_cfgs()
    specs = sample_episode_specs(cfg, 0.5, np.random.default_rng(0), 200)
    assert {s.seed for s in specs} <= set(training_seed_pool(cfg).tolist())


def test_transitions_the_gate_overrode_are_never_trained_on():
    """Regression: with the gate active in training, an untrained policy fell
    back on 100% of decisions, and PPO credited weighted_p's outcomes to
    actions the network sampled but never executed."""
    from types import SimpleNamespace

    from agents.ai_harp import executed_transitions

    t = [SimpleNamespace(used_fallback=False), SimpleNamespace(used_fallback=True),
         SimpleNamespace(used_fallback=False)]
    kept = executed_transitions(t)
    assert len(kept) == 2
    assert not any(x.used_fallback for x in kept)


def test_untrained_policy_would_trip_the_deployment_gate():
    """Why training cannot run with the gate on: a near-uniform distribution
    over nine actions has confidence ~0, below any useful tau."""
    gate = ConfidenceGate(tau=0.5)
    near_uniform = np.full(9, 1 / 9) + np.linspace(-0.005, 0.005, 9)
    assert gate.evaluate(near_uniform / near_uniform.sum()).used_fallback
