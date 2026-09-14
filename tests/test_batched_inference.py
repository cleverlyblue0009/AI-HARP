"""Batched decision inference: the correctness gate (user decision).

Batched and per-decision rollouts from the same seed must agree exactly on
everything that determines behaviour -- actions, action masks, rewards,
decision graphs, vehicle/step order, episode outcome -- and to within
``FLOAT_TOL`` on recorded log-probs, values and entropies. Exact float
equality is not attainable on CPU: dense kernels differ by matrix shape
(a bare nn.Linear over 1,300 rows vs 13-row chunks differs by 3.6e-7).
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

import torch  # noqa: E402
from torch_geometric.data import Batch  # noqa: E402

from agents.ai_harp import ACTION_NAMES, AiHarpPolicy  # noqa: E402
from agents.constrained_reward import build_training_objective  # noqa: E402
from agents.gat_drl import build_network, sample_inverse_cdf  # noqa: E402
from agents.graph import build_decision_graph  # noqa: E402
from agents.train import EpisodeSpec, _training_policy, collect_rollouts  # noqa: E402
from common.config import load_yaml  # noqa: E402
from tests.test_policies import make_ctx  # noqa: E402

FLOAT_TOL = 1e-4
CARRY = ACTION_NAMES.index("carry_and_forward")


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    cfg = load_yaml("agent.yaml")
    cfgs = {"phy": load_yaml("phy.yaml"), "hazard": load_yaml("hazard.yaml"),
            "experiment": load_yaml("experiment.yaml")}
    torch.manual_seed(0)
    net = build_network(cfg)
    objective = build_training_objective(cfg, tmp_path_factory.mktemp("targets"))
    return cfg, cfgs, net, objective


# --------------------------------------------------------------- sampling ---
def test_inverse_cdf_never_picks_a_zero_probability_action():
    probs = torch.tensor([[0.2, 0.0, 0.5, 0.0, 0.3, 0.0]])
    for u in torch.linspace(0.0, 1.0, 401):
        a = int(sample_inverse_cdf(probs, u[None])[0])
        assert probs[0, a] > 0, (float(u), a)


def test_inverse_cdf_matches_the_distribution():
    torch.manual_seed(3)
    probs = torch.tensor([[0.1, 0.6, 0.3]]).repeat(20000, 1)
    counts = torch.bincount(sample_inverse_cdf(probs, torch.rand(20000)), minlength=3)
    assert torch.allclose(counts / 20000.0, torch.tensor([0.1, 0.6, 0.3]), atol=0.015)


def test_one_batched_uniform_draw_equals_successive_draws():
    torch.manual_seed(5)
    a = torch.rand(64)
    torch.manual_seed(5)
    b = torch.stack([torch.rand(()) for _ in range(64)])
    assert torch.equal(a, b)


# ---------------------------------------------------------- network level ---
def _graphs(n):
    rng = np.random.default_rng(0)
    out = []
    for k in range(n):
        ctx = make_ctx(index=k, sender_distance_m=float(rng.uniform(50, 500)),
                       duplicate_count=int(rng.integers(0, 3)))
        out.append(build_decision_graph(ctx).to_pyg())
    return out


def test_act_batch_agrees_with_act_under_the_same_uniforms(setup):
    _, _, net, _ = setup
    graphs = _graphs(23)
    masks = torch.ones(23, len(ACTION_NAMES), dtype=torch.bool)
    masks[::3, CARRY] = False
    torch.manual_seed(11)
    u = torch.rand(23)
    batched = net.act_batch(Batch.from_data_list(graphs), action_masks=masks, uniforms=u)
    for j, g in enumerate(graphs):
        one = net.act(g, action_mask=masks[j].numpy(), uniform=float(u[j]))
        assert batched[j]["action"] == one["action"]
        assert batched[j]["relay_order"] == one["relay_order"]
        for k in ("log_prob", "value", "entropy"):
            assert batched[j][k] == pytest.approx(one[k], abs=FLOAT_TOL)
        assert np.allclose(batched[j]["probs"], one["probs"], atol=FLOAT_TOL)
        if not masks[j, CARRY]:
            assert batched[j]["action"] != CARRY


# ----------------------------------------------------------- policy level ---
def test_decide_batch_equals_sequential_decide(setup):
    cfg, _, net, _ = setup
    ctxs = [make_ctx(index=k, sender_distance_m=40.0 + 30 * k) for k in range(15)]

    def run(batched):
        pol = _training_policy(net, cfg, None)
        pol.reset(0, np.random.default_rng(1))
        pol._carries = {3: AiHarpPolicy.max_carries_per_vehicle}     # vehicle 3 is capped
        torch.manual_seed(7)
        acts = pol.decide_batch(ctxs) if batched else [pol.decide(c) for c in ctxs]
        return acts, pol.transitions

    a_seq, t_seq = run(False)
    a_bat, t_bat = run(True)
    assert [(a.kind, a.delay_steps, a.relay_indices, a.cancel_on_duplicates) for a in a_seq] == \
           [(a.kind, a.delay_steps, a.relay_indices, a.cancel_on_duplicates) for a in a_bat]
    _assert_transitions_match(t_seq, t_bat)
    assert t_bat[3].action_mask is not None and t_bat[3].action_mask[CARRY] is False


def test_decide_batch_with_a_repeated_vehicle_falls_back_to_sequential(setup):
    cfg, _, net, _ = setup
    ctxs = [make_ctx(index=5), make_ctx(index=5, duplicate_count=1)]
    pol = _training_policy(net, cfg, None)
    pol.reset(0, np.random.default_rng(0))
    torch.manual_seed(0)
    batched = pol.decide_batch(ctxs)
    pol2 = _training_policy(net, cfg, None)
    pol2.reset(0, np.random.default_rng(0))
    torch.manual_seed(0)
    seq = [pol2.decide(c) for c in ctxs]
    assert [a.kind for a in batched] == [a.kind for a in seq]


# ---------------------------------------------------------- episode level ---
def _assert_transitions_match(a, b):
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert (x.vehicle, x.step, x.action, x.action_mask, x.used_fallback) == \
               (y.vehicle, y.step, y.action, y.action_mask, y.used_fallback)
        assert x.reward == y.reward
        assert x.graph.x.tobytes() == y.graph.x.tobytes()
        assert x.graph.edge_index.tobytes() == y.graph.edge_index.tobytes()
        assert x.graph.edge_attr.tobytes() == y.graph.edge_attr.tobytes()
        for k in ("log_prob", "value", "entropy"):
            assert getattr(x, k) == pytest.approx(getattr(y, k), abs=FLOAT_TOL)


@pytest.mark.parametrize("spec", [EpisodeSpec("rural_highway", 20.0, "clear", "fog_bank", 100),
                                  EpisodeSpec("urban_nlos", 5.0, "clear", "crash", 102)])
def test_batched_episode_matches_per_decision_episode(setup, spec):
    """THE gate: same seed, batching on vs off."""
    cfg, cfgs, net, objective = setup

    def run(batch):
        pol = _training_policy(net, cfg, None)
        pol.batch_decisions = batch
        return collect_rollouts([spec], [42], net, pol, objective, cfgs)[0]

    (t_off, i_off), (t_on, i_on) = run(False), run(True)
    assert len(t_on) > 20, "episode too small to exercise batching"
    _assert_transitions_match(t_off, t_on)
    for k in ("obj_coverage", "obj_target", "transmissions", "informed", "lambda_group",
              "carry_rate", "decisions_per_informed", "obj_cost_per_at_risk"):
        assert i_off[k] == i_on[k], k


def test_batched_rollouts_are_byte_identical_on_repeat(setup):
    cfg, cfgs, net, objective = setup
    spec = EpisodeSpec("rural_highway", 20.0, "clear", "fog_bank", 100)

    def run():
        pol = _training_policy(net, cfg, None)
        assert pol.batch_decisions
        return collect_rollouts([spec], [42], net, pol, objective, cfgs)[0][0]

    a, b = run(), run()
    assert [(t.action, t.log_prob, t.value, t.entropy, t.reward) for t in a] == \
           [(t.action, t.log_prob, t.value, t.entropy, t.reward) for t in b]


def test_engine_batches_only_policies_that_opt_in():
    from agents.greedy import GreedyFarthestRelay

    assert not getattr(GreedyFarthestRelay(), "batch_decisions", False), \
        "greedy reads was_designated, so deferring its decisions would not be exact"
