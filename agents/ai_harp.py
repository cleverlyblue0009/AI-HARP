"""Phase 5: the AI-HARP policy and its reward.

Runs through the identical paired-comparison path as every Phase 4 baseline --
same engine, same mobility, same hazard, same fading per seed -- because it
implements the same :class:`~agents.base.Policy` interface. Nothing about the
comparison is special-cased for the learned policy.

Causal features only, including in the reward
---------------------------------------------
Both the node features and the reward use the **causal** relevance field from
``hazard/risk_field.py``. The oracle field exists only to score results in
``analysis/metrics.py``. Training against oracle relevance would hand the agent
knowledge of which vehicles actually reach the hazard -- information no vehicle
has at decision time -- and the resulting policy would be undeployable while
scoring beautifully. ``tests/test_oracle_isolation.py`` enforces this by
parsing this module's imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from agents.base import (
    BROADCAST_NOW,
    SUPPRESS,
    Action,
    ActionType,
    DecisionContext,
    Policy,
    Trigger,
)
from agents.confidence import ConfidenceGate
from agents.graph import DecisionGraph, FeatureNormaliser, GraphConfig, build_decision_graph
from agents.registry import build_policy

#: Action names, duplicated from agents.gat_drl so this module stays importable
#: without torch (the gate and the action mapping are torch-free).
ACTION_NAMES: tuple[str, ...] = (
    "suppress", "broadcast_now", "defer_1", "defer_2", "defer_3",
    "relay_top_1", "relay_top_2", "relay_top_3", "carry_and_forward",
)


@dataclass
class Transition:
    """One recorded decision, for the learner."""

    graph: DecisionGraph
    action: int
    log_prob: float
    value: float
    entropy: float
    vehicle: int
    step: int
    used_fallback: bool
    confidence: float
    reward: float = 0.0
    advantage: float = 0.0
    ret: float = 0.0


def executed_transitions(transitions: list[Transition]) -> list[Transition]:
    """Only the transitions whose recorded action was actually executed.

    When the confidence gate falls back, the engine runs the analytic policy's
    action, not the network's sampled one. Training on such a transition credits
    the fallback's outcome to an action that never happened, which is exactly
    what deadlocked the first training run (fallback rate 1.0 at update 1).
    Training disables the gate, so this should drop nothing; it is the guard
    that keeps the invariant from depending on a config staying correct.
    """
    return [t for t in transitions if not t.used_fallback]


def action_to_engine(
    action_index: int, graph: DecisionGraph, relay_order: list[int], carry_epochs: int = 10
) -> Action:
    """Map a discrete action index onto an engine :class:`Action`.

    ``relay_top_k`` designates the k-th most *attended* neighbour, so the
    attention weights are what actually select the relay. When the graph has
    fewer neighbours than the requested rank, the action degrades to a plain
    broadcast rather than silently becoming a no-op -- a relay action that
    quietly turned into silence would be an invisible failure.
    """
    name = ACTION_NAMES[action_index]
    if name == "suppress":
        return SUPPRESS
    if name == "broadcast_now":
        return BROADCAST_NOW
    if name.startswith("defer_"):
        return Action(ActionType.DEFER, delay_steps=int(name.split("_")[1]))
    if name == "carry_and_forward":
        return Action(ActionType.CARRY, delay_steps=carry_epochs)
    if name.startswith("relay_top_"):
        rank = int(name.split("_")[-1]) - 1
        if rank < len(relay_order):
            node = relay_order[rank]          # 1-based node index into the graph
            vid = graph.relay_target(node - 1)
            if vid is not None:
                return Action(ActionType.RELAY, relay_indices=(vid,))
        return BROADCAST_NOW
    raise ValueError(f"unmapped action {name!r}")


class AiHarpPolicy(Policy):
    """GAT-DRL relay policy with a confidence gate and analytic fallback."""

    name = "ai_harp"
    wants_duplicate_callbacks = True

    def __init__(
        self,
        network: Any | None = None,
        gate: ConfidenceGate | None = None,
        normaliser: FeatureNormaliser | None = None,
        graph_cfg: GraphConfig | None = None,
        fallback_policy: str = "weighted_p",
        deterministic: bool = False,
        record: bool = False,
        phy: Any | None = None,
        carry_epochs: int = 10,
    ) -> None:
        super().__init__(fallback_policy=fallback_policy, deterministic=deterministic)
        self.network = network
        self.gate = gate or ConfidenceGate()
        self.normaliser = normaliser
        self.graph_cfg = graph_cfg or GraphConfig()
        self.fallback = build_policy(fallback_policy)
        self.deterministic = deterministic
        self.record = record
        self.phy = phy
        self.carry_epochs = carry_epochs
        self.transitions: list[Transition] = []
        self.n_relay_actions = 0
        self.n_decisions = 0

    def reset(self, n_vehicles: int, rng: np.random.Generator) -> None:
        self.fallback.reset(n_vehicles, rng)
        self.gate.reset()
        self.transitions = []
        self.n_relay_actions = 0
        self.n_decisions = 0

    # --------------------------------------------------------------- decide --
    def decide(self, ctx: DecisionContext) -> Action:
        # Same broadcast-suppression invariant every baseline obeys: a vehicle
        # that has already relayed this message must not relay it again.
        if ctx.own_tx_count > 0:
            return SUPPRESS
        if ctx.trigger is Trigger.ORIGINATE:
            return BROADCAST_NOW
        if self.network is None:
            return self.fallback.decide(ctx)

        graph = build_decision_graph(ctx, self.graph_cfg, self.phy)
        if self.normaliser is not None:
            graph = self.normaliser.apply(graph)

        out = self.network.act(graph.to_pyg(), deterministic=self.deterministic)
        decision = self.gate.evaluate(out["probs"])
        self.n_decisions += 1

        if self.record:
            self.transitions.append(Transition(
                graph=graph, action=int(out["action"]),
                log_prob=float(out.get("log_prob", 0.0)),
                value=float(out.get("value", 0.0)),
                entropy=float(out.get("entropy", 0.0)),
                vehicle=int(ctx.index), step=int(ctx.step),
                used_fallback=decision.used_fallback, confidence=decision.confidence,
            ))

        if decision.used_fallback:
            return self.fallback.decide(ctx)

        act = action_to_engine(int(out["action"]), graph,
                               out.get("relay_order", []), self.carry_epochs)
        if act.kind is ActionType.RELAY:
            self.n_relay_actions += 1
        return act

    def stats(self) -> dict[str, float]:
        s = self.gate.stats()
        s["relay_action_frac"] = (
            self.n_relay_actions / self.n_decisions if self.n_decisions else float("nan")
        )
        return s


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------
@dataclass
class RewardWeights:
    """From ``configs/agent.yaml -> reward``. w1/w2 is calibrated, not chosen."""

    w1_relevance_informed: float = 1.0
    w2_transmission_cost: float = 1.83
    w3_collision: float = 0.5
    w4_deadline_miss: float = 4.0
    w5_coverage_bonus: float = 2.0
    normalise_by_at_risk_set: bool = True

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "RewardWeights":
        r = cfg.get("reward", {})
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in r.items() if k in known})

    @property
    def w2_over_w1(self) -> float:
        return self.w2_transmission_cost / max(self.w1_relevance_informed, 1e-9)


def compute_rewards(
    result: Any, risk: Any, hazard: Any, weights: RewardWeights
) -> dict[int, float]:
    """Per-vehicle reward for one episode, using CAUSAL relevance throughout.

    Credit is attributed structurally rather than by a shaped heuristic: the
    engine records ``informed_by``, so the vehicles a given relay actually
    informed are known exactly. A relay is credited with the causal relevance
    of the vehicles it informed, at the moment it informed them, and charged
    for its own transmissions and for the collisions they contributed to.

    A shared terminal term carries the episode-level outcome (coverage and
    actionable deadline misses), which no single vehicle can be blamed for.
    """
    tr = result.trace
    n = tr.n_vehicles
    rewards: dict[int, float] = {}

    rel_mat = risk.relevance_matrix(tr, hazard)           # causal, [T, N]
    informed = result.informed_step >= 0
    idx = np.flatnonzero(informed)

    # --- w1: relevance-weighted vehicles this relay informed -----------------
    gain = np.zeros(n)
    for v in idx:
        parent = int(result.informed_by[v])
        if parent < 0:
            continue
        gain[parent] += float(rel_mat[result.informed_step[v], v])

    # --- w2: transmission cost ------------------------------------------------
    tx_cost = result.tx_count.astype(float)

    # --- w3: collisions this vehicle actually caused --------------------------
    # Per-transmitter attribution from the engine, NOT the network mean. The
    # mean carries no per-vehicle signal (it is just a constant multiple of
    # tx_count) and it silently inflated the effective transmission cost by
    # 5.5x, which would have driven the agent to near-silence.
    if result.collisions_caused is not None:
        collisions = np.asarray(result.collisions_caused, dtype=float)
    else:
        total_tx = max(int(result.n_transmissions), 1)
        collisions = tx_cost * (result.n_fail_sinr / total_tx)

    # --- terminal: coverage and actionable deadline misses --------------------
    peak = rel_mat.max(axis=0)
    at_risk = peak > risk.at_risk_threshold
    n_at_risk = max(int(at_risk.sum()), 1)
    covered = float((peak * (informed & at_risk)).sum() / max(peak[at_risk].sum(), 1e-9))

    deadline = float(hazard.safety_deadline_s)
    onset = float(hazard.onset_time_s)
    in_time = np.zeros(n, dtype=bool)
    in_time[idx] = (result.informed_step[idx] * tr.dt - onset) <= deadline
    miss_rate = float(1.0 - (in_time & at_risk).sum() / n_at_risk)

    terminal = (weights.w5_coverage_bonus * covered
                - weights.w4_deadline_miss * miss_rate)
    scale = float(n_at_risk) if weights.normalise_by_at_risk_set else 1.0

    for v in range(n):
        r = (weights.w1_relevance_informed * gain[v]
             - weights.w2_transmission_cost * tx_cost[v]
             - weights.w3_collision * collisions[v])
        rewards[v] = float(r + terminal / scale)
    return rewards


def episode_summary(rewards: dict[int, float]) -> dict[str, float]:
    vals = np.asarray(list(rewards.values()), dtype=float)
    return {
        "reward_total": float(vals.sum()),
        "reward_mean": float(vals.mean()) if vals.size else 0.0,
        "reward_min": float(vals.min()) if vals.size else 0.0,
        "reward_max": float(vals.max()) if vals.size else 0.0,
    }
