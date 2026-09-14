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
    #: Actions available at this decision (None = all). The learner must
    #: re-apply it when recomputing log-probs, or the PPO ratio compares a
    #: masked sample against an unmasked distribution.
    action_mask: tuple[bool, ...] | None = None


def assign_terminal_rewards(
    transitions: list[Transition], rewards: dict[int, float]
) -> list[Transition]:
    """Credit each vehicle's episode reward ONCE, on its last decision.

    ``run_episode`` used to copy the vehicle's full reward onto every one of its
    decisions, and GAE then sums along the vehicle's sequence -- so a vehicle
    deciding k times was credited roughly k times, and its EARLIER decisions
    absorbed the most. Replaying an untrained network (the policy PPO starts
    from) on rural d=40, 90% of suppress decisions belonged to vehicles that
    were re-asked on a later duplicate and transmitted anyway; the suppress
    decisions were credited a mean -9.51 against -10.61 for transmitting ones,
    so the ~12-point value of actually staying silent all but vanished and PPO
    drove suppress from ~15% of decisions to 0.35% in 13 updates.

    Earlier decisions still receive the outcome through GAE's discounted
    bootstrap, which is the right amount of credit rather than a copy.
    """
    last: dict[int, int] = {}
    for i, t in enumerate(transitions):
        j = last.get(t.vehicle)
        if j is None or t.step >= transitions[j].step:
            last[t.vehicle] = i
    for i, t in enumerate(transitions):
        t.reward = float(rewards.get(t.vehicle, 0.0)) if last[t.vehicle] == i else 0.0
    return transitions


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
    action_index: int, graph: DecisionGraph, relay_order: list[int], carry_epochs: int = 10,
    defer_cancel: int = 1,
) -> Action:
    """Map a discrete action index onto an engine :class:`Action`.

    ``relay_top_k`` designates the k-th most *attended* neighbour, so the
    attention weights are what actually select the relay. When the graph has
    fewer neighbours than the requested rank, the action degrades to a plain
    broadcast rather than silently becoming a no-op -- a relay action that
    quietly turned into silence would be an invisible failure.

    ``defer_k`` cancels on ``defer_cancel`` overheard duplicates, as every
    slotted baseline's deferral does. Without the cancellation a deferral was
    "broadcast later", strictly dominated by broadcasting now; measured, the
    trained agent's argmax mode drifted to defer and its cost stayed pinned at
    flooding's ~0.97 transmissions per informed vehicle while slotted_1p's
    identical-looking deferral reached 0.229.
    """
    name = ACTION_NAMES[action_index]
    if name == "suppress":
        return SUPPRESS
    if name == "broadcast_now":
        return BROADCAST_NOW
    if name.startswith("defer_"):
        return Action(ActionType.DEFER, delay_steps=int(name.split("_")[1]),
                      cancel_on_duplicates=int(defer_cancel))
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
        suppression_bias: float = 0.0,
    ) -> None:
        super().__init__(fallback_policy=fallback_policy, deterministic=deterministic,
                         suppression_bias=suppression_bias)
        self.suppression_bias = float(suppression_bias)
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

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: Any,
        tau: float | None = None,
        suppression_bias: float = 0.0,
        deterministic: bool = True,
        fallback_policy: str | None = None,
        checkpoint_sha: str | None = None,
        **_unused: Any,
    ) -> "AiHarpPolicy":
        """Build an EVALUATION policy from a training checkpoint.

        The confidence gate is enabled here (training disables it), with tau
        from the checkpoint's own config unless overridden. Feature statistics
        come from inside the checkpoint; a checkpoint written before they were
        embedded falls back to the stats file only if its provenance matches,
        and says so. ``checkpoint_sha`` is accepted purely so it enters the
        results config hash.
        """
        from pathlib import Path

        import torch

        from agents.gat_drl import build_network
        from common.config import PROJECT_ROOT, load_yaml
        from common.logging_utils import get_logger

        path = Path(checkpoint)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        state = torch.load(str(path), map_location="cpu", weights_only=False)
        cfg = state.get("config") or load_yaml("agent.yaml")

        net = build_network(cfg)
        net.load_state_dict(state["model"])
        net.eval()                      # dropout off for evaluation

        if state.get("normaliser_stats"):
            norm = FeatureNormaliser.from_dict(state["normaliser_stats"])
        else:
            stats_path = PROJECT_ROOT / cfg["graph"]["normalisation"]["stats_path"]
            norm = FeatureNormaliser.load(stats_path)
            if not norm.matches("full", cfg["training"]["train_scenarios"]):
                raise ValueError(
                    f"{path.name} has no embedded feature statistics and {stats_path.name} "
                    f"was fitted for {norm.provenance or 'unrecorded data'}; refusing to "
                    "evaluate with statistics it may not have been trained on."
                )
            get_logger("agents.ai_harp").warning(
                "%s has no embedded feature statistics; using %s, which cannot be "
                "proven identical to the ones it was trained with.", path.name,
                stats_path.name,
            )

        gcfg = cfg["confidence_gate"]
        gate = ConfidenceGate(tau=float(gcfg["tau"] if tau is None else tau),
                              method=str(gcfg.get("method", "entropy")), enabled=True)
        policy = cls(
            network=net, gate=gate, normaliser=norm,
            graph_cfg=GraphConfig.from_config(cfg),
            fallback_policy=fallback_policy or gcfg["fallback_policy"],
            deterministic=deterministic, suppression_bias=suppression_bias,
        )
        policy.params.update({"checkpoint": str(checkpoint),
                              "checkpoint_sha": checkpoint_sha, "tau": gate.tau})
        return policy

    #: Mirrors configs/agent.yaml -> action_space (asserted equal by tests).
    #: A deferred rebroadcast is cancelled on this many duplicates, exactly as
    #: for the slotted baselines; without it the agent's defer could not
    #: express slotted-style suppression at all.
    defer_cancel_on_duplicates: int = 1
    #: A suppress on a received message is final for that vehicle.
    suppress_is_final: bool = True
    #: A vehicle may carry the message at most this many times; after that
    #: carry_and_forward is masked out of its choices. run6's policy collapsed
    #: onto carry at update 41 (92-94% of decisions, 16-20 decisions per
    #: informed vehicle, entropy 0.73 -> 0.27, 35k transitions in a batch):
    #: carrying costs no transmission, never settles, and the reward lands only
    #: on a vehicle's last decision, so postponing was free. It reached fewer
    #: vehicles than slotted_1p (586 vs 792 at rural d=40 seed 105).
    max_carries_per_vehicle: int = 3

    def reset(self, n_vehicles: int, rng: np.random.Generator) -> None:
        self.fallback.reset(n_vehicles, rng)
        self.gate.reset()
        self.transitions = []
        self.n_relay_actions = 0
        self.n_decisions = 0
        self._settled: set[int] = set()
        self._carries: dict[int, int] = {}

    _CARRY = ACTION_NAMES.index("carry_and_forward")

    def _action_mask(self, vehicle: int) -> np.ndarray | None:
        """Available actions for ``vehicle``; ``None`` when nothing is masked."""
        carries = self.__dict__.setdefault("_carries", {})
        if carries.get(int(vehicle), 0) < self.max_carries_per_vehicle:
            return None
        mask = np.ones(len(ACTION_NAMES), dtype=bool)
        mask[self._CARRY] = False
        return mask

    #: Direction each action's logit moves under a POSITIVE suppression bias,
    #: in ACTION_NAMES order: suppress up; broadcast and the three relay
    #: actions down; deferral and carrying untouched.
    _BIAS_SIGN = np.array([+1, -1, 0, 0, 0, -1, -1, -1, 0], dtype=float)

    def _apply_suppression_bias(
        self, probs: np.ndarray, rng: np.random.Generator,
        mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, int]:
        """The agent's operating-curve knob, applied in logit space.

        Positive trades coverage for fewer transmissions, negative the reverse
        -- the same role p, n_slots or the counter threshold play for the
        baselines, so the agent is compared along a curve rather than at one
        point. Masked actions stay unavailable after biasing.
        """
        logits = np.log(np.clip(probs, 1e-12, 1.0)) + self.suppression_bias * self._BIAS_SIGN
        z = np.exp(logits - logits.max())
        if mask is not None:
            z = np.where(mask, z, 0.0)
        p = z / z.sum()
        action = int(np.argmax(p)) if self.deterministic else int(rng.choice(p.size, p=p))
        return p, action

    def _settle(self, vehicle: int, act: Action) -> Action:
        """Record a final suppress so later duplicates do not re-query.

        Measured on an untrained network (the policy PPO starts from) at rural
        d=40: 90% of suppress decisions were followed by a re-ask on a later
        duplicate and a transmission. That turned the +11.58 advantage of
        staying silent into -8.02, and PPO eliminated suppress. Carrying is
        not settled -- store-carry-forward exists to be re-evaluated.
        """
        if self.suppress_is_final and act.kind is ActionType.SUPPRESS:
            self._settled.add(int(vehicle))
        return act

    # --------------------------------------------------------------- decide --
    #: When True the engine collects all of an epoch loop's decisions and calls
    #: :meth:`decide_batch`. Exact for this policy because its decision graph
    #: never reads ``ctx.was_designated`` -- the one way an earlier decision in
    #: the same loop can reach a later vehicle's context. Training sets it from
    #: ``training.batch_inference``.
    batch_decisions: bool = False

    def decide(self, ctx: DecisionContext) -> Action:
        pre = self._pre_network(ctx)
        if pre is not None:
            return pre
        graph, mask = self._prepare(ctx)
        # The mask is applied INSIDE the network's sampling so the recorded
        # log-prob and entropy describe the distribution actually sampled.
        if mask is None:
            out = self.network.act(graph.to_pyg(), deterministic=self.deterministic)
        else:
            out = self.network.act(graph.to_pyg(), deterministic=self.deterministic,
                                   action_mask=mask)
        return self._post_network(ctx, graph, mask, out)

    def decide_batch(self, ctxs: list[DecisionContext]) -> list[Action]:
        """Decide for several vehicles with one forward pass; same result as
        calling :meth:`decide` on each in order.

        Pre-network checks have no side effects once a network is present, a
        vehicle appears at most once per engine loop, and post-processing runs
        in decision order, so the gate, fallback and suppression-bias RNG draws
        happen in exactly the sequential order. The network's uniforms are one
        ``torch.rand(B)`` draw, equal to B successive single draws. Falls back
        to sequential decisions whenever those conditions cannot be guaranteed.
        """
        indices = [int(c.index) for c in ctxs]
        if (self.network is None or not hasattr(self.network, "act_batch")
                or len(set(indices)) != len(indices)):
            return [self.decide(c) for c in ctxs]

        pres = [self._pre_network(c) for c in ctxs]
        need = [k for k, p in enumerate(pres) if p is None]
        prepared = {k: self._prepare(ctxs[k]) for k in need}
        outs: dict[int, dict[str, Any]] = {}
        if need:
            import torch
            from torch_geometric.data import Batch

            batch = Batch.from_data_list([prepared[k][0].to_pyg() for k in need])
            masks = None
            if any(prepared[k][1] is not None for k in need):
                full = np.ones(len(ACTION_NAMES), dtype=bool)
                masks = torch.as_tensor(np.stack(
                    [full if prepared[k][1] is None else prepared[k][1] for k in need]))
            results = self.network.act_batch(batch, deterministic=self.deterministic,
                                             action_masks=masks)
            outs = dict(zip(need, results))

        return [pres[k] if pres[k] is not None
                else self._post_network(c, prepared[k][0], prepared[k][1], outs[k])
                for k, c in enumerate(ctxs)]

    def _pre_network(self, ctx: DecisionContext) -> Action | None:
        """Decisions made without consulting the network, or None."""
        # Same broadcast-suppression invariant every baseline obeys: a vehicle
        # that has already relayed this message must not relay it again.
        if ctx.own_tx_count > 0:
            return SUPPRESS
        if ctx.trigger is Trigger.ORIGINATE:
            return BROADCAST_NOW
        if self.network is None:
            return self.fallback.decide(ctx)
        settled = self.__dict__.setdefault("_settled", set())
        if ctx.trigger is Trigger.RECEIVE and int(ctx.index) in settled:
            # Already decided to stay silent on this message: no re-query and
            # no transition, so no transmitter's reward can land on it.
            return SUPPRESS
        return None

    def _prepare(self, ctx: DecisionContext) -> tuple[DecisionGraph, np.ndarray | None]:
        graph = build_decision_graph(ctx, self.graph_cfg, self.phy)
        if self.normaliser is not None:
            graph = self.normaliser.apply(graph)
        return graph, self._action_mask(int(ctx.index))

    def _post_network(
        self, ctx: DecisionContext, graph: DecisionGraph, mask: np.ndarray | None,
        out: dict[str, Any],
    ) -> Action:
        probs = np.asarray(out["probs"], dtype=float)
        action = int(out["action"])
        if self.suppression_bias != 0.0:
            if self.record:
                # The recorded log-prob would describe the network's sample,
                # not the biased action actually executed: off-policy again.
                raise ValueError("suppression_bias is an evaluation knob; "
                                 "it must be 0 while recording transitions")
            probs, action = self._apply_suppression_bias(probs, ctx.rng, mask)
        decision = self.gate.evaluate(probs)
        self.n_decisions += 1

        if self.record:
            self.transitions.append(Transition(
                graph=graph, action=action,
                log_prob=float(out.get("log_prob", 0.0)),
                value=float(out.get("value", 0.0)),
                entropy=float(out.get("entropy", 0.0)),
                vehicle=int(ctx.index), step=int(ctx.step),
                used_fallback=decision.used_fallback, confidence=decision.confidence,
                action_mask=None if mask is None else tuple(bool(m) for m in mask),
            ))

        if decision.used_fallback:
            # The fallback's suppress is final too: re-drawing a probabilistic
            # scheme on every duplicate would inflate its cost the same way.
            return self._settle(ctx.index, self.fallback.decide(ctx))

        act = action_to_engine(action, graph, out.get("relay_order", []),
                               self.carry_epochs,
                               defer_cancel=self.defer_cancel_on_duplicates)
        if act.kind is ActionType.RELAY:
            self.n_relay_actions += 1
        if act.kind is ActionType.CARRY:
            carries = self.__dict__.setdefault("_carries", {})
            carries[int(ctx.index)] = carries.get(int(ctx.index), 0) + 1
        return self._settle(ctx.index, act)

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
