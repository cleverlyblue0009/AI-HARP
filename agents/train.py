"""Phase 5.5: PPO training with a density curriculum.

::

    D:/aiharp-env/python.exe -m agents.train --updates 200
    D:/aiharp-env/python.exe -m agents.train --smoke      # a few updates, minutes

Requires torch (see ENVIRONMENT.md).

Credit assignment
-----------------
A vehicle makes very few decisions per episode -- often exactly one -- so this
is much closer to a contextual bandit than to a long-horizon control problem.
GAE is still applied, but *along each vehicle's own decision sequence*, not
along wall-clock time: transitions from different vehicles are not consecutive
states of one trajectory, and bootstrapping across them would propagate value
between unrelated situations.

Held-out by construction
------------------------
``urban_grid`` is never trained on (unseen-topology test), and two hazard types
and two weather conditions are held out as well. The split lives in
``configs/agent.yaml`` so it cannot drift between training and evaluation.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn

from agents.ai_harp import AiHarpPolicy, RewardWeights, compute_rewards
from agents.confidence import ConfidenceGate
from agents.gat_drl import ActorCritic, EncoderConfig, build_network, count_parameters
from agents.graph import FeatureNormaliser, GraphConfig, build_decision_graph
from common.config import PROJECT_ROOT, load_yaml
from common.logging_utils import get_logger
from common.seeding import SeedBundle, set_global_determinism
from hazard.model import hazard_from_config
from hazard.risk_field import build_risk_field
from mobility.generate import get_trace, load_scenario
from sim.engine import DisseminationEngine, SimSettings
from sim.mac import build_mac
from sim.phy import build_phy

logger = get_logger("agents.train")


@dataclass
class EpisodeSpec:
    scenario: str
    density: float
    weather: str
    hazard_type: str
    seed: int


def curriculum_densities(cfg: dict[str, Any], progress: float) -> list[float]:
    """Densities available at this point in training.

    Starts connected and anneals toward sparse: a relay decision in dense
    traffic has an immediate, visible consequence, which is learnable; in the
    disconnected regime the consequence of carrying a message arrives many
    seconds later, which is not a good place to start.
    """
    stages = cfg["training"]["curriculum"]["stages"]
    for stage in stages:
        if progress <= float(stage["until_frac"]):
            return [float(d) for d in stage["densities"]]
    return [float(d) for d in stages[-1]["densities"]]


def stage_densities(cfg: dict[str, Any], stage: str) -> tuple[list[float], np.ndarray]:
    """Densities and sampling probabilities for a staged-curriculum stage."""
    c = cfg["training"]["curriculum"]
    if stage == "pretrain":
        d = [float(x) for x in c["pretrain"]["densities"]]
        return d, np.full(len(d), 1.0 / len(d))
    if stage == "finetune":
        f = c["finetune"]
        sparse = [float(x) for x in f["sparse_densities"]]
        dense = [float(x) for x in f["dense_densities"]]
        w = float(f.get("sparse_to_dense_weight", 3.0))
        p_sparse = w / (w + 1.0) if dense else 1.0      # sparse-only finetune
        probs = [p_sparse / len(sparse)] * len(sparse) + [(1.0 - p_sparse) / max(len(dense), 1)] * len(dense)
        return sparse + dense, np.asarray(probs, dtype=float)
    raise ValueError(f"unknown curriculum stage {stage!r}")


def sample_episode_specs(
    cfg: dict[str, Any], progress: float, rng: np.random.Generator, n: int,
    stage: str | None = None,
) -> list[EpisodeSpec]:
    """Episode specs for one update. ``stage`` selects a staged-curriculum
    stage; ``None`` uses the anneal schedule at ``progress``."""
    t = cfg["training"]
    if stage is None:
        densities, probs = curriculum_densities(cfg, progress), None
    else:
        densities, probs = stage_densities(cfg, stage)
    pool = training_seed_pool(cfg)
    out = []
    for _ in range(n):
        out.append(EpisodeSpec(
            scenario=str(rng.choice(t["train_scenarios"])),
            density=float(rng.choice(densities) if probs is None
                          else rng.choice(densities, p=probs)),
            weather=str(rng.choice(t["train_weather"])),
            hazard_type=str(rng.choice(t["train_hazards"])),
            seed=int(rng.choice(pool)),
        ))
    return out


def training_seed_pool(cfg: dict[str, Any]) -> np.ndarray:
    """The mobility seeds training may draw from.

    Bounded, for two reasons. Speed: seeds were drawn from [0, 1e6), so every
    episode was a trace-cache miss and paid for fresh mobility generation --
    the first 40-update run burned ~50 CPU-minutes largely on that. And the
    train/eval split: an unbounded draw can land on the evaluation seeds (0-9),
    silently evaluating the agent on traffic it trained on.
    """
    p = cfg["training"].get("train_seed_pool", {"start": 100, "count": 32})
    return np.arange(int(p["start"]), int(p["start"]) + int(p["count"]))


def check_seed_split(cfg: dict[str, Any], cfgs: dict[str, Any]) -> None:
    """Refuse to train on seeds used for evaluation or normalisation."""
    train = set(training_seed_pool(cfg).tolist())
    evaluation = set(cfgs["experiment"].get("compare", {}).get("seeds", []))
    evaluation |= set(cfgs["experiment"].get("sweep", {}).get("seeds", []))
    holdout = set(cfg["graph"]["normalisation"]["holdout_seeds"])
    for name, other in (("evaluation", evaluation), ("normalisation hold-out", holdout)):
        overlap = sorted(train & set(other))
        if overlap:
            raise ValueError(
                f"training seed pool overlaps the {name} seeds {overlap}; the agent "
                "would be scored on traffic it trained on. Move train_seed_pool."
            )


def run_episode(
    spec: EpisodeSpec,
    policy: AiHarpPolicy,
    objective: Any,
    cfgs: dict[str, Any],
) -> tuple[list, dict[str, float]]:
    """One episode under the constrained objective; transitions carry rewards.

    ``objective`` is an :class:`agents.constrained_reward.TrainingObjective`.
    Rewards use the CURRENT value of this cell's group multiplier. Multipliers
    are stepped once per PPO update from all of that update's episodes, never
    inside an episode, so every transition of a group in a batch was scored at
    the same price.
    """
    phy_cfg, hz_cfg, exp_cfg = cfgs["phy"], cfgs["hazard"], cfgs["experiment"]
    scenario_cfg = load_scenario(spec.scenario)
    seeds = SeedBundle(master_seed=spec.seed)

    trace = get_trace(scenario_cfg, spec.density, spec.seed,
                      weather=spec.weather, phy_cfg=phy_cfg)
    hazard = hazard_from_config(hz_cfg, trace.meta, overrides={"type": spec.hazard_type})
    phy = build_phy(phy_cfg, scenario=scenario_cfg["name"], weather=spec.weather,
                    seed=spec.seed, trace_meta=trace.meta)
    mac = build_mac(phy_cfg, phy)
    risk = build_risk_field(hz_cfg, trace, hazard)

    policy.phy = phy
    policy.record = True
    engine = DisseminationEngine(trace, phy, mac, risk, hazard, policy, seeds,
                                 SimSettings.from_config(exp_cfg))
    result = engine.run()

    from agents.ai_harp import assign_terminal_rewards
    from agents.constrained_reward import episode_rewards

    target = objective.target_for(spec.scenario, spec.density, spec.weather, spec.hazard_type)
    group = objective.group_for(spec.scenario, spec.density, spec.weather, spec.hazard_type)
    lam = float(objective.multipliers.value(group))
    rewards, outcome = episode_rewards(result, risk, hazard, lam, objective.objective, target)
    # Once per vehicle, on its last decision -- not copied onto every decision.
    assign_terminal_rewards(policy.transitions, rewards)

    informed = int((result.informed_step >= 0).sum())
    info = {
        "transmissions": float(result.n_transmissions),
        "informed": float(informed),
        "reward_mean": float(np.mean([t.reward for t in policy.transitions]))
        if policy.transitions else 0.0,
        "n_decisions": float(len(policy.transitions)),
        "lambda": lam,
        "lambda_group": group,
        # Share of decisions that carried, and decisions per informed vehicle:
        # the two numbers that exposed run6's collapse onto carry.
        "carry_rate": float(np.mean([t.action == AiHarpPolicy._CARRY for t in policy.transitions]))
        if policy.transitions else 0.0,
        "decisions_per_informed": float(len(policy.transitions) / max(informed, 1)),
        **{f"obj_{k}": v for k, v in outcome.as_dict().items()},
        **policy.stats(),
    }
    return list(policy.transitions), info


# ---------------------------------------------------------------------------
# Rollout collection, serial or across worker processes
# ---------------------------------------------------------------------------
# Profiled on run7's checkpoint (one update's rollouts + PPO step): per-decision
# network inference 44%, simulator + reward 25%, PPO step 25%, decision graphs
# 5%. A GPU only helps the PPO step (~1.2x overall; single 4-node graphs are
# slower to ship to it), so the episodes of an update run in parallel CPU
# processes instead, one torch thread each.
_WORKER: dict[str, Any] = {}


def run_seeded_episode(
    policy: AiHarpPolicy, spec: EpisodeSpec, episode_seed: int, objective: Any,
    cfgs: dict[str, Any],
) -> tuple[list, dict[str, float]]:
    """One episode whose policy randomness depends only on ``episode_seed``.

    Seeds the policy's NumPy RNG and torch's global RNG (action sampling) from
    the episode seed and restores torch's RNG afterwards, so the result does
    not depend on which process runs the episode or what ran before it, and
    the caller's torch stream (PPO minibatch shuffling) is left untouched.
    """
    torch_state = torch.get_rng_state()
    try:
        torch.manual_seed(int(episode_seed))
        policy.reset(0, np.random.default_rng(int(episode_seed)))
        return run_episode(spec, policy, objective, cfgs)
    finally:
        torch.set_rng_state(torch_state)


def _training_policy(net: Any, cfg: dict[str, Any], normaliser: Any) -> AiHarpPolicy:
    # The confidence gate is DISABLED during training rollouts (see train()).
    policy = AiHarpPolicy(
        network=net,
        gate=ConfidenceGate(tau=0.0, method=cfg["confidence_gate"]["method"], enabled=False),
        normaliser=normaliser, graph_cfg=GraphConfig.from_config(cfg),
        fallback_policy=cfg["confidence_gate"]["fallback_policy"], record=True,
    )
    policy.batch_decisions = bool(cfg["training"].get("batch_inference", True))
    return policy


def _init_rollout_worker(cfg: dict[str, Any], cfgs: dict[str, Any],
                         normaliser_stats: dict[str, Any] | None) -> None:
    torch.set_num_threads(1)
    net = build_network(cfg)
    norm = FeatureNormaliser.from_dict(normaliser_stats) if normaliser_stats else None
    _WORKER.update(net=net, policy=_training_policy(net, cfg, norm), cfgs=cfgs)


def _rollout_task(work: tuple[dict[str, Any], Any, EpisodeSpec, int]) -> tuple[list, dict]:
    state_dict, objective, spec, episode_seed = work
    _WORKER["net"].load_state_dict(state_dict)
    return run_seeded_episode(_WORKER["policy"], spec, episode_seed, objective, _WORKER["cfgs"])


def resolve_device(value: Any) -> torch.device:
    """``training.device``: ``cpu``, ``cuda``, or ``auto`` (CUDA when available)."""
    v = str(value).strip().lower()
    if v == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(v)


def resolve_workers(value: Any, episodes_per_update: int | None = None) -> int:
    """``training.rollout_workers``: an integer, or ``auto``.

    ``auto`` is ``cpu_count - 1`` capped at ``episodes_per_update``: one episode
    runs in one worker, so extra workers only idle. Measured on the 12-thread
    laptop (dense episodes): 8 workers 13.1 s, 11 workers 13.6 s with 3 idle.
    """
    if isinstance(value, str) and value.strip().lower() == "auto":
        import os

        n = max(1, (os.cpu_count() or 2) - 1)
        return min(n, int(episodes_per_update)) if episodes_per_update else n
    return max(1, int(value))


@dataclass
class ConstraintEarlyStop:
    """Stop finetuning once the coverage constraints have held for a while.

    An update counts as "met" when every group sampled in it has pooled
    shortfall <= 0. Stopping requires ``window`` consecutive met updates AND
    every group in ``required`` to have appeared within that run -- otherwise
    a rarely drawn sparse group could "pass" simply by not being sampled.
    """

    window: int
    required: frozenset[str]
    run: int = 0
    seen: list[list[str]] = field(default_factory=list)

    def update(self, shortfall_by_group: dict[str, float]) -> bool:
        finite = {g: s for g, s in shortfall_by_group.items() if np.isfinite(s)}
        if finite and all(s <= 0.0 for s in finite.values()):
            self.run += 1
            self.seen.append(sorted(finite))
            self.seen = self.seen[-self.window:]
        else:
            self.run, self.seen = 0, []
        covered = {g for groups in self.seen for g in groups}
        return self.run >= self.window and self.required <= covered

    def state_dict(self) -> dict[str, Any]:
        return {"run": self.run, "seen": self.seen}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.run = int(state.get("run", 0))
        self.seen = [list(s) for s in state.get("seen", [])]


def training_plan(cfg: dict[str, Any], updates: int | None, stage: str,
                  smoke: bool) -> list[tuple[str, int]]:
    """Ordered ``(stage, n_updates)`` for this run.

    ``anneal`` mode is one stage of ``updates``. ``staged`` mode is pretrain
    then finetune (stage ``all``), or just one of them; ``updates``, when given,
    caps the total, filling pretrain first.
    """
    c = cfg["training"]["curriculum"]
    if str(c.get("mode", "anneal")) != "staged":
        n = updates or (3 if smoke else int(cfg["training"]["total_updates"]))
        return [("anneal", n)]
    p = 2 if smoke else int(c["pretrain"]["updates"])
    f = 1 if smoke else int(c["finetune"]["max_updates"])
    plan = {"all": [("pretrain", p), ("finetune", f)], "pretrain": [("pretrain", p)],
            "finetune": [("finetune", f)]}[stage]
    if updates is not None:
        capped, left = [], int(updates)
        for name, n in plan:
            take = min(n, left)
            capped.append((name, take))
            left -= take
        plan = capped
    return plan


def make_rollout_pool(cfg: dict[str, Any], cfgs: dict[str, Any],
                      normaliser_stats: dict[str, Any] | None, workers: int):
    """A persistent worker pool, or ``None`` for serial collection."""
    if workers <= 1:
        return None
    from concurrent.futures import ProcessPoolExecutor

    return ProcessPoolExecutor(max_workers=workers, initializer=_init_rollout_worker,
                               initargs=(cfg, cfgs, normaliser_stats))


def collect_rollouts(
    specs: Sequence[EpisodeSpec], episode_seeds: Sequence[int], net: Any,
    policy: AiHarpPolicy, objective: Any, cfgs: dict[str, Any], pool: Any = None,
) -> list[tuple[list, dict[str, float]]]:
    """Every episode of one update, in spec order; identical with or without ``pool``.

    Workers receive the current weights and the multipliers as they stand
    before this update's dual step, so every episode is priced exactly as in
    the serial loop.

    Serial collection also runs with ONE torch thread (restored afterwards).
    With 6 threads the recorded log-probs differed from a worker's by up to
    1.7e-6 -- float32 kernels are not bit-identical across thread counts --
    while actions, rewards and coverage matched; single-threaded they match
    exactly. It is also faster for these 4-node graphs (2.47 vs 3.04 ms per
    decision).
    """
    if pool is None:
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            return [run_seeded_episode(policy, s, e, objective, cfgs)
                    for s, e in zip(specs, episode_seeds)]
        finally:
            torch.set_num_threads(threads)
    state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    return list(pool.map(_rollout_task,
                         [(state, objective, s, e) for s, e in zip(specs, episode_seeds)]))


def compute_gae(
    transitions: Sequence[Any], gamma: float, lam: float
) -> tuple[np.ndarray, np.ndarray]:
    """GAE along each vehicle's own decision sequence.

    Transitions are grouped by vehicle and ordered by step before the recursion
    runs. Treating the flat list as one trajectory would bootstrap the value of
    one vehicle's decision from an unrelated vehicle's next state.
    """
    n = len(transitions)
    adv = np.zeros(n, dtype=float)
    ret = np.zeros(n, dtype=float)

    by_vehicle: dict[int, list[int]] = {}
    for i, t in enumerate(transitions):
        by_vehicle.setdefault(t.vehicle, []).append(i)

    for idxs in by_vehicle.values():
        idxs.sort(key=lambda i: transitions[i].step)
        last_adv = 0.0
        for k in reversed(range(len(idxs))):
            i = idxs[k]
            v = transitions[i].value
            next_v = transitions[idxs[k + 1]].value if k + 1 < len(idxs) else 0.0
            delta = transitions[i].reward + gamma * next_v - v
            last_adv = delta + gamma * lam * last_adv
            adv[i] = last_adv
            ret[i] = last_adv + v
    return adv, ret


def ppo_update(
    net: ActorCritic,
    optimiser: torch.optim.Optimizer,
    transitions: Sequence[Any],
    adv: np.ndarray,
    ret: np.ndarray,
    cfg: dict[str, Any],
) -> dict[str, float]:
    from torch_geometric.data import Batch

    p = cfg["algorithm"]["ppo"]
    clip, vcoef, ecoef = float(p["clip_ratio"]), float(p["value_coef"]), float(p["entropy_coef"])
    epochs, mb = int(p["epochs_per_update"]), int(p["minibatch_size"])
    max_norm = float(p["max_grad_norm"])

    # The learner may live on a GPU (training.device); graphs are built on the
    # CPU and each minibatch is moved over. On CPU every .to() is a no-op and
    # the step is exactly the CPU step it always was.
    dev = next(net.parameters()).device
    graphs = [t.graph.to_pyg() for t in transitions]
    actions = torch.tensor([t.action for t in transitions], dtype=torch.long)
    # Each decision's available actions, re-applied so new and old log-probs
    # come from the same (masked) distribution.
    n_act = int(net.n_actions)
    masks = torch.tensor([t.action_mask if getattr(t, "action_mask", None) is not None
                          else (True,) * n_act for t in transitions], dtype=torch.bool)
    old_lp = torch.tensor([t.log_prob for t in transitions], dtype=torch.float32)
    adv_t = torch.tensor(adv, dtype=torch.float32)
    ret_t = torch.tensor(ret, dtype=torch.float32)
    # Normalised advantages: with a near-bandit reward the raw scale varies a
    # lot between cells, and unnormalised advantages let dense cells dominate.
    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
    actions, masks, old_lp, adv_t, ret_t = (x.to(dev) for x in (actions, masks, old_lp, adv_t, ret_t))

    n = len(transitions)
    stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "clip_frac": 0.0}
    n_batches = 0

    for _ in range(epochs):
        order = torch.randperm(n)
        for start in range(0, n, mb):
            sel = order[start:start + mb]
            if sel.numel() < 2:
                continue
            batch = Batch.from_data_list([graphs[i] for i in sel.tolist()]).to(dev)
            sel = sel.to(dev)
            lp, value, entropy = net.evaluate_actions(batch, actions[sel], masks[sel])

            ratio = torch.exp(lp - old_lp[sel])
            a = adv_t[sel]
            unclipped = ratio * a
            clipped = torch.clamp(ratio, 1 - clip, 1 + clip) * a
            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = nn.functional.mse_loss(value, ret_t[sel])
            loss = policy_loss + vcoef * value_loss - ecoef * entropy.mean()

            optimiser.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), max_norm)
            optimiser.step()

            stats["policy_loss"] += float(policy_loss)
            stats["value_loss"] += float(value_loss)
            stats["entropy"] += float(entropy.mean())
            stats["clip_frac"] += float(((ratio - 1).abs() > clip).float().mean())
            n_batches += 1

    for k in stats:
        stats[k] /= max(n_batches, 1)
    return stats


def fit_normaliser(
    cfg: dict[str, Any],
    cfgs: dict[str, Any],
    n_graphs: int | None = None,
    path: Path | None = None,
    mode: str = "full",
) -> FeatureNormaliser:
    """Fit frozen feature statistics on a REPRESENTATIVE held-out sample.

    Held-out seeds, so the statistics never see a training episode. Frozen
    afterwards: a deployed OBU normalises with baked-in constants.

    Representative is the operative word. The first version looped
    ``for seed: for scenario:``, broke out once it had enough graphs, and kept
    the FIRST ``n_graphs`` -- so all 120 came from the opening ~0.3 s of one
    rural run. Measured consequences in the frozen stats: ``rel_vy`` and
    ``heading_sin`` std exactly 0 (an east-west corridor), ``message_age_s``
    std 0.07 s against a 60 s TTL, ``neighbour_count`` std 2.9. Every urban
    episode then fed the encoder inputs scaled by up to ~1e6.

    So this samples every training scenario, across the density range, over
    several hold-out seeds, and strides uniformly through each run instead of
    taking its opening decisions.
    """
    from agents.registry import build_policy

    norm_cfg = cfg["graph"]["normalisation"]
    scenarios = list(cfg["training"]["train_scenarios"])
    densities = [float(d) for d in norm_cfg.get("fit_densities", [2, 5, 20, 80])]
    seeds = list(norm_cfg["holdout_seeds"])[: int(norm_cfg.get("fit_seeds", 2))]
    n_graphs = int(n_graphs or norm_cfg.get("fit_graphs", 2000))
    gcfg = GraphConfig.from_config(cfg)

    cells = [(sc, d, s) for sc in scenarios for d in densities for s in seeds]
    per_cell = max(1, n_graphs // len(cells))
    graphs: list = []
    for scenario, density, seed in cells:
        scenario_cfg = load_scenario(scenario)
        trace = get_trace(scenario_cfg, density, seed, phy_cfg=cfgs["phy"])
        hazard = hazard_from_config(cfgs["hazard"], trace.meta)
        phy = build_phy(cfgs["phy"], scenario_cfg["name"], "clear", seed,
                        trace_meta=trace.meta)
        mac = build_mac(cfgs["phy"], phy)
        risk = build_risk_field(cfgs["hazard"], trace, hazard)

        captured: list = []
        probe = build_policy("weighted_p")
        original = probe.decide

        def decide(ctx, _orig=original, _cap=captured, _phy=phy):
            _cap.append(build_decision_graph(ctx, gcfg, _phy))
            return _orig(ctx)

        probe.decide = decide  # type: ignore[method-assign]
        DisseminationEngine(trace, phy, mac, risk, hazard, probe,
                            SeedBundle(master_seed=seed),
                            SimSettings.from_config(cfgs["experiment"])).run()
        if captured:
            # Stride through the whole run: early decisions all share a young
            # message, a low hop count and the originator's neighbourhood.
            take = np.unique(np.linspace(0, len(captured) - 1,
                                         min(per_cell, len(captured))).astype(int))
            graphs.extend(captured[i] for i in take)

    if not graphs:
        raise RuntimeError("collected no graphs for normalisation")
    norm = FeatureNormaliser.fit(graphs, provenance={
        "mode": mode, "scenarios": scenarios, "densities": densities,
        "seeds": seeds, "graphs_per_cell": per_cell, "n_cells": len(cells),
    })
    norm.save(path or PROJECT_ROOT / norm_cfg["stats_path"])
    return norm


def train(cfg: dict[str, Any], cfgs: dict[str, Any], updates: int | None,
          out_dir: Path, smoke: bool = False, stage: str = "all",
          resume: Path | None = None, init_from: Path | None = None) -> dict[str, Any]:
    """Train under the curriculum's plan (see :func:`training_plan`).

    ``resume`` continues a run in ``out_dir`` from its checkpoint exactly: model,
    optimiser, multipliers, both RNG streams, stage position and early-stop
    state are restored. ``init_from`` starts a NEW run (e.g. ``--stage
    finetune``) from another run's checkpoint -- the stage boundary -- so stage 2
    can be re-run with different weightings without repeating stage 1.
    """
    cuda_deterministic = bool(cfg["training"].get("cuda_deterministic", True))
    set_global_determinism(int(cfg["training"]["seed"]), cuda_deterministic=cuda_deterministic)
    rng = np.random.default_rng(int(cfg["training"]["seed"]))

    net = build_network(cfg)
    from agents.constrained_reward import (
        build_training_objective, curriculum_all_densities, pooled_shortfall,
    )

    # Constrained objective (agents/constrained_reward.py). The weighted-sum
    # reward ranked silence above every working scheme; the builder refuses it.
    # Full runs refuse to start on fallback targets; smoke runs may use them.
    objective = build_training_objective(cfg, PROJECT_ROOT, require_targets=not smoke)
    logger.info(
        "network: %s | %d parameters | objective: constrained, lambda_init=%.2f "
        "lr=%.2f max=%.1f, one multiplier per %s on pooled shortfall, "
        "target = %.0f%% of cell ceiling",
        cfg["encoder"]["type"], count_parameters(net), objective.multipliers.init,
        objective.multipliers.lr, objective.multipliers.max_value,
        " x ".join(objective.multipliers.group_by) or "run (global)",
        100 * objective.objective.target_fraction,
    )

    loaded = None
    if resume is not None or init_from is not None:
        src = Path(resume or init_from)
        loaded = torch.load(str(src), map_location="cpu", weights_only=False)
        net.load_state_dict(loaded["model"])
        objective.load_state_dict(loaded["objective"])
        normaliser_stats = loaded["normaliser_stats"]
        norm = FeatureNormaliser.from_dict(normaliser_stats)
        logger.info("%s from %s (update %d, stage %s)", "resuming" if resume else "initialised",
                    src, int(loaded["update"]), loaded.get("stage", "?"))
    else:
        # Smoke runs fit and keep their own statistics. A smoke fit used to be
        # saved to the canonical path and silently reused by the next full run.
        mode = "smoke" if smoke else "full"
        canonical = PROJECT_ROOT / cfg["graph"]["normalisation"]["stats_path"]
        stats_path = (canonical.with_name(f"{canonical.stem}_smoke{canonical.suffix}")
                      if smoke else canonical)
        scenarios = cfg["training"]["train_scenarios"]
        norm = FeatureNormaliser.load(stats_path) if stats_path.exists() else None
        if norm is not None and not norm.matches(mode, scenarios):
            logger.warning(
                "Refitting feature statistics: %s was fitted on %s, not for a %s run "
                "over %s.", stats_path.name, norm.provenance or "unrecorded data",
                mode, sorted(scenarios),
            )
            norm = None
        if norm is None:
            fit_cfg = cfg
            if smoke:
                # Small but still multi-scenario: one density, one seed per scenario.
                nc = {**cfg["graph"]["normalisation"], "fit_densities": [20], "fit_seeds": 1}
                fit_cfg = {**cfg, "graph": {**cfg["graph"], "normalisation": nc}}
            norm = fit_normaliser(fit_cfg, cfgs, n_graphs=200 if smoke else None,
                                  path=stats_path, mode=mode)
        # The feature statistics travel INSIDE the checkpoint. Loading them from
        # results/feature_stats.json at evaluation time would silently pair a
        # checkpoint with whatever statistics were fitted most recently.
        normaliser_stats = json.loads(stats_path.read_text(encoding="utf-8"))

    # The confidence gate is DISABLED during training rollouts. Training must be
    # on-policy: the first 40-update run recorded fallback_rate = 1.0 at update
    # 1, because an untrained policy is near-uniform (entropy 2.15 against a
    # maximum of ln 9 = 2.20), so confidence ~0.02 < tau = 0.5 and every
    # decision was handed to weighted_p. PPO then credited weighted_p's
    # outcomes to actions the network sampled but never executed, the policy
    # could not sharpen, and the gate kept falling back -- a deadlock.
    # The gate is a deployment mechanism; its tau sweep is run at evaluation.
    # Built by _training_policy, exactly as in the workers: constructing it
    # directly here left serial training unbatched while workers batched.
    policy = _training_policy(net, cfg, norm)
    # Rollouts always run on the CPU copy `net` (serially or in workers); the
    # PPO step runs on `learner`, which is `net` itself on CPU or a device copy
    # synced back into `net` after every step. Measured: PPO step 13.87 s on
    # CPU vs 3.07 s on an RTX 4050 Laptop GPU (experiments/bench_ppo_device.py).
    device = resolve_device(cfg["training"].get("device", "cpu"))
    learner = net if device.type == "cpu" else __import__("copy").deepcopy(net).to(device)
    opt = torch.optim.Adam(learner.parameters(), lr=float(cfg["algorithm"]["ppo"]["lr"]))
    if loaded is not None:
        opt.load_state_dict(loaded["optimiser"])
        if loaded.get("rng_state") is not None:
            rng.bit_generator.state = loaded["rng_state"]
            torch.set_rng_state(loaded["torch_rng_state"])

    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(str(PROJECT_ROOT / cfg["training"]["tensorboard_dir"]))
    except Exception as exc:  # pragma: no cover
        logger.warning("TensorBoard unavailable (%s); logging to JSONL only", exc)

    check_seed_split(cfg, cfgs)
    out_dir.mkdir(parents=True, exist_ok=True)
    history_path = out_dir / "history.jsonl"
    plan = training_plan(cfg, updates, stage, smoke)
    history: list[dict[str, Any]] = []
    start_stage, start_stage_update, update = plan[0][0], 0, 0
    if resume is not None:
        update = int(loaded["update"])
        start_stage = str(loaded.get("stage", plan[0][0]))
        start_stage_update = int(loaded.get("stage_update", update))
        if history_path.exists():
            # By update number, not count: an update that skipped its PPO step
            # (too few transitions) wrote no history row.
            history = [r for r in (json.loads(ln) for ln in history_path.read_text(
                encoding="utf-8").splitlines() if ln.strip()) if int(r["update"]) <= update]
        history_path.write_text("".join(json.dumps(r) + "\n" for r in history), encoding="utf-8")
    else:
        history_path.write_text("", encoding="utf-8")    # one run per directory
        if init_from is not None:
            update = int(loaded["update"])               # global count continues
    (out_dir / "crash.json").unlink(missing_ok=True)
    n_eps = 2 if smoke else int(cfg["training"]["rollout_episodes_per_update"])
    ckpt_every = int(cfg["training"]["checkpoint_every_updates"])
    t0 = time.time()

    t = cfg["training"]
    finetune_groups = frozenset(
        objective.group_for(sc, d, w, h)
        for sc in t["train_scenarios"] for d in curriculum_all_densities(t)
        for w in t["train_weather"] for h in t["train_hazards"]
        if "finetune" in t["curriculum"] and d in stage_densities(cfg, "finetune")[0])
    early = ConstraintEarlyStop(
        window=int(t["curriculum"].get("early_stop", {}).get("consecutive_updates", 30)),
        required=finetune_groups)
    if resume is not None and loaded.get("early_stop"):
        early.load_state_dict(loaded["early_stop"])
    current = {"stage": start_stage, "stage_update": start_stage_update}

    def _checkpoint(tag: str, at_update: int) -> None:
        torch.save({"update": at_update, "model": net.state_dict(),
                    "optimiser": opt.state_dict(), "config": cfg,
                    "normaliser_stats": normaliser_stats,
                    "objective": objective.state_dict(),
                    "stage": current["stage"], "stage_update": current["stage_update"],
                    "early_stop": early.state_dict(),
                    "rng_state": rng.bit_generator.state,
                    "torch_rng_state": torch.get_rng_state()},
                   out_dir / f"ckpt_{tag}.pt")

    workers = 1 if smoke else resolve_workers(cfg["training"].get("rollout_workers", 1), n_eps)
    pool = make_rollout_pool(cfg, cfgs, normaliser_stats, workers)
    reproducible = device.type == "cpu" or cuda_deterministic
    logger.info("rollouts: %s | PPO device %s%s | plan %s",
                f"{workers} worker processes" if pool else "serial", device,
                "" if reproducible else " (NON-DETERMINISTIC: exploratory, not for results/)",
                plan)

    # Skip the stages a resumed run already finished.
    names = [name for name, _ in plan]
    if start_stage not in names:
        raise ValueError(f"checkpoint stage {start_stage!r} is not in this run's plan {plan}")
    schedule = []
    for name, n in plan[names.index(start_stage):]:
        first = start_stage_update + 1 if name == start_stage else 1
        schedule.extend((name, k, n) for k in range(first, n + 1))

    stopped = "completed"
    try:
        for stage_name, stage_update, stage_total in schedule:
            if stage_name != current["stage"]:
                _checkpoint(current["stage"], update)     # the stage boundary
            current["stage"], current["stage_update"] = stage_name, stage_update
            update += 1
            progress = stage_update / max(stage_total, 1)
            specs = sample_episode_specs(cfg, progress, rng, n_eps,
                                         stage=None if stage_name == "anneal" else stage_name)

            transitions: list = []
            infos: list[dict[str, float]] = []
            # One seed per episode, drawn in order from the training RNG, so a
            # run is identical whether its episodes run serially or in workers.
            episode_seeds = [int(rng.integers(2**31 - 1)) for _ in specs]
            for tr_, info in collect_rollouts(specs, episode_seeds, net, policy,
                                              objective, cfgs, pool):
                transitions.extend(tr_)
                infos.append(info)

            # Step the price once per update from ALL episodes, and BEFORE the
            # small-batch skip below. A near-silent policy produces too few
            # transitions to update on; skipping the dual step as well would
            # freeze lambda exactly where silence pays.
            lam_used_by = {g: objective.multipliers.value(g)
                           for g in sorted({i["lambda_group"] for i in infos})}
            lam_next_by = objective.update(
                [(i["lambda_group"], i["obj_coverage"], i["obj_target"]) for i in infos])
            shortfall_by = dict(objective.multipliers.last_shortfalls)
            lam_used = float(np.mean(list(lam_used_by.values())))
            lam_next = float(np.mean(list(lam_next_by.values())))

            from agents.ai_harp import executed_transitions

            n_recorded = len(transitions)
            transitions = executed_transitions(transitions)
            if len(transitions) < n_recorded:
                logger.warning(
                    "update %d: dropped %d/%d transitions the gate overrode; "
                    "training must be on-policy", update,
                    n_recorded - len(transitions), n_recorded,
                )

            if len(transitions) < 4:
                logger.warning("update %d: only %d transitions; skipping the PPO step "
                               "(lambda %.3f -> %.3f still applied)",
                               update, len(transitions), lam_used, lam_next)
                # Persist the lambda step and the RNG draws this update consumed,
                # or a resume from the previous checkpoint would replay it.
                _checkpoint("latest", update)
                continue

            adv, ret = compute_gae(transitions,
                                   float(cfg["algorithm"]["ppo"]["gamma"]),
                                   float(cfg["algorithm"]["ppo"]["gae_lambda"]))
            losses = ppo_update(learner, opt, transitions, adv, ret, cfg)
            if learner is not net:
                net.load_state_dict({k: v.detach().cpu() for k, v in learner.state_dict().items()})

            met_stop = stage_name == "finetune" and early.update(shortfall_by)
            rec = {
                "update": update,
                "stage": stage_name,
                "stage_update": stage_update,
                "early_stop_run": early.run if stage_name == "finetune" else 0,
                "elapsed_s": round(time.time() - t0, 1),
                "densities": sorted(set(s.density for s in specs)),
                "reward_mean": float(np.mean([i["reward_mean"] for i in infos])),
                "transmissions": float(np.mean([i["transmissions"] for i in infos])),
                "informed": float(np.mean([i["informed"] for i in infos])),
                "fallback_rate": float(np.nanmean([i["gate_fallback_rate"] for i in infos])),
                "confidence_mean": float(np.nanmean([i["gate_confidence_mean"] for i in infos])),
                "n_transitions": len(transitions),
                "carry_rate": float(np.mean([i["carry_rate"] for i in infos])),
                "decisions_per_informed": float(np.mean(
                    [i["decisions_per_informed"] for i in infos])),
                # lambda / lambda_next: mean over the groups sampled this update;
                # the per-group values are what the dual step actually used.
                "lambda": lam_used,
                "lambda_next": lam_next,
                "lambda_by_group": lam_used_by,
                "lambda_next_by_group": lam_next_by,
                "lambda_saturated": bool(objective.multipliers.saturated_groups),
                "lambda_saturated_groups": objective.multipliers.saturated_groups,
                "coverage": float(np.nanmean([i["obj_coverage"] for i in infos])),
                "target": float(np.nanmean([i["obj_target"] for i in infos])),
                # shortfall: run4's mean of per-episode ratios, kept for
                # comparison; shortfall_pooled / _by_group drive the update.
                "shortfall": float(np.nanmean([i["obj_shortfall"] for i in infos])),
                "shortfall_pooled": pooled_shortfall(
                    [i["obj_coverage"] for i in infos], [i["obj_target"] for i in infos]),
                "shortfall_by_group": shortfall_by,
                "cost_per_at_risk": float(np.nanmean(
                    [i["obj_cost_per_at_risk"] for i in infos])),
                **losses,
            }
            history.append(rec)
            # Append and flush EVERY update. History used to be written only
            # at the end, so the first 40-update run died leaving checkpoints/
            # empty and no record of how far it got or why it stopped.
            with history_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
            if writer:
                for k, v in rec.items():
                    if isinstance(v, (int, float)):
                        writer.add_scalar(f"train/{k}", v, update)
                writer.flush()
            if stage_update % max(1, stage_total // 10) == 0 or stage_update == 1:
                logger.info(
                    "update %4d (%s %d/%d) | cov %.3f tgt %.3f | tx=%.0f inf=%.0f "
                    "ent=%.3f carry=%.2f",
                    update, stage_name, stage_update, stage_total, rec["coverage"],
                    rec["target"], rec["transmissions"], rec["informed"], rec["entropy"],
                    rec["carry_rate"],
                )
            _checkpoint("latest", update)                  # ~0.5 MB, always resumable
            if update % ckpt_every == 0:
                _checkpoint(f"{update:06d}", update)
            if met_stop:
                stopped = "early_stop"
                logger.info("early stop at update %d: every group met its coverage target "
                            "for %d consecutive updates", update, early.window)
                break
        if schedule and stopped == "completed" and current["stage"] == "finetune":
            stopped = "max_updates"
            logger.warning(
                "finetune reached its update cap (%d total updates) without the coverage "
                "constraints holding for %d consecutive updates. That is a finding about "
                "the objective; the run stops here.", update, early.window)
        _checkpoint(current["stage"], update)
        _checkpoint("final", update)
    except BaseException as exc:
        # BaseException, not Exception: an interrupted or killed run must still
        # leave evidence of where it stopped and why.
        crash = {"crashed_at_update": update, "completed_updates": len(history),
                 "error": f"{type(exc).__name__}: {exc}",
                 "elapsed_s": round(time.time() - t0, 1)}
        (out_dir / "crash.json").write_text(json.dumps(crash, indent=1), encoding="utf-8")
        logger.error("training stopped at update %d: %s", update, crash["error"])
        raise
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
        if writer:
            writer.close()

    last = history[-1] if history else {}
    summary = {
        "stopped": stopped, "total_updates": update, "plan": plan,
        "final_stage": current["stage"], "final_stage_update": current["stage_update"],
        "early_stop_window": early.window, "early_stop_run": early.run,
        "elapsed_s_this_session": round(time.time() - t0, 1),
        "ppo_device": str(device),
        "bit_reproducible": reproducible,
        "resumed_from": str(resume) if resume else None,
        "initialised_from": str(init_from) if init_from else None,
        "last_shortfall_by_group": last.get("shortfall_by_group"),
        "last_lambda_by_group": last.get("lambda_next_by_group"),
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    logger.info("trained to update %d (%s) in %.1fs", update, stopped, time.time() - t0)
    return {"history": history, "network": net, "normaliser": norm, "summary": summary}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="AI-HARP PPO training")
    ap.add_argument("--updates", type=int, default=None)
    ap.add_argument("--smoke", action="store_true", help="a few updates, minutes")
    ap.add_argument("--algorithm", default=None, choices=["ppo", "dueling_dqn"])
    ap.add_argument("--encoder", default=None, choices=["gatv2", "gcn", "mlp"])
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--keep-awake", action="store_true",
                    help="hold off idle sleep while training (Windows); see common/keep_awake.py")
    ap.add_argument("--stage", default="all", choices=["all", "pretrain", "finetune"],
                    help="staged curriculum: which stage(s) to run")
    ap.add_argument("--resume", default=None,
                    help="continue the run that wrote this checkpoint (e.g. its ckpt_latest.pt)")
    ap.add_argument("--init-from", default=None,
                    help="start a new run from another run's checkpoint (e.g. ckpt_pretrain.pt)")
    ap.add_argument("--sparse-weight", type=float, default=None,
                    help="override curriculum.finetune.sparse_to_dense_weight")
    ap.add_argument("--sparse-only", action="store_true",
                    help="finetune on the sparse densities alone (no dense episodes)")
    ap.add_argument("--lambda-max", type=float, default=None,
                    help="override objective.lambda_max (the multiplier cap)")
    ap.add_argument("--lambda-init", type=float, default=None,
                    help="override objective.lambda_init: the starting price of every group "
                         "the checkpoint does not already hold")
    ap.add_argument("--workers", default=None, help="override training.rollout_workers")
    ap.add_argument("--device", default=None, help="override training.device (cpu/cuda/auto)")
    ap.add_argument("--nondeterministic-gpu", action="store_true",
                    help="fast non-deterministic CUDA kernels (2.4x faster PPO); the run is "
                         "not bit-reproducible and its numbers may not enter results/")
    args = ap.parse_args(argv)
    if args.quiet:
        logging.getLogger("aiharp").setLevel(logging.WARNING)

    cfg = load_yaml("agent.yaml")
    if args.resume:
        # A resumed run keeps the configuration it started with.
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        cfg = state["config"]
    if args.algorithm:
        cfg["algorithm"]["name"] = args.algorithm
    if args.encoder:
        cfg["encoder"]["type"] = args.encoder
    if args.tau is not None:
        cfg["confidence_gate"]["tau"] = args.tau
    if args.sparse_weight is not None:
        cfg["training"]["curriculum"]["finetune"]["sparse_to_dense_weight"] = args.sparse_weight
    if args.sparse_only:
        cfg["training"]["curriculum"]["finetune"]["dense_densities"] = []
    if args.lambda_max is not None:
        cfg["objective"]["lambda_max"] = args.lambda_max
    if args.lambda_init is not None:
        cfg["objective"]["lambda_init"] = args.lambda_init
    if args.workers is not None:
        cfg["training"]["rollout_workers"] = args.workers
    if args.device is not None:
        cfg["training"]["device"] = args.device
    if args.nondeterministic_gpu:
        cfg["training"]["cuda_deterministic"] = False

    cfgs = {"phy": load_yaml("phy.yaml"), "hazard": load_yaml("hazard.yaml"),
            "experiment": load_yaml("experiment.yaml")}
    updates = args.updates
    if args.resume and not args.out:
        out = Path(args.resume).parent
    else:
        out = Path(args.out) if args.out else PROJECT_ROOT / cfg["training"]["checkpoint_dir"]
    from common.keep_awake import keep_awake

    with keep_awake(args.keep_awake) as awake:
        if args.keep_awake and not awake:
            logger.warning("--keep-awake requested but unavailable on this platform; "
                           "idle sleep can still suspend training")
        elif awake:
            logger.info("idle sleep held off for the duration of training")
        train(cfg, cfgs, updates, out, smoke=args.smoke, stage=args.stage,
              resume=Path(args.resume) if args.resume else None,
              init_from=Path(args.init_from) if args.init_from else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
