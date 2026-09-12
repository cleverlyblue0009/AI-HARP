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
from dataclasses import dataclass
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


def sample_episode_specs(
    cfg: dict[str, Any], progress: float, rng: np.random.Generator, n: int
) -> list[EpisodeSpec]:
    t = cfg["training"]
    densities = curriculum_densities(cfg, progress)
    out = []
    for _ in range(n):
        out.append(EpisodeSpec(
            scenario=str(rng.choice(t["train_scenarios"])),
            density=float(rng.choice(densities)),
            weather=str(rng.choice(t["train_weather"])),
            hazard_type=str(rng.choice(t["train_hazards"])),
            seed=int(rng.integers(0, 1_000_000)),
        ))
    return out


def run_episode(
    spec: EpisodeSpec,
    policy: AiHarpPolicy,
    weights: RewardWeights,
    cfgs: dict[str, Any],
) -> tuple[list, dict[str, float]]:
    """One episode; returns its transitions with rewards attached."""
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

    rewards = compute_rewards(result, risk, hazard, weights)
    for tr_ in policy.transitions:
        tr_.reward = float(rewards.get(tr_.vehicle, 0.0))

    informed = int((result.informed_step >= 0).sum())
    info = {
        "transmissions": float(result.n_transmissions),
        "informed": float(informed),
        "reward_mean": float(np.mean([t.reward for t in policy.transitions]))
        if policy.transitions else 0.0,
        "n_decisions": float(len(policy.transitions)),
        **policy.stats(),
    }
    return list(policy.transitions), info


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

    graphs = [t.graph.to_pyg() for t in transitions]
    actions = torch.tensor([t.action for t in transitions], dtype=torch.long)
    old_lp = torch.tensor([t.log_prob for t in transitions], dtype=torch.float32)
    adv_t = torch.tensor(adv, dtype=torch.float32)
    ret_t = torch.tensor(ret, dtype=torch.float32)
    # Normalised advantages: with a near-bandit reward the raw scale varies a
    # lot between cells, and unnormalised advantages let dense cells dominate.
    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

    n = len(transitions)
    stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "clip_frac": 0.0}
    n_batches = 0

    for _ in range(epochs):
        order = torch.randperm(n)
        for start in range(0, n, mb):
            sel = order[start:start + mb]
            if sel.numel() < 2:
                continue
            batch = Batch.from_data_list([graphs[i] for i in sel.tolist()])
            lp, value, entropy = net.evaluate_actions(batch, actions[sel])

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


def fit_normaliser(cfg: dict[str, Any], cfgs: dict[str, Any], n_graphs: int = 400):
    """Fit frozen feature statistics on HELD-OUT traces.

    Held-out seeds, so the statistics never see a training episode. Frozen
    afterwards: a deployed OBU normalises with baked-in constants, and training
    must match that.
    """
    from agents.registry import build_policy

    seeds = cfg["graph"]["normalisation"]["holdout_seeds"]
    graphs: list = []
    collector = build_policy("weighted_p")

    class _Collect(type(collector)):  # type: ignore[misc]
        pass

    for seed in seeds:
        for scenario in cfg["training"]["train_scenarios"]:
            if len(graphs) >= n_graphs:
                break
            scenario_cfg = load_scenario(scenario)
            trace = get_trace(scenario_cfg, 20, seed, phy_cfg=cfgs["phy"])
            hazard = hazard_from_config(cfgs["hazard"], trace.meta)
            phy = build_phy(cfgs["phy"], scenario_cfg["name"], "clear", seed,
                            trace_meta=trace.meta)
            mac = build_mac(cfgs["phy"], phy)
            risk = build_risk_field(cfgs["hazard"], trace, hazard)

            captured: list = []
            probe = build_policy("weighted_p")
            original = probe.decide

            def decide(ctx, _orig=original, _cap=captured, _phy=phy):
                _cap.append(build_decision_graph(ctx, GraphConfig.from_config(cfg), _phy))
                return _orig(ctx)

            probe.decide = decide  # type: ignore[method-assign]
            DisseminationEngine(trace, phy, mac, risk, hazard, probe,
                                SeedBundle(master_seed=seed),
                                SimSettings.from_config(cfgs["experiment"])).run()
            graphs.extend(captured)

    if not graphs:
        raise RuntimeError("collected no graphs for normalisation")
    norm = FeatureNormaliser.fit(graphs[:n_graphs])
    path = PROJECT_ROOT / cfg["graph"]["normalisation"]["stats_path"]
    norm.save(path)
    return norm


def train(cfg: dict[str, Any], cfgs: dict[str, Any], updates: int,
          out_dir: Path, smoke: bool = False) -> dict[str, Any]:
    set_global_determinism(int(cfg["training"]["seed"]))
    rng = np.random.default_rng(int(cfg["training"]["seed"]))

    net = build_network(cfg)
    weights = RewardWeights.from_config(cfg)
    logger.info("network: %s | %d parameters | w2/w1 = %.2f",
                cfg["encoder"]["type"], count_parameters(net), weights.w2_over_w1)

    stats_path = PROJECT_ROOT / cfg["graph"]["normalisation"]["stats_path"]
    norm = (FeatureNormaliser.load(stats_path) if stats_path.exists()
            else fit_normaliser(cfg, cfgs, n_graphs=120 if smoke else 400))

    policy = AiHarpPolicy(
        network=net, gate=ConfidenceGate.from_config(cfg), normaliser=norm,
        graph_cfg=GraphConfig.from_config(cfg),
        fallback_policy=cfg["confidence_gate"]["fallback_policy"], record=True,
    )
    opt = torch.optim.Adam(net.parameters(), lr=float(cfg["algorithm"]["ppo"]["lr"]))

    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(str(PROJECT_ROOT / cfg["training"]["tensorboard_dir"]))
    except Exception as exc:  # pragma: no cover
        logger.warning("TensorBoard unavailable (%s); logging to JSONL only", exc)

    out_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    n_eps = 2 if smoke else int(cfg["training"]["rollout_episodes_per_update"])
    ckpt_every = int(cfg["training"]["checkpoint_every_updates"])
    t0 = time.time()

    for update in range(1, updates + 1):
        progress = update / max(updates, 1)
        specs = sample_episode_specs(cfg, progress, rng, n_eps)

        transitions: list = []
        infos: list[dict[str, float]] = []
        for spec in specs:
            policy.reset(0, rng)
            tr_, info = run_episode(spec, policy, weights, cfgs)
            transitions.extend(tr_)
            infos.append(info)

        if len(transitions) < 4:
            logger.warning("update %d: only %d transitions; skipping",
                           update, len(transitions))
            continue

        adv, ret = compute_gae(transitions,
                               float(cfg["algorithm"]["ppo"]["gamma"]),
                               float(cfg["algorithm"]["ppo"]["gae_lambda"]))
        losses = ppo_update(net, opt, transitions, adv, ret, cfg)

        rec = {
            "update": update,
            "densities": sorted(set(s.density for s in specs)),
            "reward_mean": float(np.mean([i["reward_mean"] for i in infos])),
            "transmissions": float(np.mean([i["transmissions"] for i in infos])),
            "informed": float(np.mean([i["informed"] for i in infos])),
            "fallback_rate": float(np.nanmean([i["gate_fallback_rate"] for i in infos])),
            "confidence_mean": float(np.nanmean([i["gate_confidence_mean"] for i in infos])),
            "n_transitions": len(transitions),
            **losses,
        }
        history.append(rec)
        if writer:
            for k, v in rec.items():
                if isinstance(v, (int, float)):
                    writer.add_scalar(f"train/{k}", v, update)
        if update % max(1, updates // 10) == 0 or update == 1:
            logger.info(
                "update %4d/%d | r=%+.3f tx=%.0f inf=%.0f fb=%.2f ent=%.3f pl=%+.4f",
                update, updates, rec["reward_mean"], rec["transmissions"],
                rec["informed"], rec["fallback_rate"], rec["entropy"], rec["policy_loss"],
            )
        if update % ckpt_every == 0 or update == updates:
            ckpt = out_dir / f"ckpt_{update:06d}.pt"
            torch.save({"update": update, "model": net.state_dict(),
                        "optimiser": opt.state_dict(), "config": cfg}, ckpt)

    (out_dir / "history.jsonl").write_text(
        "\n".join(json.dumps(h) for h in history), encoding="utf-8"
    )
    if writer:
        writer.close()
    logger.info("trained %d updates in %.1fs", updates, time.time() - t0)
    return {"history": history, "network": net, "normaliser": norm}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="AI-HARP PPO training")
    ap.add_argument("--updates", type=int, default=None)
    ap.add_argument("--smoke", action="store_true", help="a few updates, minutes")
    ap.add_argument("--algorithm", default=None, choices=["ppo", "dueling_dqn"])
    ap.add_argument("--encoder", default=None, choices=["gatv2", "gcn", "mlp"])
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    if args.quiet:
        logging.getLogger("aiharp").setLevel(logging.WARNING)

    cfg = load_yaml("agent.yaml")
    if args.algorithm:
        cfg["algorithm"]["name"] = args.algorithm
    if args.encoder:
        cfg["encoder"]["type"] = args.encoder
    if args.tau is not None:
        cfg["confidence_gate"]["tau"] = args.tau

    cfgs = {"phy": load_yaml("phy.yaml"), "hazard": load_yaml("hazard.yaml"),
            "experiment": load_yaml("experiment.yaml")}
    updates = args.updates or (3 if args.smoke else int(cfg["training"]["total_updates"]))
    out = Path(args.out) if args.out else PROJECT_ROOT / cfg["training"]["checkpoint_dir"]
    train(cfg, cfgs, updates, out, smoke=args.smoke)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
