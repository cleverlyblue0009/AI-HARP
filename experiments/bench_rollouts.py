"""Rollout scaling benchmark: serial vs worker pools, with an identity check.

::

    D:/aiharp-env/python.exe -m experiments.bench_rollouts --workers 1 4 8 11
    python -m experiments.bench_rollouts --workers 1 8 16 32 48 --episodes 48   # many-core box

Times one update's rollouts (``--episodes`` episodes, drawn by the training
sampler) at each worker count and checks every count yields exactly the same
transitions as serial collection.

**Ceiling.** One episode runs in one worker, so a pool larger than the number
of episodes per update (``training.rollout_episodes_per_update``, 8) cannot
speed up training -- the extra workers are idle. ``--episodes`` above that
measures throughput per episode (the right number when sizing a machine), but
raising episodes per update changes each PPO batch, which is an optimisation
change, not a free speed-up.
"""

from __future__ import annotations

import argparse
import logging
import time

import numpy as np
import torch

from agents.constrained_reward import build_training_objective
from agents.gat_drl import build_network
from agents.graph import FeatureNormaliser
from agents.train import (
    _training_policy, collect_rollouts, make_rollout_pool, sample_episode_specs,
)
from common.config import PROJECT_ROOT, load_yaml


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Benchmark rollout scaling")
    ap.add_argument("--workers", type=int, nargs="+", default=[1, 4, 8])
    ap.add_argument("--episodes", type=int, default=None,
                    help="episodes to time (default: training.rollout_episodes_per_update)")
    ap.add_argument("--checkpoint", default="checkpoints/run7/ckpt_000200.pt")
    ap.add_argument("--stage", default="pretrain", choices=["pretrain", "finetune"],
                    help="which curriculum stage's densities to sample")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)
    logging.getLogger("aiharp").setLevel(logging.WARNING)

    cfg = load_yaml("agent.yaml")
    cfgs = {"phy": load_yaml("phy.yaml"), "hazard": load_yaml("hazard.yaml"),
            "experiment": load_yaml("experiment.yaml")}
    per_update = int(cfg["training"]["rollout_episodes_per_update"])
    n_eps = args.episodes or per_update
    net = build_network(cfg)
    state = torch.load(str(PROJECT_ROOT / args.checkpoint), map_location="cpu", weights_only=False)
    net.load_state_dict(state["model"])
    stats = state["normaliser_stats"]
    objective = build_training_objective(cfg, PROJECT_ROOT, require_targets=True)
    rng = np.random.default_rng(args.seed)
    specs = sample_episode_specs(cfg, 0.0, rng, n_eps, stage=args.stage)
    seeds = [int(rng.integers(2**31 - 1)) for _ in specs]
    print(f"{n_eps} episodes ({args.stage} densities): "
          + ", ".join(f"{s.scenario.split('_')[0]} d={s.density:g}" for s in specs), flush=True)

    def signature(results):
        return [([t.action for t in tr], [t.log_prob for t in tr], info["obj_coverage"])
                for tr, info in results]

    ref_sig, serial_t = None, None
    for w in args.workers:
        if w <= 1:
            policy = _training_policy(net, cfg, FeatureNormaliser.from_dict(stats))
            t0 = time.perf_counter()
            res = collect_rollouts(specs, seeds, net, policy, objective, cfgs)
            dt, warm = time.perf_counter() - t0, 0.0
        else:
            pool = make_rollout_pool(cfg, cfgs, stats, w)
            try:
                t0 = time.perf_counter()
                collect_rollouts(specs[:w], seeds[:w], net, None, objective, cfgs, pool)
                warm = time.perf_counter() - t0
                t0 = time.perf_counter()
                res = collect_rollouts(specs, seeds, net, None, objective, cfgs, pool)
                dt = time.perf_counter() - t0
            finally:
                pool.shutdown(wait=True)
        n_tr = sum(len(tr) for tr, _ in res)
        sig = signature(res)
        if ref_sig is None:
            ref_sig, serial_t = sig, dt
        idle = max(0, w - per_update)
        print(f"workers={w:>3}: {dt:7.1f} s  speed-up {serial_t / dt:5.2f}x  "
              f"{n_tr / dt:7.0f} transitions/s  identical={sig == ref_sig}  "
              f"pool start {warm:5.1f} s"
              + (f"  [{idle} of {w} workers idle at {per_update} episodes/update]"
                 if idle else ""), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
