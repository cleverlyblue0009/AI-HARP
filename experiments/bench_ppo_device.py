"""Step 3 benchmark: is the PPO step faster on a GPU?

::

    # 1. collect one update's transitions once (CPU environment)
    D:/aiharp-env/python.exe -m experiments.bench_ppo_device --collect cache/bench_transitions.pt
    # 2. time the PPO step on each device (same transitions, same epochs/minibatch)
    D:/aiharp-env/python.exe -m experiments.bench_ppo_device --load cache/bench_transitions.pt --device cpu
    D:/aiharp-gpu/Scripts/python.exe -m experiments.bench_ppo_device --load cache/bench_transitions.pt --device cuda

Mirrors ``agents.train.ppo_update`` (clipped PPO, value and entropy terms,
``epochs_per_update`` x ``minibatch_size``) with tensors on ``--device``. The
decision whether to add a ``device`` config key is taken from this number
alone; ``ppo_update`` stays CPU-only unless the speed-up is >= 1.3x.
"""

from __future__ import annotations

import argparse
import logging
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Batch

from agents.gat_drl import build_network
from common.config import PROJECT_ROOT, load_yaml


def collect(path: Path, checkpoint: str, seed: int) -> None:
    from agents.ai_harp import executed_transitions
    from agents.constrained_reward import build_training_objective
    from agents.graph import FeatureNormaliser
    from agents.train import _training_policy, collect_rollouts, compute_gae, sample_episode_specs

    cfg = load_yaml("agent.yaml")
    cfgs = {"phy": load_yaml("phy.yaml"), "hazard": load_yaml("hazard.yaml"),
            "experiment": load_yaml("experiment.yaml")}
    state = torch.load(str(PROJECT_ROOT / checkpoint), map_location="cpu", weights_only=False)
    net = build_network(cfg)
    net.load_state_dict(state["model"])
    objective = build_training_objective(cfg, PROJECT_ROOT, require_targets=True)
    rng = np.random.default_rng(seed)
    specs = sample_episode_specs(cfg, 1.0, rng, int(cfg["training"]["rollout_episodes_per_update"]))
    seeds = [int(rng.integers(2**31 - 1)) for _ in specs]
    res = collect_rollouts(specs, seeds, net, _training_policy(
        net, cfg, FeatureNormaliser.from_dict(state["normaliser_stats"])), objective, cfgs)
    tr = executed_transitions([t for trs, _ in res for t in trs])
    adv, ret = compute_gae(tr, float(cfg["algorithm"]["ppo"]["gamma"]),
                           float(cfg["algorithm"]["ppo"]["gae_lambda"]))
    payload = {
        "graphs": [(t.graph.x, t.graph.edge_index, t.graph.edge_attr) for t in tr],
        "actions": [t.action for t in tr], "log_probs": [t.log_prob for t in tr],
        "masks": [t.action_mask for t in tr], "adv": adv, "ret": ret,
        "checkpoint": checkpoint,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
    print(f"saved {len(tr)} transitions -> {path}")


def bench(path: Path, device: str, repeats: int) -> None:
    from torch_geometric.data import Data

    logging.getLogger("aiharp").setLevel(logging.WARNING)
    cfg = load_yaml("agent.yaml")
    p = cfg["algorithm"]["ppo"]
    payload = pickle.loads(path.read_bytes())
    state = torch.load(str(PROJECT_ROOT / payload["checkpoint"]), map_location="cpu",
                       weights_only=False)
    dev = torch.device(device)
    graphs = [Data(x=torch.as_tensor(x, dtype=torch.float32),
                   edge_index=torch.as_tensor(ei, dtype=torch.long),
                   edge_attr=torch.as_tensor(ea, dtype=torch.float32))
              for x, ei, ea in payload["graphs"]]
    n = len(graphs)
    n_act = 9
    actions = torch.tensor(payload["actions"], dtype=torch.long)
    old_lp = torch.tensor(payload["log_probs"], dtype=torch.float32)
    masks = torch.tensor([m if m is not None else (True,) * n_act for m in payload["masks"]])
    adv = torch.tensor(payload["adv"], dtype=torch.float32)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    ret = torch.tensor(payload["ret"], dtype=torch.float32)
    mb, epochs = int(p["minibatch_size"]), int(p["epochs_per_update"])
    clip, vcoef, ecoef = float(p["clip_ratio"]), float(p["value_coef"]), float(p["entropy_coef"])

    times = []
    for r in range(repeats + 1):                          # first repeat = warm-up
        net = build_network(cfg)
        net.load_state_dict(state["model"])
        net.to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=float(p["lr"]))
        torch.manual_seed(0)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(epochs):
            order = torch.randperm(n)
            for start in range(0, n, mb):
                sel = order[start:start + mb]
                if sel.numel() < 2:
                    continue
                batch = Batch.from_data_list([graphs[i] for i in sel.tolist()]).to(dev)
                lp, value, entropy = net.evaluate_actions(batch, actions[sel].to(dev),
                                                          masks[sel].to(dev))
                ratio = torch.exp(lp - old_lp[sel].to(dev))
                a = adv[sel].to(dev)
                loss = (-torch.min(ratio * a, torch.clamp(ratio, 1 - clip, 1 + clip) * a).mean()
                        + vcoef * nn.functional.mse_loss(value, ret[sel].to(dev))
                        - ecoef * entropy.mean())
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), float(p["max_grad_norm"]))
                opt.step()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        if r:
            times.append(time.perf_counter() - t0)
    name = torch.cuda.get_device_name(0) if dev.type == "cuda" else f"cpu ({torch.get_num_threads()} threads)"
    print(f"device={device} [{name}] torch {torch.__version__}: PPO step on {n} transitions "
          f"({epochs} epochs x minibatch {mb}): {np.mean(times):.2f} s "
          f"(min {min(times):.2f}, {repeats} repeats)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="PPO step CPU vs GPU")
    ap.add_argument("--collect", default=None)
    ap.add_argument("--load", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--threads", type=int, default=None, help="torch CPU threads")
    ap.add_argument("--deterministic", action="store_true",
                    help="torch.use_deterministic_algorithms (bit-reproducible on CUDA)")
    ap.add_argument("--checkpoint", default="checkpoints/run7/ckpt_000200.pt")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args(argv)
    logging.getLogger("aiharp").setLevel(logging.WARNING)
    if args.threads:
        torch.set_num_threads(args.threads)
    if args.deterministic:
        import os

        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
    if args.collect:
        collect(PROJECT_ROOT / args.collect, args.checkpoint, args.seed)
    if args.load:
        bench(PROJECT_ROOT / args.load, args.device, args.repeats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
