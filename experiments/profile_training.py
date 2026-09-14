"""Profile where a training update's wall-clock goes.

::

    D:/aiharp-env/python.exe -m experiments.profile_training --updates 3 \\
        --out results/profile_baseline.txt

Runs ``--updates`` real updates (curriculum's final, mixed-density stage, the
run's own sampler and seeds) twice: once plainly for honest wall-clock, once
instrumented. It reports

* forward passes: count, total and mean time, mean graph size;
* decisions per engine epoch (the batch size batching could exploit);
* simulator time (engine + reward, excluding the policy);
* decision-graph construction, normalisation and ``to_pyg`` conversion;
* PPO update split into forward, backward and optimiser step;
* worker-pool round trip: pickled transition payload size and transfer time;
* the Amdahl bound on any inference-only speed-up.

Checkpoint weights are used when given (``--checkpoint``), so action mixes and
episode lengths look like a trained policy's rather than an untrained one's.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import logging
import pickle
import pstats
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

import agents.ai_harp as ah
from agents.constrained_reward import build_training_objective
from agents.gat_drl import build_network
from agents.graph import FeatureNormaliser
from agents.train import (
    _training_policy, collect_rollouts, compute_gae, make_rollout_pool, ppo_update,
    sample_episode_specs,
)
from common.config import PROJECT_ROOT, load_yaml
from common.logging_utils import get_logger

logger = get_logger("experiments.profile_training")


class _Timer:
    def __init__(self) -> None:
        self.t: dict[str, float] = defaultdict(float)
        self.n: Counter = Counter()

    def wrap(self, name: str, fn):
        def inner(*a, **k):
            t0 = time.perf_counter()
            try:
                return fn(*a, **k)
            finally:
                self.t[name] += time.perf_counter() - t0
                self.n[name] += 1
        return inner


def _setup(args):
    cfg = load_yaml("agent.yaml")
    cfgs = {"phy": load_yaml("phy.yaml"), "hazard": load_yaml("hazard.yaml"),
            "experiment": load_yaml("experiment.yaml")}
    net = build_network(cfg)
    stats = None
    if args.checkpoint:
        state = torch.load(str(PROJECT_ROOT / args.checkpoint), map_location="cpu",
                           weights_only=False)
        net.load_state_dict(state["model"])
        stats = state.get("normaliser_stats")
    if stats is None:
        stats_path = PROJECT_ROOT / cfg["graph"]["normalisation"]["stats_path"]
        import json
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
    objective = build_training_objective(cfg, PROJECT_ROOT, require_targets=True)
    return cfg, cfgs, net, stats, objective


def _plan(cfg, n_updates: int, seed: int):
    rng = np.random.default_rng(seed)
    n_eps = int(cfg["training"]["rollout_episodes_per_update"])
    plan = []
    for _ in range(n_updates):
        specs = sample_episode_specs(cfg, 1.0, rng, n_eps)     # mixed densities
        plan.append((specs, [int(rng.integers(2**31 - 1)) for _ in specs]))
    return plan


def _ppo(net, cfg, transitions, timer: _Timer | None):
    from agents.ai_harp import executed_transitions

    tr = executed_transitions(transitions)
    adv, ret = compute_gae(tr, float(cfg["algorithm"]["ppo"]["gamma"]),
                           float(cfg["algorithm"]["ppo"]["gae_lambda"]))
    opt = torch.optim.Adam(net.parameters(), lr=float(cfg["algorithm"]["ppo"]["lr"]))
    if timer is None:
        ppo_update(net, opt, tr, adv, ret, cfg)
        return
    # Split forward / backward / optimiser by wrapping the three calls.
    orig_eval = net.evaluate_actions
    net.evaluate_actions = timer.wrap("ppo_forward", orig_eval)
    orig_step = opt.step
    opt.step = timer.wrap("ppo_optimiser", orig_step)
    orig_backward = torch.Tensor.backward
    torch.Tensor.backward = timer.wrap("ppo_backward", orig_backward)
    try:
        t0 = time.perf_counter()
        ppo_update(net, opt, tr, adv, ret, cfg)
        timer.t["ppo_total"] += time.perf_counter() - t0
    finally:
        net.evaluate_actions = orig_eval
        opt.step = orig_step
        torch.Tensor.backward = orig_backward


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Profile training throughput")
    ap.add_argument("--updates", type=int, default=3)
    ap.add_argument("--checkpoint", default="checkpoints/run7/ckpt_000200.pt")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default="results/profile_baseline.txt")
    args = ap.parse_args(argv)
    logging.getLogger("aiharp").setLevel(logging.WARNING)

    cfg, cfgs, net, stats, objective = _setup(args)
    plan = _plan(cfg, args.updates, args.seed)
    lines: list[str] = []
    say = lambda s="": (lines.append(s), print(s, flush=True))  # noqa: E731

    say(f"# python -m experiments.profile_training --updates {args.updates} "
        f"--checkpoint {args.checkpoint} --workers {args.workers} --seed {args.seed}")
    import platform
    import subprocess
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                         cwd=PROJECT_ROOT, text=True).strip()
    except Exception:  # pragma: no cover
        commit = "unknown"
    say(f"# commit {commit} | {platform.processor()} | torch {torch.__version__} "
        f"(threads {torch.get_num_threads()}) | {time.strftime('%Y-%m-%d %H:%M')}")
    say(f"# episodes per update {len(plan[0][0])}; curriculum progress 1.0 (mixed densities)")
    for u, (specs, _) in enumerate(plan, 1):
        say(f"#   update {u}: " + ", ".join(f"{s.scenario.split('_')[0]} d={s.density:g}"
                                          for s in specs))
    say()

    # ---------------------------------------------------------------- 1. wall-clock
    say("== 1. Wall-clock, uninstrumented ==")
    norm = FeatureNormaliser.from_dict(stats)
    policy = _training_policy(net, cfg, norm)
    serial_roll, serial_ppo, n_tr = 0.0, 0.0, 0
    serial_results = []
    for specs, seeds in plan:
        t0 = time.perf_counter()
        res = collect_rollouts(specs, seeds, net, policy, objective, cfgs)
        serial_roll += time.perf_counter() - t0
        trs = [t for tr, _ in res for t in tr]
        n_tr += len(trs)
        serial_results.append(res)
        t0 = time.perf_counter()
        _ppo(net, cfg, trs, None)
        serial_ppo += time.perf_counter() - t0
    per_update = (serial_roll + serial_ppo) / len(plan)
    say(f"serial      : {per_update:7.1f} s/update  (rollouts {serial_roll / len(plan):6.1f} s, "
        f"PPO {serial_ppo / len(plan):6.1f} s) | {n_tr / len(plan):.0f} transitions/update | "
        f"{n_tr / (serial_roll + serial_ppo):.0f} transitions/s end to end")

    # Reload weights: the serial PPO steps above changed them.
    cfg, cfgs, net, stats, objective = _setup(args)
    pool = make_rollout_pool(cfg, cfgs, stats, args.workers)
    policy = _training_policy(net, cfg, FeatureNormaliser.from_dict(stats))
    try:
        t0 = time.perf_counter()
        collect_rollouts(plan[0][0][:args.workers], plan[0][1][:args.workers], net, None,
                         objective, cfgs, pool)
        warm = time.perf_counter() - t0
        par_roll, par_ppo = 0.0, 0.0
        for specs, seeds in plan:
            t0 = time.perf_counter()
            res = collect_rollouts(specs, seeds, net, None, objective, cfgs, pool)
            par_roll += time.perf_counter() - t0
            trs = [t for tr, _ in res for t in tr]
            t0 = time.perf_counter()
            _ppo(net, cfg, trs, None)
            par_ppo += time.perf_counter() - t0
    finally:
        pool.shutdown(wait=True)
    par_update = (par_roll + par_ppo) / len(plan)
    say(f"{args.workers} workers   : {par_update:7.1f} s/update  (rollouts {par_roll / len(plan):6.1f} s, "
        f"PPO {par_ppo / len(plan):6.1f} s) | {n_tr / (par_roll + par_ppo):.0f} transitions/s | "
        f"pool start + warm-up {warm:.1f} s once per run")
    say(f"2,000 updates: serial {2000 * per_update / 3600:.1f} h, "
        f"{args.workers} workers {2000 * par_update / 3600:.1f} h")
    say()

    # ------------------------------------------------------------- 2. instrumented
    say("== 2. Instrumented breakdown (serial, same episodes, run7 weights) ==")
    cfg, cfgs, net, stats, objective = _setup(args)
    norm = FeatureNormaliser.from_dict(stats)
    policy = _training_policy(net, cfg, norm)
    timer = _Timer()
    nodes: list[int] = []
    decisions_per_epoch: Counter = Counter()
    epoch_counts: dict[tuple[int, int], int] = defaultdict(int)

    orig_act = net.act
    orig_act_batch = getattr(net, "act_batch", None)
    batch_sizes: list[int] = []
    batch_nodes: list[int] = []

    def act_counting(data, *a, **k):
        nodes.append(int(data.x.shape[0]))
        batch_sizes.append(1)
        batch_nodes.append(int(data.x.shape[0]))
        return orig_act(data, *a, **k)

    def act_batch_counting(batch, *a, **k):
        sizes = (batch.ptr[1:] - batch.ptr[:-1]).tolist()
        nodes.extend(int(s) for s in sizes)
        batch_sizes.append(len(sizes))
        batch_nodes.append(int(batch.x.shape[0]))
        return orig_act_batch(batch, *a, **k)

    net.act = timer.wrap("forward_pass", act_counting)
    if orig_act_batch is not None:
        net.act_batch = timer.wrap("forward_pass", act_batch_counting)
    orig_build = ah.build_decision_graph
    ah.build_decision_graph = timer.wrap("graph_build", orig_build)
    orig_apply = FeatureNormaliser.apply
    FeatureNormaliser.apply = timer.wrap("normalise", orig_apply)
    from agents.graph import DecisionGraph
    orig_pyg = DecisionGraph.to_pyg
    DecisionGraph.to_pyg = timer.wrap("to_pyg", orig_pyg)
    orig_decide = ah.AiHarpPolicy.decide
    orig_decide_batch = getattr(ah.AiHarpPolicy, "decide_batch", None)
    episode_id = [0]
    inside_batch = [False]
    decide_t = {"t": 0.0}

    def decide_counting(self, ctx):
        if inside_batch[0]:                      # already timed by decide_batch
            return orig_decide(self, ctx)
        epoch_counts[(episode_id[0], int(ctx.step))] += 1
        t0 = time.perf_counter()
        try:
            return orig_decide(self, ctx)
        finally:
            decide_t["t"] += time.perf_counter() - t0

    def decide_batch_counting(self, ctxs):
        for c in ctxs:
            epoch_counts[(episode_id[0], int(c.step))] += 1
        inside_batch[0] = True
        t0 = time.perf_counter()
        try:
            return orig_decide_batch(self, ctxs)
        finally:
            inside_batch[0] = False
            decide_t["t"] += time.perf_counter() - t0

    ah.AiHarpPolicy.decide = decide_counting
    if orig_decide_batch is not None:
        ah.AiHarpPolicy.decide_batch = decide_batch_counting
    import sim.engine as eng
    orig_run = eng.DisseminationEngine.run
    eng.DisseminationEngine.run = timer.wrap("engine_run_total", orig_run)
    try:
        profiler = cProfile.Profile()
        ppo_timer = _Timer()
        total_roll = 0.0
        for specs, seeds in plan:
            for s, e in zip(specs, seeds):
                episode_id[0] += 1
                t0 = time.perf_counter()
                profiler.enable()
                collect_rollouts([s], [e], net, policy, objective, cfgs)
                profiler.disable()
                total_roll += time.perf_counter() - t0
        # One PPO step per update, on the transitions the uninstrumented serial
        # pass collected, under the autograd profiler.
        prof = None
        for res in serial_results:
            trs = [t for tr, _ in res for t in tr]
            with torch.autograd.profiler.profile() as prof:
                _ppo(net, cfg, trs, ppo_timer)
    finally:
        net.act = orig_act
        if orig_act_batch is not None:
            net.act_batch = orig_act_batch
        ah.build_decision_graph = orig_build
        FeatureNormaliser.apply = orig_apply
        DecisionGraph.to_pyg = orig_pyg
        ah.AiHarpPolicy.decide = orig_decide
        if orig_decide_batch is not None:
            ah.AiHarpPolicy.decide_batch = orig_decide_batch
        eng.DisseminationEngine.run = orig_run

    decide = decide_t["t"]
    engine = timer.t["engine_run_total"]
    fwd, gb, nm, pyg = (timer.t[k] for k in ("forward_pass", "graph_build", "normalise", "to_pyg"))
    policy_other = decide - fwd - gb - nm - pyg
    simulator = engine - decide
    reward_etc = total_roll - engine
    ppo_t = ppo_timer.t["ppo_total"]
    total = total_roll + ppo_t
    n_fwd = timer.n["forward_pass"]
    say(f"(instrumented total {total:.1f} s over {len(plan)} updates; instrumentation adds "
        f"overhead, so use shares, not absolute times)")
    rows = [
        ("forward passes (inference)", fwd),
        ("decision-graph construction", gb),
        ("feature normalisation", nm),
        ("to_pyg conversion", pyg),
        ("other policy logic (mask, gate, mapping)", policy_other),
        ("simulator (engine, excl. policy)", simulator),
        ("reward, trace/hazard setup (run_episode excl. engine)", reward_etc),
        ("PPO forward (evaluate_actions)", ppo_timer.t["ppo_forward"]),
        ("PPO backward", ppo_timer.t["ppo_backward"]),
        ("PPO optimiser step", ppo_timer.t["ppo_optimiser"]),
        ("PPO other (batching graphs, GAE tensors, loss)",
         ppo_t - ppo_timer.t["ppo_forward"] - ppo_timer.t["ppo_backward"]
         - ppo_timer.t["ppo_optimiser"]),
    ]
    for name, t in rows:
        say(f"  {name:<52} {t:8.1f} s  {100 * t / total:5.1f}%")
    say(f"forward passes: {n_fwd} calls for {len(nodes)} decisions "
        f"({1e3 * fwd / max(n_fwd, 1):.2f} ms/call, {1e3 * fwd / max(len(nodes), 1):.2f} ms/decision), "
        f"graph size mean {np.mean(nodes):.1f} nodes (median {np.median(nodes):.0f}, "
        f"p95 {np.percentile(nodes, 95):.0f}, max {max(nodes)})")
    bs, bn = np.array(batch_sizes), np.array(batch_nodes)
    say(f"graphs per forward call: mean {bs.mean():.1f} (median {np.median(bs):.0f}, "
        f"p90 {np.percentile(bs, 90):.0f}, max {bs.max()}); nodes per forward call: mean "
        f"{bn.mean():.0f} (median {np.median(bn):.0f}, p90 {np.percentile(bn, 90):.0f}); "
        f"decisions evaluated in calls of >= 256 nodes: "
        f"{bs[bn >= 256].sum() / max(bs.sum(), 1):.1%}")
    per_epoch = np.array(list(epoch_counts.values()))
    say(f"decisions per engine epoch (epochs with >=1 decision): mean {per_epoch.mean():.2f}, "
        f"median {np.median(per_epoch):.0f}, p90 {np.percentile(per_epoch, 90):.0f}, "
        f"max {per_epoch.max()} over {per_epoch.size} epochs; share of decisions in epochs "
        f"with >=8 decisions: {per_epoch[per_epoch >= 8].sum() / per_epoch.sum():.1%}")
    inference_frac = (fwd + pyg) / total
    say(f"Amdahl bound for an inference-only optimisation (forward + to_pyg = "
        f"{inference_frac:.1%}): 1 / (1 - {inference_frac:.3f}) = {1 / (1 - inference_frac):.2f}x")
    policy_frac = decide / total
    say(f"Amdahl bound if the whole policy path (graph build + normalise + forward) went to "
        f"zero ({policy_frac:.1%}): {1 / (1 - policy_frac):.2f}x")
    say()

    # ------------------------------------------------------------- 3. pool transfer
    say("== 3. Worker-pool transfer cost ==")
    res = serial_results[0]
    payload = pickle.dumps(res, protocol=pickle.HIGHEST_PROTOCOL)
    t0 = time.perf_counter()
    for _ in range(3):
        pickle.loads(pickle.dumps(res, protocol=pickle.HIGHEST_PROTOCOL))
    rt = (time.perf_counter() - t0) / 3
    state = {k: v.detach().clone() for k, v in net.state_dict().items()}
    sd = pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)
    say(f"one update's returned transitions: {len(payload) / 1e6:.1f} MB pickled, "
        f"serialise + deserialise {rt:.2f} s ({100 * rt / max(par_update, 1e-9):.1f}% of a "
        f"{args.workers}-worker update)")
    say(f"weights sent per episode: {len(sd) / 1e6:.2f} MB")
    say()

    # ------------------------------------------------------------ 4. top functions
    say("== 4. cProfile, rollouts only, top 25 by cumulative time ==")
    s = io.StringIO()
    pstats.Stats(profiler, stream=s).sort_stats("cumulative").print_stats(25)
    for ln in s.getvalue().splitlines():
        if ln.strip():
            say(ln.rstrip())
    say()
    say("== 5. torch autograd profiler, last PPO update, top 15 ops by self CPU time ==")
    for ln in prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=15).splitlines():
        say(ln)

    out = PROJECT_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
