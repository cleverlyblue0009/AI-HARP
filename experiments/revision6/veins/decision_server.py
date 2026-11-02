"""Revision 6 Step 2: decision server for the learned policy in Veins (pinned interpreter, unchanged agent code).

    python -m experiments.revision6.veins.decision_server --connect 127.0.0.1:5801 --config <worker cfg.json>

One server per Veins worker; it dials that worker's WSL broker (bridge_broker.py). Each connection is one replay.
Before launching a replay the orchestrator writes the worker's config file = {run, bias, tau, scenario, density, seed,
n_vehicles, log, name}; at the replay's first request the server builds the policy exactly as the custom evaluation does (registry "ai_harp" -> from_checkpoint
with tau, suppression_bias, deterministic=False; policy.phy = the run's PHY, experiments/run_sim.py) and resets it
for the replay. Then, per newline-delimited JSON request (AiharpApp::bridgeDecide), it rebuilds the DecisionContext
(agents/base.py) from the app's own state and beacon-derived neighbour table, calls policy.decide unchanged, and
replies "act delay cancel relay1 relay2" (act: 0 suppress, 1 broadcast, 2 defer, 3 carry, 4 relay; -1 = none).
Randomness: per request, ctx.rng and the network's sampling uniform come from streams named by
(replay seed, vehicle, epoch, k), k = the vehicle's decision count within that epoch. Every request and reply is
appended to the replay's log (JSONL) for the offline trace check (validate_policy.py).
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
from pathlib import Path
from typing import Any

import numpy as np

from agents.base import ActionType, DecisionContext, Trigger

ACT = {ActionType.SUPPRESS: 0, ActionType.BROADCAST: 1, ActionType.DEFER: 2, ActionType.CARRY: 3, ActionType.RELAY: 4}
TRIG = {"receive": Trigger.RECEIVE, "timer": Trigger.TIMER, "originate": Trigger.ORIGINATE}
CARRY_RECHECK_STEPS = 10           # sim/engine.py SimSettings.carry_recheck_steps (default; not overridden in experiment.yaml)
STREAM_SALT = 0x5EED6              # names the revision-6 decision streams


def stream_key(seed: int, vehicle: int, step: int, k: int) -> list[int]:
    return [int(seed), int(vehicle), int(step), int(k), STREAM_SALT]


def ctx_from_request(q: dict, rng: np.random.Generator) -> DecisionContext:
    nb = np.asarray(q["nb"], dtype=np.int64)
    size = int(max(nb.max(initial=-1), q["index"]) + 1)
    eta_all = np.full(size, np.inf)
    rel_all = np.zeros(size)
    eta_all[nb] = np.asarray(q["nb_eta"], dtype=float)
    rel_all[nb] = np.asarray(q["nb_rel"], dtype=float)
    eta_all[q["index"]], rel_all[q["index"]] = q["eta_s"], q["relevance"]
    return DecisionContext(
        step=int(q["step"]), time_s=float(q["time_s"]), dt=float(q["dt"]), index=int(q["index"]),
        trigger=TRIG[q["trigger"]], hop_count=int(q["hop_count"]), duplicate_count=int(q["duplicate_count"]),
        own_tx_count=int(q["own_tx_count"]),
        sender_index=None if int(q["sender_index"]) < 0 else int(q["sender_index"]),
        sender_distance_m=float(q["sender_distance_m"]), max_sender_distance_m=float(q["max_sender_distance_m"]),
        age_s=float(q["age_s"]), x=float(q["x"]), y=float(q["y"]), speed_ms=float(q["speed_ms"]),
        heading=float(q["heading"]), direction=int(q["direction"]), relevance=float(q["relevance"]),
        eta_s=float(q["eta_s"]), vclass=str(q["vclass"]),
        # dtypes as sim/engine.py provides them (trace arrays are float32; agents/graph.py then computes in float32)
        neighbours=nb, neighbour_distances=np.asarray(q["nb_dist"], dtype=np.float32),
        neighbour_dx=np.asarray(q["nb_dx"], dtype=np.float32), neighbour_dy=np.asarray(q["nb_dy"], dtype=np.float32),
        neighbour_vx=np.asarray(q["nb_vx"], dtype=np.float32), neighbour_vy=np.asarray(q["nb_vy"], dtype=np.float32),
        neighbour_relevance=np.asarray(q["nb_rel"], dtype=float),
        neighbour_informed=np.asarray(q["nb_informed"], dtype=bool),
        comm_range_m=float(q["comm_range_m"]), rng=rng,
        sender_dx=float(q["sender_dx"]), sender_dy=float(q["sender_dy"]), was_designated=bool(q["was_designated"]),
        extras={"eta_all": eta_all, "relevance_all": rel_all})


def build_policy(cfg: dict) -> Any:
    """The policy exactly as experiments/run_sim.py builds it for an evaluation run of this (cell, seed)."""
    from agents.registry import build_policy as reg
    from common.config import load_yaml
    from experiments.revision5.sweep import checkpoint_for
    from experiments.evaluate_agent import agent_params
    from mobility.generate import get_trace, load_scenario
    from sim.phy import build_phy
    phy_cfg = load_yaml("phy.yaml")
    scfg = load_scenario(cfg["scenario"])
    trace = get_trace(scfg, float(cfg["density"]), int(cfg["seed"]), weather="clear", backend="sumo", phy_cfg=phy_cfg)
    ck = checkpoint_for(cfg["run"])
    pol = reg("ai_harp", **agent_params(ck, tau=float(cfg["tau"]), deterministic=False),
              suppression_bias=float(cfg["bias"]))
    pol.phy = build_phy(phy_cfg, scenario=scfg.get("phy_profile", scfg["name"]), weather="clear",
                        seed=int(cfg["seed"]), trace_meta=trace.meta)
    pol.reset(int(cfg["n_vehicles"]), np.random.default_rng([int(cfg["seed"]), STREAM_SALT, 1]))   # session stream
    return pol


class Session:
    """One replay: the policy instance (its per-vehicle state persists across requests) and the stream counters."""

    def __init__(self, cfg: dict):
        import torch
        self.torch = torch
        self.cfg = cfg
        self.policy = build_policy(cfg)
        self.k: dict[tuple[int, int], int] = {}

    def decide(self, q: dict) -> tuple[str, dict]:
        key = (int(q["index"]), int(q["step"]))
        k = self.k.get(key, 0)
        self.k[key] = k + 1
        sk = stream_key(self.cfg["seed"], key[0], key[1], k)
        rng = np.random.default_rng(sk)
        # the network's sampling uniform: AiHarpPolicy draws it from its private evaluation generator; seed that
        # generator from the same named stream (agent code unchanged)
        self.policy._eval_gen = self.torch.Generator().manual_seed(int(np.random.default_rng(sk + [1]).integers(2**62)))
        act = self.policy.decide(ctx_from_request(q, rng))
        delay = act.delay_steps if act.kind in (ActionType.DEFER, ActionType.CARRY) else 1
        if act.kind is ActionType.CARRY:
            delay = act.delay_steps or CARRY_RECHECK_STEPS
        cancel = -1 if act.cancel_on_duplicates is None else int(act.cancel_on_duplicates)
        relays = [int(r) for r in (act.relay_indices or ())][:2] + [-1, -1]
        reply = f"{ACT[act.kind]} {int(delay)} {cancel} {relays[0]} {relays[1]}"
        return reply, {"stream": sk, "kind": act.kind.value}


def session(conn: socket.socket, cfg_path: Path) -> None:
    """Serve one replay on one connection. The replay's configuration is read at its first request (the server may
    connect before the orchestrator has written it)."""
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    buf, n, sess, log, cfg = b"", 0, None, None, {}
    try:
        while True:
            chunk = conn.recv(65536)
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                if sess is None:
                    cfg = json.loads(cfg_path.read_text())
                    sess = Session(cfg)
                    log = open(cfg["log"], "a", encoding="utf-8")
                    log.write(json.dumps({"session": cfg}) + "\n")
                    print(f"[server] replay {cfg.get('name')}", flush=True)
                q = json.loads(line)
                reply, info = sess.decide(q)
                conn.sendall(reply.encode() + b"\n")
                log.write(json.dumps({"req": q, "reply": reply, **info}) + "\n")
                n += 1
    finally:
        if log:
            log.close()
        conn.close()
        if sess is not None:
            print(f"[server] replay {cfg.get('name')} done: {n} decisions", flush=True)


def serve(connect: str, cfg_path: Path) -> None:
    """Dial the WSL broker (bridge_broker.py) and serve one replay per connection, forever."""
    import time
    import numpy
    print(f"[env] {sys.executable} numpy {numpy.__version__}", flush=True)
    host, port = connect.rsplit(":", 1)
    while True:
        try:
            conn = socket.create_connection((host, int(port)))
        except OSError:
            time.sleep(1.0)
            continue
        session(conn, cfg_path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--connect", required=True, help="the worker's broker, e.g. 127.0.0.1:5801")
    ap.add_argument("--config", type=Path, required=True, help="replay configuration file written by the orchestrator")
    a = ap.parse_args(argv)
    serve(a.connect, a.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
