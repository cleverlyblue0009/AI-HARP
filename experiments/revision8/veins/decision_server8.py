"""Revision 8: decision server for STRUCT and for Experiment C (results/revision8/ADDENDUM.md sections 6 and 8).

    python -m experiments.revision8.veins.decision_server8 --connect 127.0.0.1:5801 --config <worker cfg.json>

Extends the revision-6 server (experiments/revision6/veins/decision_server.py, reused unchanged: request format,
ctx_from_request, per-request streams named (replay seed, vehicle, epoch, k), JSONL request/reply log). Per-replay
configuration (written by queue8 before each replay):
  {"policy": "struct", "run": "plain_s0", "delta": 1.0, "scenario", "density", "seed", "n_vehicles", "log", "name"}
      STRUCT(theta of results/revision8/train/<run>/theta.json, delta) decides; the app never sends duplicates for it.
  {"policy": "ai_harp", "run": "ref_s0", "bias": 1.0, "tau": 0.5, ..., "neighbour_table": "geometric"}
      Experiment C: before the unchanged AI-HARP code decides, the request's neighbour arrays are replaced by the custom
      engine's geometric table at the request's step (sim/engine.py DisseminationEngine._context, including the NLOS
      link-budget test), from the SUMO trace; neighbour_informed = in the app's overheard set (request field "heard",
      sent by the app with sendHeard=true). Own state, trigger and message state are the app's.
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
from pathlib import Path

import numpy as np

from common.config import PROJECT_ROOT
from experiments.revision6.veins import decision_server as ds6

TRAIN = PROJECT_ROOT / "results" / "revision8" / "train"


def struct_policy(cfg: dict):
    from agents.structured import StructuredPolicy
    theta = json.loads((TRAIN / cfg["run"] / "theta.json").read_text())["theta"]
    pol = StructuredPolicy(theta=theta, delta=float(cfg["delta"]))
    pol.reset(int(cfg["n_vehicles"]), np.random.default_rng([int(cfg["seed"]), ds6.STREAM_SALT, 1]))
    return pol


class Geometric:
    """The custom engine's decision-time neighbour table for one (scenario, density, seed), SUMO mobility."""

    def __init__(self, scenario: str, density: float, seed: int):
        from agents.registry import build_policy
        from common.config import load_yaml
        from common.seeding import SeedBundle
        from hazard.model import hazard_from_config
        from hazard.risk_field import build_risk_field
        from mobility.generate import get_trace, load_scenario
        from sim.engine import DisseminationEngine, SimSettings
        from sim.mac import build_mac
        from sim.phy import build_phy
        phy_cfg, hz_cfg = load_yaml("phy.yaml"), load_yaml("hazard.yaml")
        sc = load_scenario(scenario)
        tr = get_trace(sc, float(density), int(seed), weather="clear", backend="sumo", phy_cfg=phy_cfg)
        hz = hazard_from_config(hz_cfg, tr.meta, overrides={"type": "fog_bank"})
        phy = build_phy(phy_cfg, scenario=sc.get("phy_profile", sc["name"]), weather="clear", seed=int(seed),
                        trace_meta=tr.meta)
        self.eng = DisseminationEngine(tr, phy, build_mac(phy_cfg, phy), build_risk_field(hz_cfg, tr, hz), hz,
                                       build_policy("flooding"), SeedBundle(master_seed=int(seed)),
                                       SimSettings.from_config(load_yaml("experiment.yaml")))
        self.eng._feats_cache = None
        self.N = tr.n_vehicles

    def rewrite(self, q: dict) -> dict:
        from agents.base import Trigger
        step, i = int(q["step"]), int(q["index"])
        if not self.eng.trace.active[step, i]:
            return q                                         # not in the trace's active set at this step: unchanged
        informed = np.full(self.N, -1, dtype=np.int32)
        heard = [int(h) for h in q.get("heard", []) if 0 <= int(h) < self.N]
        informed[heard] = 0
        z = np.zeros(self.N, dtype=np.int32)
        ctx = self.eng._context(i, step, Trigger.RECEIVE, z.astype(np.int16), z, np.zeros(self.N, np.float32), informed,
                                0, None, None, float(q["sender_distance_m"]))
        f = lambda a: [float(v) for v in np.asarray(a, dtype=float)]  # noqa: E731
        nb = [int(v) for v in ctx.neighbours]
        out = dict(q)
        out.update(nb=nb, nb_dist=f(ctx.neighbour_distances), nb_dx=f(ctx.neighbour_dx), nb_dy=f(ctx.neighbour_dy),
                   nb_vx=f(ctx.neighbour_vx), nb_vy=f(ctx.neighbour_vy), nb_rel=f(ctx.neighbour_relevance),
                   nb_eta=f(np.asarray(ctx.extras["eta_all"])[nb]) if nb else [],
                   nb_informed=[int(bool(v)) for v in ctx.neighbour_informed])
        return out


class Session8:
    """One replay. STRUCT is built here; AI-HARP through the unchanged revision-6 Session."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.kind = cfg.get("policy", "ai_harp")
        self.k: dict[tuple[int, int], int] = {}
        self.inner = None
        if self.kind == "struct":
            self.policy = struct_policy(cfg)
        else:
            self.inner = ds6.Session(cfg)
        self.geo = (Geometric(cfg["scenario"], cfg["density"], cfg["seed"])
                    if cfg.get("neighbour_table") == "geometric" else None)

    def decide(self, q: dict) -> tuple[str, dict]:
        if self.geo is not None:
            q = self.geo.rewrite(q)
        if self.inner is not None:
            reply, info = self.inner.decide(q)
            return reply, ({**info, "rewritten": True} if self.geo is not None else info)
        key = (int(q["index"]), int(q["step"]))
        k = self.k.get(key, 0)
        self.k[key] = k + 1
        sk = ds6.stream_key(self.cfg["seed"], key[0], key[1], k)
        act = self.policy.decide(ds6.ctx_from_request(q, np.random.default_rng(sk)))
        return reply_of(act), {"stream": sk, "kind": act.kind.value}


def reply_of(act) -> str:
    from agents.base import ActionType
    delay = act.delay_steps if act.kind in (ActionType.DEFER, ActionType.CARRY) else 1
    if act.kind is ActionType.CARRY:
        delay = act.delay_steps or ds6.CARRY_RECHECK_STEPS
    cancel = -1 if act.cancel_on_duplicates is None else int(act.cancel_on_duplicates)
    relays = [int(r) for r in (act.relay_indices or ())][:2] + [-1, -1]
    return f"{ds6.ACT[act.kind]} {int(delay)} {cancel} {relays[0]} {relays[1]}"


def session(conn: socket.socket, cfg_path: Path) -> None:
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
                    sess = Session8(cfg)
                    log = open(cfg["log"], "a", encoding="utf-8")
                    log.write(json.dumps({"session": cfg}) + "\n")
                    print(f"[server8] replay {cfg.get('name')}", flush=True)
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
            print(f"[server8] replay {cfg.get('name')} done: {n} decisions", flush=True)


def serve(connect: str, cfg_path: Path) -> None:
    import time
    print(f"[env] {sys.executable} numpy {np.__version__}", flush=True)
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
    ap.add_argument("--connect", required=True)
    ap.add_argument("--config", type=Path, required=True)
    a = ap.parse_args(argv)
    serve(a.connect, a.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
