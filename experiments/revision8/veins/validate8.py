"""Revision 8: bridge validation for STRUCT and Experiment C (results/revision8/ADDENDUM.md section 8), the same form
as revision 6 (experiments/revision6/veins/validate_policy.py).

    python -m experiments.revision8.veins.validate8 unit --run plain_s0      (1) >= 1,000 recorded contexts
    python -m experiments.revision8.veins.validate8 geometric                (2) Exp C table == custom engine table
    python -m experiments.revision8.veins.validate8 trace <log.jsonl[.gz]>   (3) offline re-send of a logged replay

(1) The custom simulator runs STRUCT(<run>, delta 0) on SUMO mobility, rural d=20 and urban NLOS d=20, seeds 200, 201,
...; every first-reception DecisionContext is recorded until >= 1,000. Each is serialised exactly as the app serialises
its request (revision-6 request_from_ctx), rebuilt by the server (ctx_from_request), and both decide from the same
named stream: the replies (action, delay, cancel threshold) must be identical, and so must the draws.
(2) For the same kind of contexts, the request is rewritten by decision_server8.Geometric (with "heard" = the
context's informed neighbours); the rewritten neighbour arrays must equal the engine's own, value for value.
(3) Every request logged by the server during a Veins replay is re-sent, in order, to a fresh Session8 with the
logged session configuration; every reply must be reproduced exactly.
Outputs: results/revision8/veins/port_validation_{unit_<run>,geometric,trace}.json.
"""

from __future__ import annotations

import dataclasses
import gzip
import json
import sys
from pathlib import Path

import numpy as np

from common.config import PROJECT_ROOT
from experiments.revision1.provenance import stamp, write_json
from experiments.revision6.veins import decision_server as ds6
from experiments.revision6.veins.validate_policy import request_from_ctx
from experiments.revision8.veins.decision_server8 import Geometric, Session8, reply_of

OUT = PROJECT_ROOT / "results" / "revision8" / "veins"
CELLS = (("rural_highway", 20.0), ("urban_nlos", 20.0))
N_MIN = 1000


def record(run: str, n_min: int = N_MIN, policy_theta=None) -> list[tuple]:
    """(cell, seed, context) for first receptions of STRUCT(run) on SUMO mobility until n_min are recorded."""
    import logging
    from agents import registry
    from agents.structured import StructuredPolicy
    from experiments.run_sim import RunSpec, run_single
    logging.disable(logging.WARNING)
    theta = policy_theta or json.loads((PROJECT_ROOT / "results" / "revision8" / "train" / run / "theta.json")
                                       .read_text())["theta"]
    rec: list[tuple] = []
    cur: dict = {}

    class _Rec(StructuredPolicy):
        def decide(self, ctx):
            if ctx.trigger.value == "receive":
                rec.append((cur["cell"], cur["seed"], dataclasses.replace(ctx, rng=None)))
            return super().decide(ctx)

    registry._REGISTRY["_struct_rec"] = _Rec
    try:
        seed = 200
        while len(rec) < n_min:
            for sc, d in CELLS:
                cur.update(cell=(sc, d), seed=seed)
                run_single(RunSpec(scenario=sc, density_veh_km_lane=d, weather="clear", policy="_struct_rec",
                                   policy_params={"theta": theta}, seed=seed, hazard_type="fog_bank", backend="sumo"))
            seed += 1
    finally:
        registry._REGISTRY.pop("_struct_rec", None)
    return rec


def unit(run: str) -> dict:
    from agents.structured import StructuredPolicy
    theta = json.loads((PROJECT_ROOT / "results" / "revision8" / "train" / run / "theta.json").read_text())["theta"]
    rec = record(run)
    ref = StructuredPolicy(theta=theta, delta=0.0)
    mism, draws, ex = 0, 0, []
    for (sc, d), seed, ctx in rec:
        sk = ds6.stream_key(seed, ctx.index, ctx.step, 0)
        g1, g2 = np.random.default_rng(sk), np.random.default_rng(sk)
        a = ref.decide(dataclasses.replace(ctx, rng=g1))
        b = ref.decide(ds6.ctx_from_request(json.loads(json.dumps(request_from_ctx(ctx, seed))), g2))
        if reply_of(a) != reply_of(b):
            mism += 1
            ex.append({"cell": [sc, d], "seed": seed, "index": ctx.index, "step": ctx.step,
                       "in_process": reply_of(a), "bridge": reply_of(b)})
        draws += g1.bit_generator.state != g2.bit_generator.state
    res = {"run": run, "contexts": len(rec), "decision_mismatches": mism, "draw_mismatches": int(draws),
           "examples": ex[:10], "pass": bool(len(rec) >= N_MIN and mism == 0 and draws == 0)}
    write_json(OUT / f"port_validation_unit_{run}.json", {"provenance": stamp("revision8/veins/validate8 unit",
                                                                              {"run": run}), **res})
    return res


def geometric() -> dict:
    from agents.structured import theta_slotted
    rec = record("-", policy_theta=theta_slotted(5).tolist())
    geos: dict = {}
    mism, ex = 0, []
    for (sc, d), seed, ctx in rec:
        g = geos.setdefault((sc, d, seed), Geometric(sc, d, seed))
        q = request_from_ctx(ctx, seed)
        q["heard"] = [int(v) for v in np.asarray(ctx.neighbours)[np.asarray(ctx.neighbour_informed, dtype=bool)]]
        r = g.rewrite({**q, "nb": [], "nb_dist": [], "nb_dx": [], "nb_dy": [], "nb_vx": [], "nb_vy": [],
                       "nb_rel": [], "nb_eta": [], "nb_informed": []})
        same = all(r[k] == q[k] for k in ("nb", "nb_dist", "nb_dx", "nb_dy", "nb_vx", "nb_vy", "nb_rel", "nb_eta",
                                          "nb_informed"))
        if not same:
            mism += 1
            if len(ex) < 5:
                ex.append({"cell": [sc, d], "seed": seed, "index": ctx.index, "step": ctx.step,
                           "n_engine": len(q["nb"]), "n_rewrite": len(r["nb"])})
    res = {"contexts": len(rec), "mismatches": mism, "examples": ex, "pass": bool(len(rec) >= N_MIN and mism == 0)}
    write_json(OUT / "port_validation_geometric.json", {"provenance": stamp("revision8/veins/validate8 geometric", {}),
                                                         **res})
    return res


def trace(path: Path) -> dict:
    op = gzip.open if path.suffix == ".gz" else open
    lines = [json.loads(l) for l in op(path, "rt", encoding="utf-8")]
    cfg = next(x["session"] for x in lines if "session" in x)
    s = Session8(cfg)
    n, mism = 0, 0
    for x in lines:
        if "req" not in x:
            continue
        reply, _ = s.decide(x["req"])
        n += 1
        mism += reply != x["reply"]
    res = {"log": path.name, "requests": n, "mismatches": mism, "pass": bool(n > 0 and mism == 0)}
    p = OUT / "port_validation_trace.json"
    prev = json.loads(p.read_text())["logs"] if p.exists() else []
    write_json(p, {"provenance": stamp("revision8/veins/validate8 trace", {}), "logs": prev + [res]})
    return res


def main(argv=None) -> int:
    a = list(sys.argv[1:] if argv is None else argv)
    print(f"[env] {sys.executable} numpy {np.__version__}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    if a[0] == "unit":
        r = unit(a[a.index("--run") + 1])
    elif a[0] == "geometric":
        r = geometric()
    else:
        r = trace(Path(a[1]))
    print(json.dumps({k: v for k, v in r.items() if k != "examples"}))
    return 0 if r["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
