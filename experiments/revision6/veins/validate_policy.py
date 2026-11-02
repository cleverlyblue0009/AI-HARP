"""Revision 6 Step 2: validate the decision-server bridge (results/revision6/veins/port_validation_policy.md).

    python -m experiments.revision6.veins.validate_policy unit             (1) 1,000 recorded contexts
    python -m experiments.revision6.veins.validate_policy trace <log.jsonl> (2) offline re-send of a logged replay

(1) The custom simulator runs ref_s0 at bias +1 (its revision-5 SEL bias in rural d=20), rural d=20, seed 200, SUMO
mobility, and every DecisionContext the policy receives is recorded (up to 1,000 network decisions). Each is
serialised exactly as the Veins app serialises its request (the same JSON keys) and rebuilt by the server's
ctx_from_request. For both, the unchanged policy code computes the action distribution (network probabilities after
the suppression bias, AiHarpPolicy._prepare -> network.act -> _apply_suppression_bias) and the gate verdict; they
must agree to 1e-6 (max abs over all actions and contexts) with identical gate verdicts.
(2) Every request logged by the server during a Veins replay is re-sent, in order, to a fresh session with the logged
session configuration; every reply must be reproduced exactly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from common.config import PROJECT_ROOT
from experiments.revision1.provenance import stamp
from experiments.revision6.veins.decision_server import Session, build_policy, ctx_from_request

OUT = PROJECT_ROOT / "results" / "revision6" / "veins"
CELL = {"scenario": "rural_highway", "density": 20.0, "seed": 200, "run": "ref_s0", "bias": 1.0, "tau": 0.5}


def request_from_ctx(c, seed: int) -> dict:
    """The app's request (AiharpApp::bridgeDecide keys) for a custom-simulator context."""
    eta_all = c.extras.get("eta_all")
    nb = np.asarray(c.neighbours, dtype=int)
    f = lambda a: [float(v) for v in np.asarray(a, dtype=float)]  # noqa: E731
    return {"seed": seed, "step": int(c.step), "time_s": float(c.time_s), "dt": float(c.dt), "index": int(c.index),
            "trigger": c.trigger.value, "hop_count": int(c.hop_count), "duplicate_count": int(c.duplicate_count),
            "own_tx_count": int(c.own_tx_count), "sender_index": -1 if c.sender_index is None else int(c.sender_index),
            "sender_distance_m": float(c.sender_distance_m), "max_sender_distance_m": float(c.max_sender_distance_m),
            "age_s": float(c.age_s), "x": float(c.x), "y": float(c.y), "speed_ms": float(c.speed_ms),
            "heading": float(c.heading), "direction": int(c.direction), "relevance": float(c.relevance),
            "eta_s": float(c.eta_s), "vclass": str(c.vclass), "comm_range_m": float(c.comm_range_m),
            "was_designated": bool(c.was_designated), "sender_dx": float(c.sender_dx), "sender_dy": float(c.sender_dy),
            "nb": [int(v) for v in nb], "nb_dist": f(c.neighbour_distances), "nb_dx": f(c.neighbour_dx),
            "nb_dy": f(c.neighbour_dy), "nb_vx": f(c.neighbour_vx), "nb_vy": f(c.neighbour_vy),
            "nb_rel": f(c.neighbour_relevance), "nb_eta": f(np.asarray(eta_all)[nb]) if eta_all is not None else
            [float(c.eta_s)] * len(nb), "nb_informed": [int(bool(v)) for v in c.neighbour_informed]}


def distribution(policy, ctx) -> tuple[np.ndarray, bool]:
    """Action distribution after the suppression bias, and the gate verdict, for one context (no state change)."""
    graph, mask = policy._prepare(ctx)
    kw = {"deterministic": True}
    if mask is not None:
        kw["action_mask"] = mask
    out = policy.network.act(graph.to_pyg(), **kw)
    probs = np.asarray(out["probs"], dtype=float)
    if policy.suppression_bias != 0.0:
        probs, _ = policy._apply_suppression_bias(probs, np.random.default_rng(0), mask)
    return probs, bool(policy.gate.evaluate(probs).used_fallback)


def unit(n_max: int = 1000) -> dict:
    from experiments.run_sim import RunSpec, run_single
    from experiments.revision5.sweep import checkpoint_for
    from experiments.evaluate_agent import agent_params
    import agents.ai_harp as ah
    orig = ah.AiHarpPolicy.decide
    params = {**agent_params(checkpoint_for(CELL["run"]), tau=CELL["tau"], deterministic=False),
              "suppression_bias": CELL["bias"]}
    total, dmax, gate_mismatch, per_seed, custom = 0, 0.0, 0, {}, None
    seed = CELL["seed"]
    while total < n_max:                                          # seeds 200, 201, ... of the same cell until n_max
        recorded = []

        def rec(self, ctx):
            if self._pre_network(ctx) is None and total + len(recorded) < n_max:
                recorded.append(ctx)
            return orig(self, ctx)
        ah.AiHarpPolicy.decide = rec
        try:
            metrics, res = run_single(RunSpec(scenario=CELL["scenario"], density_veh_km_lane=CELL["density"],
                                              seed=seed, policy="ai_harp", policy_params=params, backend="sumo"),
                                      return_result=True)
        finally:
            ah.AiHarpPolicy.decide = orig
        if custom is None:
            custom = {"rwcr": metrics.get("rwcr"), "action_counts": {str(k): int(v) for k, v in
                                                                   (res.action_counts or {}).items()}}
        cfg = {**CELL, "seed": seed, "n_vehicles": int(res.trace.x.shape[1])}
        a, b = build_policy(cfg), build_policy(cfg)               # in-process reference / the server's path
        for c in recorded:
            pa, ga = distribution(a, c)
            q = json.loads(json.dumps(request_from_ctx(c, seed)))  # through the wire format
            pb, gb = distribution(b, ctx_from_request(q, np.random.default_rng(0)))
            dmax = max(dmax, float(np.max(np.abs(pa - pb))))
            gate_mismatch += int(ga != gb)
        per_seed[seed] = len(recorded)
        total += len(recorded)
        seed += 1
    return {"contexts": total, "contexts_per_seed": per_seed, "max_abs_diff_action_distribution": dmax,
            "gate_verdict_mismatches": gate_mismatch,
            "pass": bool(total >= n_max and dmax <= 1e-6 and gate_mismatch == 0),
            "custom_run_seed200": custom}


def trace(log: Path) -> dict:
    lines = [json.loads(l) for l in log.read_text(encoding="utf-8").splitlines() if l.strip()]
    sess_cfg = next(l["session"] for l in lines if "session" in l)
    reqs = [l for l in lines if "req" in l]
    s = Session(sess_cfg)
    mism = [i for i, r in enumerate(reqs) if s.decide(r["req"])[0] != r["reply"]]
    kinds: dict[str, int] = {}
    for r in reqs:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    return {"log": log.name, "requests": len(reqs), "mismatches": len(mism), "first_mismatches": mism[:10],
            "action_mix": kinds, "pass": bool(reqs and not mism)}


def main(argv=None) -> int:
    import numpy
    print(f"[env] {sys.executable} numpy {numpy.__version__}", flush=True)
    argv = argv if argv is not None else sys.argv[1:]
    OUT.mkdir(parents=True, exist_ok=True)
    if argv[0] == "unit":
        r = unit()
        name = "port_validation_policy_unit.json"
    else:
        r = trace(Path(argv[1]))
        name = "port_validation_policy_trace.json"
    (OUT / name).write_text(json.dumps({"provenance": stamp(f"revision6/veins/{name}", CELL, seeds=[CELL["seed"]]), **r},
                                       indent=1, default=float))
    print(json.dumps(r, indent=1, default=float))
    return 0 if r["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
