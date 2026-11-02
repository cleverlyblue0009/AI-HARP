"""Revision 6 Step 1: validate the RAR and SCF-slotted Veins ports (results/revision6/veins/port_validation_rules.md).

    python -m experiments.revision6.veins.validate_rules

Input: the two validation replays (rural d=20, seed 200; RAR abc=(1.0, 0.5, 0.0), SCF (8 slots, 5 s)) in
cache/veins_runs/r6_validation/, run with experiments/revision6/veins/run_veins6.py. Every rule decision in Veins logs
its inputs (F line) followed by its action (D line). Checks, with the unchanged Python code:
  * rho_c: RiskField.evaluate on the inputs the app used (position, velocity, carriageway sign, time) vs the app's
    rho_c -- must agree to 1e-9; and, reported as measured, vs the custom simulator's trace-based rho_c at the same
    (vehicle, step) (FCD-printed positions, float32 velocities);
  * decisions: RelevanceAwareRule / StoreCarryForwardSlotted.decide on a DecisionContext with the logged inputs (the
    logged uniform draw for RAR; the logged carry start for SCF) vs the app's action and delay -- must be identical;
  * where the beacon-derived neighbour count equals the custom simulator's geometric count at that (vehicle, step),
    the RAR rebroadcast probability from the trace-based rho_c, next to the app's.
"""

from __future__ import annotations

import gzip
import json
import math
import sys
from pathlib import Path

import numpy as np

from agents.base import ActionType, DecisionContext, Trigger
from agents.rule_baselines import RelevanceAwareRule, StoreCarryForwardSlotted
from common.config import PROJECT_ROOT
from experiments.revision1.provenance import stamp
from experiments.run_sim import RunSpec, run_single

RUNS = PROJECT_ROOT / "cache" / "veins_runs" / "r6_validation"
OUT = PROJECT_ROOT / "results" / "revision6" / "veins"
CELL = ("rural_highway", 20.0, 200)
RAR_ABC, SCF_ST = (1.0, 0.5, 0.0), (8, 5.0)
TRIG = {"receive": Trigger.RECEIVE, "timer": Trigger.TIMER, "originate": Trigger.ORIGINATE}


class _FixedRng:
    """ctx.rng stand-in returning the uniform the app drew (RAR draws exactly one)."""
    def __init__(self, u: float):
        self.u = u

    def random(self):
        return self.u


def ctx_for(idx, step, dt, trig, n, sender_dist, comm_range, rel=0.0, rng=None) -> DecisionContext:
    z = np.zeros(n)
    return DecisionContext(
        step=step, time_s=step * dt, dt=dt, index=idx, trigger=TRIG[trig], hop_count=1, duplicate_count=0,
        own_tx_count=0, sender_index=None, sender_distance_m=sender_dist, max_sender_distance_m=sender_dist,
        age_s=0.0, x=0.0, y=0.0, speed_ms=0.0, heading=0.0, direction=1, relevance=rel, eta_s=0.0, vclass="car",
        neighbours=np.arange(n), neighbour_distances=z, neighbour_dx=z, neighbour_dy=z, neighbour_vx=z,
        neighbour_vy=z, neighbour_relevance=z, neighbour_informed=z.astype(bool), comm_range_m=comm_range,
        rng=rng or np.random.default_rng(0))


def events(name: str) -> list[list[str]]:
    with gzip.open(RUNS / f"{name}.events.csv.gz", "rt") as f:
        return [line.rstrip("\n").split(",") for line in f]


def pairs(ev):
    """(F fields, D fields) for each rule decision: an F line followed by the D line of the same vehicle and time."""
    out, last = [], {}
    for e in ev:
        if e[0] == "F":
            last[(e[1], e[3])] = e
        elif e[0] == "D" and (e[1], e[3]) in last:
            out.append((last.pop((e[1], e[3])), e))
    return out


def main(argv=None) -> int:
    import numpy
    print(f"[env] {sys.executable} numpy {numpy.__version__}", flush=True)
    sc, d, seed = CELL
    _, res = run_single(RunSpec(scenario=sc, density_veh_km_lane=d, seed=seed, policy="flooding", backend="sumo"),
                        return_result=True)
    rf, hz, tr = res.risk, res.hazard, res.trace
    dt, R = float(tr.dt), float(res.meta["comm_range_m"])
    eng = res  # geometric neighbour count as the custom engine forms it (sim/engine.py _context): active, <= R
    out = {"cell": f"{sc} d={d:g}", "seed": seed}

    # ---- RAR -------------------------------------------------------------------------------------------------
    rar_name = f"{sc}_d{d:g}_s{seed}__rar_{'_'.join(f'{v:g}' for v in RAR_ABC)}__s{seed}"
    rec = json.loads((RUNS / f"{rar_name}.run.json").read_text())
    pol = RelevanceAwareRule(abc=list(RAR_ABC))
    n_dec = 0
    dr_inputs, dr_trace, dp_same_n, mism = [], [], [], []
    n_same_n = 0
    for f, dline in pairs(events(rar_name)):
        _, t, step, idx, trig, x, y, vx, vy, dr, rel, eta, n, sdist, u, p = f
        step, idx, n = int(step), int(idx), int(n)
        x, y, vx, vy, rel, sdist, u, p = map(float, (x, y, vx, vy, rel, sdist, u, p))
        ref = rf.evaluate(np.array([x]), np.array([y]), np.array([vx]), np.array([vy]), np.array([int(dr)]), hz,
                          step * dt)["relevance"][0]
        dr_inputs.append(abs(ref - rel))
        feats = rf.features_at(tr, step, hz)
        rel_tr = float(feats["relevance"][idx])
        dr_trace.append(abs(rel_tr - rel))
        c = ctx_for(idx, step, dt, trig, n, sdist, R, rel=float(ref), rng=_FixedRng(u))
        p_py = pol.rebroadcast_probability(c)
        act = pol.decide(c)
        name = {ActionType.SUPPRESS: "suppress", ActionType.DEFER: "defer"}.get(act.kind, act.kind.value)
        delay = act.delay_steps if act.kind is ActionType.DEFER else 1
        if name != dline[5] or (name == "defer" and int(dline[6]) != delay) or abs(p_py - p) > 1e-9:
            mism.append({"step": step, "idx": idx, "veins": dline[5:7], "python": [name, delay], "p": [p, p_py]})
        # custom simulator's geometric neighbour count at this (vehicle, step)
        act_s = tr.active[step]
        dd = np.hypot(tr.x[step] - tr.x[step, idx], tr.y[step] - tr.y[step, idx])
        n_tr = int(np.sum(act_s & (dd <= R)) - 1)
        if n_tr == n:
            n_same_n += 1
            p_tr = pol.rebroadcast_probability(ctx_for(idx, step, dt, trig, n_tr, sdist, R, rel=rel_tr))
            dp_same_n.append(abs(p_tr - p))
        n_dec += 1
    out["rar"] = {"replay": rar_name, "exit_code": rec["exit_code"], "trace_identical": rec["fcd_check"]["identical"],
                  "decisions": n_dec,
                  "rho_c_max_abs_diff_identical_inputs": max(dr_inputs) if dr_inputs else None,
                  "rho_c_max_abs_diff_vs_custom_trace": max(dr_trace) if dr_trace else None,
                  "rho_c_median_abs_diff_vs_custom_trace": float(np.median(dr_trace)) if dr_trace else None,
                  "decisions_with_same_neighbour_count": n_same_n,
                  "p_max_abs_diff_same_neighbour_count": max(dp_same_n) if dp_same_n else None,
                  "decision_mismatches": len(mism), "mismatch_examples": mism[:10]}

    # ---- SCF -------------------------------------------------------------------------------------------------
    scf_name = f"{sc}_d{d:g}_s{seed}__scf_slotted_{SCF_ST[0]:g}_{SCF_ST[1]:g}__s{seed}"
    rec = json.loads((RUNS / f"{scf_name}.run.json").read_text())
    smism, n_dec = [], 0
    for f, dline in pairs(events(scf_name)):
        step, idx, trig, n, sdist, cs = int(f[2]), int(f[3]), f[4], int(f[12]), float(f[13]), float(f[15])
        pol = StoreCarryForwardSlotted(slots_timeout=list(SCF_ST))
        if cs >= 0:
            pol._carry_start[idx] = cs
        act = pol.decide(ctx_for(idx, step, dt, trig, n, sdist, R))
        name = {ActionType.SUPPRESS: "suppress", ActionType.DEFER: "defer", ActionType.CARRY: "carry",
                ActionType.BROADCAST: "broadcast"}[act.kind]
        delay = act.delay_steps if act.kind in (ActionType.DEFER, ActionType.CARRY) else 1
        if name != dline[5] or int(dline[6]) != delay:
            smism.append({"step": step, "idx": idx, "trigger": trig, "n": n, "veins": dline[5:7], "python": [name, delay]})
        n_dec += 1
    out["scf"] = {"replay": scf_name, "exit_code": rec["exit_code"], "trace_identical": rec["fcd_check"]["identical"],
                  "decisions": n_dec, "decision_mismatches": len(smism), "mismatch_examples": smism[:10]}
    r = out["rar"]
    out["pass"] = bool(r["exit_code"] == 0 and r["trace_identical"] and r["decisions"] > 0
                       and r["rho_c_max_abs_diff_identical_inputs"] <= 1e-9 and r["decision_mismatches"] == 0
                       and out["scf"]["exit_code"] == 0 and out["scf"]["trace_identical"]
                       and out["scf"]["decisions"] > 0 and out["scf"]["decision_mismatches"] == 0)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "port_validation_rules.json").write_text(json.dumps(
        {"provenance": stamp("revision6/veins/port_validation_rules", {"cell": CELL, "rar": RAR_ABC, "scf": SCF_ST},
                             seeds=[seed]), **out}, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
