"""Revision 8, step 0 (design probe, not training and not evaluation): can the STRUCT heads as specified in the
revision-8 instruction contain RAR and slotted_1p exactly?

    python -m experiments.revision8.probe_heads        -> results/revision8/probe_heads.json

Records every first-reception RAR decision context (default abc = (1, 0.5, 0)) on TRAINING-POOL seeds 100-103 only
(rural d=20, urban NLOS d=20, rural d=2; fallback mobility), with the uniform RAR draws, and compares:
  p head:     sigmoid(w.x) (as specified) vs clip(w.x, 0, 1) (alternative), x = (1, rho_c, min(n,30)/30, d/R);
  delay head: 1 + round(50*sigmoid(w.x')) (as specified) vs 1 + clip(round(w.x'), 0, 50) (alternative), with one
              distance-slot term in x' (s50 = slot index for 50 slots, or its logit transform), against
              slotted_1p's 1 + slot(n_slots) for n_slots = 5 and 50.
The sigmoid weights are FITTED ON THE SAME CONTEXTS (least squares for p; exhaustive grid for the delay), i.e. the
most favourable case for the sigmoid. No outcome metric (RWCR, cost, miss) is read or stored.
"""

from __future__ import annotations

import json
import logging

import numpy as np

from agents import registry
from agents.persistence import distance_slot
from agents.rule_baselines import RelevanceAwareRule
from common.config import PROJECT_ROOT

SEEDS = (100, 101, 102, 103)
CELLS = (("rural_highway", 20.0), ("urban_nlos", 20.0), ("rural_highway", 2.0))
REC: list[tuple] = []


class _Recording(RelevanceAwareRule):
    def decide(self, ctx):
        if ctx.trigger.value == "receive" and ctx.own_tx_count == 0:
            g = np.random.default_rng()
            g.bit_generator.state = ctx.rng.bit_generator.state      # the uniform RAR is about to draw
            REC.append((ctx.relevance, ctx.n_neighbours, ctx.sender_distance_m, ctx.comm_range_m, g.random(),
                        self.rebroadcast_probability(ctx)))
        return super().decide(ctx)


def main() -> int:
    from scipy.optimize import minimize

    from experiments.run_sim import RunSpec, run_single
    logging.disable(logging.WARNING)
    registry._REGISTRY["_rar_recording"] = _Recording
    for seed in SEEDS:
        for sc, d in CELLS:
            run_single(RunSpec(scenario=sc, density_veh_km_lane=d, weather="clear", policy="_rar_recording",
                               policy_params={"abc": [1.0, 0.5, 0.0]}, seed=seed, hazard_type="fog_bank",
                               backend="fallback"))
    A = np.array(REC)
    rho, n, sd, R, u, p = A.T
    x = np.stack([np.ones_like(rho), rho, np.minimum(n, 30) / 30, np.clip(sd / R, 0, 1)], 1)
    out: dict = {"contexts": int(len(p)), "seeds": list(SEEDS), "cells": [list(c) for c in CELLS]}

    def loss(w):
        return float(np.mean((1 / (1 + np.exp(-x @ w)) - p) ** 2))
    fits = [minimize(loss, w0, method="Nelder-Mead", options={"maxiter": 40000, "xatol": 1e-10, "fatol": 1e-15})
            for w0 in ([0, 4, -2, 0], [-2, 6, -3, 0], [0, 10, -5, 0], [1, 3, -1, 0])]
    best = min(fits, key=lambda r: r.fun)
    ps = 1 / (1 + np.exp(-x @ best.x))
    ph = np.clip(x @ np.array([0.5, 1.0, -0.5, 0.0]), 0, 1)       # a*rho + b*(1 - n/30) + c with (1, 0.5, 0)
    out["p_head"] = {
        "sigmoid_best_fit_w": best.x.tolist(), "sigmoid_max_abs_dp": float(np.abs(ps - p).max()),
        "sigmoid_decision_mismatches": int(((u < ps) != (u < p)).sum()),
        "clipped_linear_w": [0.5, 1.0, -0.5, 0.0], "clipped_linear_max_abs_dp": float(np.abs(ph - p).max()),
        "clipped_linear_decision_mismatches": int(((u < ph) != (u < p)).sum()),
        "share_rar_p_clipped_to_0_or_1": float(((p <= 0) | (p >= 1)).mean()),
    }
    # delay head: slot terms for every recorded sender distance
    s50 = np.array([distance_slot(a, b, 50) for a, b in zip(sd, R)], dtype=float)
    q = (s50 + 0.25) / 50
    t50 = np.log(q / (1 - q))
    res = {}
    for n_slots in (5, 50):
        target = np.array([distance_slot(a, b, n_slots) for a, b in zip(sd, R)], dtype=float)
        # the slot term takes at most 50 values: search on the distinct values, weighted by their counts
        for name, feat in (("s50", s50), ("logit_s50", t50)):
            keys, inv = np.unique(np.stack([feat, target], 1), axis=0, return_inverse=True)
            cnt = np.bincount(inv.ravel(), minlength=len(keys))
            fu, tu = keys[:, 0], keys[:, 1]
            A_, B_ = np.meshgrid(np.linspace(-3, 3, 1201), np.linspace(-10, 10, 2001), indexing="ij")
            bestm = None
            for i in range(A_.shape[0]):
                z = A_[i][:, None] * fu[None, :] + B_[i][:, None]
                m = ((np.round(50 / (1 + np.exp(-z))) != tu[None, :]) * cnt[None, :]).sum(axis=1)
                j = int(np.argmin(m))
                if bestm is None or m[j] < bestm[0]:
                    bestm = (int(m[j]), float(A_[i, j]), float(B_[i, j]))
            res[f"n_slots={n_slots}|sigmoid|{name}"] = {"min_mismatches": bestm[0], "a": bestm[1], "b": bestm[2]}
        w = 1.0 if n_slots == 50 else 0.1
        c = 0.0 if n_slots == 50 else -0.45
        hard = np.clip(np.round(w * s50 + c), 0, 50)
        res[f"n_slots={n_slots}|clipped_linear|s50"] = {"mismatches": int((hard != target).sum()), "a": w, "b": c}
    out["delay_head"] = res
    from experiments.revision1.provenance import stamp, write_json
    p_out = PROJECT_ROOT / "results" / "revision8" / "probe_heads.json"
    write_json(p_out, {"provenance": stamp("revision8/probe_heads", {"seeds": list(SEEDS)}, seeds=list(SEEDS)), **out})
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
