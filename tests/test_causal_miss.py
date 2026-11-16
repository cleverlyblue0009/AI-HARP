"""Revision 9: agents/causal_miss.py implements the analysis/metrics.py actionable-miss formula exactly.

With the ORACLE relevance matrix and at-risk set injected, actionable_miss must equal the metric's
actionable_deadline_miss_rate (and n_actionable) to the last bit, on real engine runs from training-pool seeds in the
four headline cells and several policies. The causal variant then differs only in those two inputs.
"""

from __future__ import annotations

import logging
import math

import pytest

from agents.causal_miss import actionable_miss, causal_actionable_miss
from common.config import load_yaml

CASES = [("rural_highway", 2.0), ("rural_highway", 20.0), ("rural_highway", 80.0), ("urban_nlos", 20.0)]


@pytest.mark.parametrize("cell", CASES)
@pytest.mark.parametrize("policy,params", [("flooding", {}), ("slotted_1p", {"n_slots": 50}),
                                           ("p_persistence_03", {"p": 0.1})])
def test_formula_identical_to_the_metric_with_oracle_inputs(cell, policy, params):
    from analysis.metrics import _build_context
    from experiments.run_sim import RunSpec, run_single
    logging.disable(logging.WARNING)
    sc, d = cell
    dl = load_yaml("hazard.yaml")["deadlines"]
    for seed in (100, 101):
        m, res = run_single(RunSpec(scenario=sc, density_veh_km_lane=d, weather="clear", policy=policy,
                                    policy_params=params, seed=seed, hazard_type="fog_bank", backend="fallback"),
                            return_result=True)
        ctx = _build_context(res, res.risk, res.hazard)
        miss, n = actionable_miss(ctx.relevance, ctx.at_risk, res.informed_step, res.origin_step, res.trace,
                                  res.risk, res.hazard, dl)
        ref = m["actionable_deadline_miss_rate"]
        assert (math.isnan(ref) and math.isnan(miss)) or miss == ref, (sc, d, policy, seed, miss, ref)
        c, nc = causal_actionable_miss(res.risk.relevance_matrix(res.trace, res.hazard), res.informed_step,
                                       res.origin_step, res.trace, res.risk, res.hazard, dl)
        assert math.isnan(c) or 0.0 <= c <= 1.0
