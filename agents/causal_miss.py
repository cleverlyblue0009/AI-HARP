"""Causal actionable deadline-miss rate (revision 9, results/revision9/ADDENDUM.md).

The SAME formula as ``analysis.metrics.actionable_deadline_miss_rate`` (analysis/metrics.py:246-307), with the two
oracle-only inputs replaced by their causal counterparts from ``hazard/risk_field.py``:

* at-risk set: causal peak relevance > ``at_risk_threshold`` (``RiskField.at_risk_set`` on the causal relevance matrix),
  instead of the oracle field's;
* "informed in time": causal relevance > 0 at the informing step, instead of oracle relevance > 0.

Everything else is identical and already causal in the metric: ETA at origination and at informing from the causal
risk field (``RiskField.evaluate``), the latency bound ``hazard.safety_deadline_s``, the reaction time and
``require_actionable_eta`` from ``configs/hazard.yaml`` ``deadlines``, and "absent at origination counts as actionable".

Passing the ORACLE relevance matrix and at-risk set instead reproduces the metric exactly (tests/test_causal_miss.py);
that is how the formula's identity is checked. Training never does that: this module, like everything training imports,
must not reach hazard.oracle, privileged/ or analysis.metrics (tests/test_oracle_isolation.py checks it transitively).
"""

from __future__ import annotations

from typing import Any

import numpy as np


def actionable_miss(rel: np.ndarray, at_risk: np.ndarray, informed_step: np.ndarray, origin_step: int,
                    trace: Any, risk: Any, hazard: Any, deadlines: dict[str, Any]) -> tuple[float, int]:
    """(miss rate, n_actionable) by the analysis/metrics.py formula on the given relevance matrix and at-risk set.
    NaN, 0 if there is no origination or no actionable vehicle."""
    tr = trace
    reaction = float(deadlines.get("reaction_time_s", 2.5))
    if origin_step < 0 or not at_risk.any():
        return float("nan"), 0
    informed = informed_step >= 0
    idx = np.flatnonzero(informed)
    rel_at_info = np.zeros(tr.n_vehicles)
    rel_at_info[idx] = rel[informed_step[idx], idx]
    informed_in_time = informed & (rel_at_info > 0.0)
    latency = np.full(tr.n_vehicles, np.inf)
    latency[idx] = informed_step[idx] * tr.dt - hazard.onset_time_s
    latency = np.maximum(latency, 0.0)

    s = origin_step
    feats = risk.evaluate(np.nan_to_num(tr.x[s]), np.nan_to_num(tr.y[s]), tr.vx[s], tr.vy[s],
                          tr.direction, hazard, s * tr.dt, active=tr.active[s])
    present = tr.active[s]
    had_time = (~present) | (feats["eta_s"] >= reaction)
    actionable = at_risk & had_time
    n_act = int(actionable.sum())
    if n_act == 0:
        return float("nan"), 0
    ok = informed_in_time & actionable
    ok &= latency <= hazard.safety_deadline_s
    if deadlines.get("require_actionable_eta", False):
        oi = np.flatnonzero(ok)
        if oi.size:
            eta = np.array([float(risk.evaluate(tr.x[st, v:v + 1], tr.y[st, v:v + 1], tr.vx[st, v:v + 1],
                                                tr.vy[st, v:v + 1], tr.direction[v:v + 1], hazard,
                                                st * tr.dt)["eta_s"][0])
                            for st, v in zip(informed_step[oi], oi)])
            ok[oi] = eta >= reaction
    return float(1.0 - ok.sum() / n_act), n_act


def causal_actionable_miss(rel_causal: np.ndarray, informed_step: np.ndarray, origin_step: int, trace: Any,
                           risk: Any, hazard: Any, deadlines: dict[str, Any]) -> tuple[float, int]:
    """The causal actionable miss: the metric's formula on the CAUSAL relevance matrix and causal at-risk set."""
    _, at_risk = risk.at_risk_set(rel_causal)
    return actionable_miss(rel_causal, at_risk, informed_step, origin_step, trace, risk, hazard, deadlines)
