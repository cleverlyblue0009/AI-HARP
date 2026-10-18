"""Oracle-relevance reward (rung B) and oracle-relevance node input (rung C).

Training-only; see privileged/__init__.py for the exemption and its enforcement.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from agents.constrained_reward import (EpisodeOutcome, causal_coverage, tree_credit,
                                       warned_at_risk)
from hazard.oracle import build_oracle_risk_field

ORACLE_FEATURE = "extra_relevance_oracle"


def privileged_episode_rewards(result: Any, risk: Any, hazard: Any, lam: float, objective: Any,
                               target: float):
    """agents.constrained_reward.episode_rewards with ORACLE relevance in the coverage
    term and the tree credit. Transmission and collision terms are unchanged."""
    base = getattr(risk, "causal_field", risk)                 # unwrap rung C's wrapper
    oracle = build_oracle_risk_field(base, hazard)
    rel = oracle.relevance_matrix(result.trace, hazard)       # ORACLE
    peak, at_risk, warned = warned_at_risk(rel, result.informed_step, base.at_risk_threshold)
    n_ar = int(at_risk.sum())
    cov = causal_coverage(peak, at_risk, warned)              # same formula, oracle relevance
    credit = tree_credit(result.informed_by, peak, at_risk, warned)
    tx = result.tx_count.astype(float)
    coll = (np.asarray(result.collisions_caused, dtype=float)
            if getattr(result, "collisions_caused", None) is not None else np.zeros_like(tx))
    r = -(tx + objective.collision_weight * coll) + float(lam) * credit
    shortfall = (target - cov) / target if np.isfinite(cov) and target > 0 else 0.0
    outcome = EpisodeOutcome(coverage=cov, target=float(target), n_at_risk=n_ar,
                             transmissions=int(result.n_transmissions),
                             cost_per_at_risk=float(tx.sum() / max(n_ar, 1)),
                             shortfall=float(shortfall), credit_total=float(credit.sum()))
    return {v: float(r[v]) for v in range(len(r))}, outcome


class OracleFeatureRisk:
    """Wraps a causal RiskField; adds per-vehicle oracle relevance to features_at().

    Everything else (relevance, eta, at-risk threshold...) is the causal field's, so
    every original input stays causal and the oracle arrives only as the extra column.
    """

    def __init__(self, risk: Any, trace: Any, hazard: Any) -> None:
        self.causal_field = risk
        self._rel = build_oracle_risk_field(risk, hazard).relevance_matrix(trace, hazard)

    def features_at(self, trace, step, hazard):
        f = dict(self.causal_field.features_at(trace, step, hazard))
        f[ORACLE_FEATURE] = self._rel[step]
        return f

    def __getattr__(self, name):
        return getattr(self.causal_field, name)


def wrap_builder(build_risk_field):
    """A drop-in for hazard.risk_field.build_risk_field that returns OracleFeatureRisk."""

    def build(hz_cfg, trace, hazard):
        return OracleFeatureRisk(build_risk_field(hz_cfg, trace, hazard), trace, hazard)

    build.__privileged__ = True
    return build
