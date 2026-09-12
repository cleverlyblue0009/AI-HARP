"""Oracle risk field: ground-truth at-risk labelling for EVALUATION ONLY.

.. danger::
   **Nothing under ``agents/`` or ``sim/`` may import this module.**
   ``tests/test_oracle_isolation.py`` fails the build if anything does.

   The oracle looks at a vehicle's *realised* trajectory to decide whether it
   actually encountered the hazard. That information does not exist at decision
   time. Letting it reach a node feature, a reward term, or a policy input
   would leak the future into the controller and invalidate every result.

Why an oracle is needed
-----------------------
The causal risk field in :mod:`hazard.risk_field` estimates at-risk-ness from
information available at time *t*: position, heading, speed. On a straight
corridor that estimate is essentially exact, because heading determines
destiny. In a grid it is not: vehicles turn, so an instantaneous heading says
little about whether a vehicle will reach the hazard.

Measured consequence, which is what forced this split: on ``urban_nlos`` at
20 veh/km/lane, *every* vehicle was informed, yet causal RWCR read 0.291,
because ~70% of vehicles happened to be pointing away from the hazard at the
instant the packet arrived and were therefore scored as "not informed in time".
RWCR was measuring vehicle heading, not warning effectiveness.

The split
---------
``relevance_oracle``  ground truth. Did this vehicle *actually* enter the
                      hazard span, and how long before it did so was it
                      informed? Used only for the at-risk set, RWCR and TIR.
``relevance_causal``  the estimate a vehicle could actually form at time *t*.
                      Used for agent node features and the reward.

Bounded lookahead
-----------------
The oracle horizon is **not** unbounded. A vehicle that reaches the hazard in
ten minutes is not meaningfully "at risk now", and counting it would inflate
the denominator exactly as the broken causal version did. The horizon is
derived, not chosen: it is the ETA at which the risk field's own kernel decays
below its own at-risk threshold,

    H = eta_full + tau * ln(severity / at_risk_threshold)

capped by the hazard's remaining lifetime under its decay law. Beyond H the
causal field would score the vehicle below threshold anyway, so the two
definitions are asking the same question over the same window.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hazard.model import DecayKind, Hazard
from hazard.risk_field import INSIDE, RiskField
from mobility.trace import Trace

#: Sentinel for "this vehicle never encounters the hazard".
NEVER = np.inf


@dataclass
class OracleRiskField:
    """Ground-truth relevance from realised trajectories. Evaluation only."""

    causal: RiskField
    horizon_s: float

    # ------------------------------------------------------------ encounters --
    def encounter_time_s(self, trace: Trace, hazard: Hazard) -> np.ndarray:
        """``[N]`` absolute time at which each vehicle enters the hazard span.

        ``inf`` for vehicles that never do. Vehicles still approaching when the
        trace ends are extrapolated from their final position and speed rather
        than censored -- censoring them would bias the at-risk set toward
        vehicles that happen to start near the hazard.
        """
        x = np.nan_to_num(trace.x)
        y = np.nan_to_num(trace.y)
        direction = np.broadcast_to(trace.direction, x.shape)
        _, state = self.causal.geometry.distance_and_state(
            x, y, trace.vx, trace.vy, direction, hazard
        )
        inside = (state == INSIDE) & trace.active

        # A directional hazard only threatens its own carriageway; a vehicle
        # physically passing the site on the other side has not "encountered" it.
        if hazard.affected_direction != 0 and np.any(trace.direction != 0):
            on_affected = trace.direction == hazard.affected_direction
            inside = inside & on_affected[None, :]

        ever = inside.any(axis=0)
        first_step = np.where(ever, inside.argmax(axis=0), -1)
        out = np.where(ever, first_step * trace.dt, NEVER).astype(float)

        # Extrapolate vehicles that are still closing when the window ends.
        undetermined = ~ever & trace.active.any(axis=0)
        if undetermined.any():
            idx = np.flatnonzero(undetermined)
            last_step = (trace.n_steps - 1) - np.argmax(
                trace.active[::-1, idx], axis=0
            )
            feats = self.causal.evaluate(
                x[last_step, idx], y[last_step, idx],
                trace.vx[last_step, idx], trace.vy[last_step, idx],
                trace.direction[idx], hazard, last_step * trace.dt,
            )
            approaching = feats["state"] != 2          # not AWAY
            eta = feats["eta_s"]
            est = last_step * trace.dt + eta
            out[idx] = np.where(approaching & np.isfinite(eta), est, NEVER)
        return out

    # ------------------------------------------------------------- relevance --
    def relevance_matrix(self, trace: Trace, hazard: Hazard) -> np.ndarray:
        """``[T, N]`` ground-truth relevance.

        Same kernel and severity law as the causal field, so the two are
        directly comparable; the only difference is that the geometry gate
        comes from what actually happened rather than from heading.
        """
        enc = self.encounter_time_s(trace, hazard)
        t = (np.arange(trace.n_steps, dtype=float) * trace.dt)[:, None]
        eta = enc[None, :] - t                       # time until encounter

        in_window = (eta >= 0.0) & (eta <= self.horizon_s)
        severity = np.asarray(hazard.severity_at(t), dtype=float)
        rel = (
            in_window
            * self.causal.eta_weight(np.where(in_window, eta, 0.0))
            * np.power(np.maximum(severity, 0.0), self.causal.severity_gamma)
        )
        rel = np.where(trace.active, rel, 0.0)
        return np.clip(np.nan_to_num(rel, nan=0.0, posinf=0.0), 0.0, 1.0)

    def at_risk_set(self, relevance_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        peak = relevance_matrix.max(axis=0)
        return peak, peak > self.causal.at_risk_threshold


def oracle_horizon_s(risk: RiskField, hazard: Hazard) -> float:
    """Derived lookahead horizon -- see the module docstring.

    The ETA beyond which the risk field's own kernel falls below its own
    at-risk threshold, capped by how long the hazard will still exist.
    """
    sev = max(hazard.severity0, 1e-6)
    thresh = max(risk.at_risk_threshold, 1e-6)
    if sev <= thresh:
        horizon = risk.eta_full_s
    else:
        horizon = risk.eta_full_s + risk.eta_decay_tau_s * float(np.log(sev / thresh))

    # A hazard that will have decayed away cannot threaten anyone beyond that.
    if hazard.decay is DecayKind.LINEAR_TTL and hazard.ttl_s:
        horizon = min(horizon, float(hazard.ttl_s))
    elif hazard.decay is DecayKind.EXPONENTIAL and hazard.tau_s:
        horizon = min(horizon, float(hazard.tau_s) * np.log(sev / thresh))
    return float(max(horizon, risk.eta_full_s))


def build_oracle_risk_field(risk: RiskField, hazard: Hazard) -> OracleRiskField:
    return OracleRiskField(causal=risk, horizon_s=oracle_horizon_s(risk, hazard))


# ---------------------------------------------------------------------------
# Oracle-vs-causal disagreement: a reported result in its own right.
# ---------------------------------------------------------------------------
def estimation_agreement(
    trace: Trace, hazard: Hazard, risk: RiskField, oracle: OracleRiskField
) -> dict[str, float]:
    """How well can risk be estimated from information available at time t?

    This is not diagnostics -- it is a finding. On a corridor, heading
    determines destiny and the causal estimate is near-exact. In a grid it is
    not, and the gap quantifies how much harder the estimation problem is when
    vehicles can turn. That difficulty is precisely what a learned policy is
    being asked to overcome, so the paper should report it before reporting
    the policy's performance.

    Returns precision/recall of the causal at-risk set against ground truth,
    plus the correlation of peak relevance.
    """
    causal_mat = risk.relevance_matrix(trace, hazard)
    oracle_mat = oracle.relevance_matrix(trace, hazard)
    c_peak, c_set = risk.at_risk_set(causal_mat)
    o_peak, o_set = oracle.at_risk_set(oracle_mat)

    tp = int((c_set & o_set).sum())
    fp = int((c_set & ~o_set).sum())
    fn = int((~c_set & o_set).sum())
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0
          else float("nan"))

    both = np.isfinite(c_peak) & np.isfinite(o_peak)
    corr = (float(np.corrcoef(c_peak[both], o_peak[both])[0, 1])
            if both.sum() > 2 and c_peak[both].std() > 0 and o_peak[both].std() > 0
            else float("nan"))

    return {
        "risk_est_precision": precision,
        "risk_est_recall": recall,
        "risk_est_f1": f1,
        "risk_peak_corr": corr,
        "n_at_risk_oracle": int(o_set.sum()),
        "n_at_risk_causal": int(c_set.sum()),
        "oracle_horizon_s": oracle.horizon_s,
    }
