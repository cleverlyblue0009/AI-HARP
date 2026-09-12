"""The risk field: ``relevance(v, h, t) -> [0, 1]``.

This is the quantity the whole paper turns on. It answers "how much does *this*
vehicle need *this* hazard message *right now*", and it is what separates
AI-HARP's objective from conventional coverage-maximising dissemination.

Definition
----------
For vehicle :math:`v`, hazard :math:`h` and time :math:`t`:

.. math::

    \\mathrm{rel}(v,h,t) \\;=\\;
        g_{\\text{dir}}(v,h)\\;\\cdot\\;
        g_{\\text{geom}}(v,h,t)\\;\\cdot\\;
        w_{\\eta}\\!\\big(\\eta(v,h,t)\\big)\\;\\cdot\\;
        \\sigma_h(t)^{\\gamma}

with

* **Direction gate** :math:`g_{\\text{dir}}` -- 1 if the hazard threatens both
  carriageways or the vehicle travels on the affected one, otherwise
  ``opposing_direction_relevance`` (default 0). A crash on the opposing
  carriageway is not this vehicle's problem.

* **Geometry gate** :math:`g_{\\text{geom}}` -- 1 while the vehicle is upstream
  of the hazard span and heading into it; ``inside_span_weight`` while it is
  inside the span; **0** once it has passed the span or is travelling away.
  This is the term that makes a vehicle moving away from a landslide score ~0.

* **ETA kernel** :math:`w_\\eta` -- a plateau then exponential decay:

  .. math::

      w_\\eta(\\eta) = \\begin{cases}
        1 & \\eta \\le \\eta_{\\text{full}} \\\\
        e^{-(\\eta - \\eta_{\\text{full}})/\\tau_\\eta} & \\eta > \\eta_{\\text{full}}
      \\end{cases}

  where :math:`\\eta = d_{\\text{to span}} / \\max(\\lVert \\mathbf{v}\\rVert,
  v_{\\min})`. A truck 40 s upstream of a fog bank sits on the plateau and
  scores ~1; a vehicle 5 minutes out is genuinely at risk but not urgent.

* **Severity** :math:`\\sigma_h(t)^{\\gamma}` -- the hazard's severity under its
  decay law, so a fog bank that has dissipated stops generating relevance.

All parameters live in ``configs/hazard.yaml`` under ``risk_field``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from hazard.model import Hazard
from mobility.trace import Trace

# Geometry state codes.
APPROACHING = 0
INSIDE = 1
AWAY = 2  # passed the span, or travelling away from it


class RiskGeometry(Protocol):
    """Maps vehicle kinematics to (distance-to-hazard, geometric state)."""

    def distance_and_state(
        self, x: np.ndarray, y: np.ndarray, vx: np.ndarray, vy: np.ndarray,
        direction: np.ndarray, hazard: Hazard,
    ) -> tuple[np.ndarray, np.ndarray]:
        ...


@dataclass(frozen=True)
class HighwayRiskGeometry:
    """Exact along-corridor geometry for the straight highway scenario.

    The corridor runs along +x. A vehicle's travel direction is its static
    carriageway sign. Distance is measured to the *near edge* of the hazard
    span, which is where the vehicle would first encounter it.
    """

    length_m: float

    def distance_and_state(
        self, x: np.ndarray, y: np.ndarray, vx: np.ndarray, vy: np.ndarray,
        direction: np.ndarray, hazard: Hazard,
    ) -> tuple[np.ndarray, np.ndarray]:
        d = np.asarray(direction, dtype=float)
        # Direction from velocity where the static sign is absent (d == 0).
        d = np.where(d != 0, d, np.sign(np.where(vx != 0, vx, 1.0)))

        inside = (x >= hazard.span_start_m) & (x <= hazard.span_end_m)
        # Heading +x: approaching while upstream of the span's start.
        up_pos = x < hazard.span_start_m
        up_neg = x > hazard.span_end_m
        approaching = np.where(d > 0, up_pos, up_neg)

        dist = np.where(
            d > 0,
            hazard.span_start_m - x,     # +x traffic closes on the near edge
            x - hazard.span_end_m,       # -x traffic closes on the far edge
        )
        dist = np.where(inside, 0.0, np.maximum(dist, 0.0))

        state = np.where(inside, INSIDE, np.where(approaching, APPROACHING, AWAY))
        return dist.astype(float), state.astype(np.int8)


@dataclass(frozen=True)
class GridRiskGeometry:
    """Approximate geometry for the urban grid.

    A grid hazard occupies a stretch of one directed edge, whose midpoint is
    ``hazard_xy`` and whose bearing is ``hazard_heading``.

    In a grid a vehicle's future route is not known (it turns at random), so
    "will this vehicle reach the hazard" cannot be answered exactly. We use
    network (Manhattan) distance to the hazard's midpoint and gate on whether
    the vehicle's *current* heading reduces that distance.

    For a **directional** hazard (a crash blocking one carriageway) there is no
    corridor sign to compare against, so the gate is bearing alignment: a
    vehicle is on the affected approach only if its heading is within
    ``heading_tolerance_rad`` of the hazard edge's heading. Without this, a
    directional hazard in a grid would either match nobody (grid vehicles carry
    ``direction == 0``) or match everybody.

    This is an approximation and is labelled as one wherever grid results are
    reported. It errs in the safe direction: a vehicle that turns away later
    was still counted as at-risk, so RWCR is under- rather than over-stated.
    """

    hazard_xy: tuple[float, float] | None = None
    hazard_heading: float | None = None
    span_tolerance_m: float = 25.0
    heading_tolerance_rad: float = np.pi / 2.0

    def distance_and_state(
        self, x: np.ndarray, y: np.ndarray, vx: np.ndarray, vy: np.ndarray,
        direction: np.ndarray, hazard: Hazard,
    ) -> tuple[np.ndarray, np.ndarray]:
        hx, hy = self.hazard_xy if self.hazard_xy is not None else (hazard.span_mid_m, 0.0)
        dx, dy = hx - x, hy - y
        dist = np.abs(dx) + np.abs(dy)  # Manhattan distance along the grid
        # Closing rate along the grid axes: positive means the current heading
        # reduces the Manhattan distance.
        closing = np.sign(dx) * vx + np.sign(dy) * vy
        inside = dist <= self.span_tolerance_m + hazard.span_length_m / 2.0
        state = np.where(inside, INSIDE, np.where(closing > 0.1, APPROACHING, AWAY))

        heading = self.hazard_heading if self.hazard_heading is not None else hazard.heading_rad
        if hazard.affected_direction != 0 and heading is not None:
            veh_heading = np.arctan2(vy, vx)
            delta = np.abs((veh_heading - heading + np.pi) % (2 * np.pi) - np.pi)
            state = np.where(delta <= self.heading_tolerance_rad, state, AWAY)

        return np.maximum(dist, 0.0), state.astype(np.int8)


@dataclass
class RiskField:
    """Evaluates relevance for every vehicle, with all parameters from YAML."""

    geometry: RiskGeometry
    eta_full_s: float
    eta_decay_tau_s: float
    inside_span_weight: float
    min_speed_ms: float
    severity_gamma: float
    at_risk_threshold: float
    high_relevance_threshold: float
    opposing_direction_relevance: float

    # ------------------------------------------------------------------ core --
    def eta_weight(self, eta_s: np.ndarray) -> np.ndarray:
        """Plateau-then-exponential ETA kernel (see module docstring)."""
        excess = np.maximum(eta_s - self.eta_full_s, 0.0)
        return np.exp(-excess / self.eta_decay_tau_s)

    def direction_gate(self, direction: np.ndarray, hazard: Hazard) -> np.ndarray:
        d = np.asarray(direction, dtype=float)
        if hazard.affected_direction == 0:
            return np.ones_like(d)
        if not np.any(d != 0.0):
            # Grid scenarios carry no carriageway sign; the geometry has
            # already applied the equivalent bearing-alignment gate, so
            # applying a sign comparison here would zero out every vehicle.
            return np.ones_like(d)
        on_affected = d == hazard.affected_direction
        return np.where(on_affected, 1.0, self.opposing_direction_relevance)

    def evaluate(
        self, x: np.ndarray, y: np.ndarray, vx: np.ndarray, vy: np.ndarray,
        direction: np.ndarray, hazard: Hazard, t: float | np.ndarray,
        active: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """Relevance plus the intermediate terms (used as agent features).

        Shapes broadcast: pass ``[N]`` arrays for one timestep or ``[T, N]``
        arrays with ``t`` of shape ``[T, 1]`` for a whole run.
        """
        dist, state = self.geometry.distance_and_state(x, y, vx, vy, direction, hazard)
        speed = np.hypot(vx, vy)
        eta = dist / np.maximum(speed, self.min_speed_ms)

        geom_gate = np.where(
            state == INSIDE, self.inside_span_weight,
            np.where(state == APPROACHING, 1.0, 0.0),
        )
        severity = np.asarray(hazard.severity_at(t), dtype=float)
        rel = (
            self.direction_gate(direction, hazard)
            * geom_gate
            * self.eta_weight(eta)
            * np.power(np.maximum(severity, 0.0), self.severity_gamma)
        )
        rel = np.clip(np.nan_to_num(rel, nan=0.0), 0.0, 1.0)
        if active is not None:
            rel = np.where(active, rel, 0.0)
            eta = np.where(active, eta, np.inf)
        return {"relevance": rel, "eta_s": eta, "distance_m": dist, "state": state,
                "severity": severity}

    def relevance_at(self, trace: Trace, step: int, hazard: Hazard) -> np.ndarray:
        r = self.evaluate(
            trace.x[step], trace.y[step], trace.vx[step], trace.vy[step],
            trace.direction, hazard, trace.time_of(step), active=trace.active[step],
        )
        return r["relevance"]

    def features_at(self, trace: Trace, step: int, hazard: Hazard) -> dict[str, np.ndarray]:
        return self.evaluate(
            trace.x[step], trace.y[step], trace.vx[step], trace.vy[step],
            trace.direction, hazard, trace.time_of(step), active=trace.active[step],
        )

    def relevance_matrix(self, trace: Trace, hazard: Hazard) -> np.ndarray:
        """``[T, N]`` relevance over the whole trace (drives RWCR and TIR)."""
        t = (np.arange(trace.n_steps, dtype=float) * trace.dt)[:, None]
        direction = np.broadcast_to(trace.direction, trace.x.shape)
        out = self.evaluate(
            np.nan_to_num(trace.x), np.nan_to_num(trace.y), trace.vx, trace.vy,
            direction, hazard, t, active=trace.active,
        )
        return out["relevance"]

    # ------------------------------------------------------------------ sets --
    def at_risk_set(self, relevance_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Peak relevance per vehicle and the at-risk membership mask.

        A vehicle belongs to the at-risk set if its relevance *ever* exceeds
        ``at_risk_threshold`` during the run. Peak relevance is the weight it
        carries in RWCR.
        """
        peak = relevance_matrix.max(axis=0)
        return peak, peak > self.at_risk_threshold


def build_risk_field(hz_cfg: dict[str, Any], trace: Trace, hazard: Hazard | None = None) -> RiskField:
    """Construct the risk field for a trace's scenario kind."""
    rf = hz_cfg["risk_field"]
    kind = trace.meta.get("kind", "highway")
    if kind == "grid":
        hazard_xy, hazard_heading = None, None
        if hazard is not None and hazard.edge_id is not None:
            for e in trace.meta.get("edges", []):
                if e["id"] == hazard.edge_id:
                    (x0, y0), (x1, y1) = e["p0"], e["p1"]
                    f = hazard.span_mid_m / max(e["length"], 1e-9)
                    hazard_xy = (x0 + (x1 - x0) * f, y0 + (y1 - y0) * f)
                    hazard_heading = float(e["heading"])
                    break
        if hazard is not None and hazard_xy is None:
            raise ValueError(
                f"Grid hazard {hazard.hazard_id} has no locatable edge "
                f"(edge_id={hazard.edge_id}); the risk field would be meaningless."
            )
        geometry: RiskGeometry = GridRiskGeometry(
            hazard_xy=hazard_xy, hazard_heading=hazard_heading
        )
    else:
        geometry = HighwayRiskGeometry(length_m=float(trace.meta.get("length_m", 10000.0)))

    return RiskField(
        geometry=geometry,
        eta_full_s=float(rf["eta_full_s"]),
        eta_decay_tau_s=float(rf["eta_decay_tau_s"]),
        inside_span_weight=float(rf["inside_span_weight"]),
        min_speed_ms=float(rf["min_speed_ms"]),
        severity_gamma=float(rf["severity_gamma"]),
        at_risk_threshold=float(rf["at_risk_threshold"]),
        high_relevance_threshold=float(rf["high_relevance_threshold"]),
        opposing_direction_relevance=float(rf["opposing_direction_relevance"]),
    )
