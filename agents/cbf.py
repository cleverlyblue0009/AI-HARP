"""ETSI GeoNetworking Contention-Based Forwarding (CBF), for simulator validation.

Not one of the paper's baselines: it exists so the simulator can be checked
against a published curve (experiments/validate_amador.py, reproducing
Amador et al., Vehicular Communications 34 (2022) 100454, Table 3).

ETSI EN 302 636-4-1 CBF, as that paper describes it:

* a receiver inside the destination area buffers the packet for

  .. math:: TO = TO_{max} + \\frac{TO_{min} - TO_{max}}{DIST_{MAX}} \\min(d, DIST_{MAX})

  where ``d`` is its distance from the sender, so farther receivers speak first;
* if it hears the same packet again while buffered, it drops the packet and
  removes the buffered copy (cancel on one duplicate);
* receivers outside the destination area do not rebroadcast.

The timer is rounded UP to whole engine epochs, never below one epoch (a
rebroadcast cannot leave in the epoch of the reception that triggered it). At
the default 100 ms epoch every TO in [1, 100] ms is one epoch, so CBF cannot
express its ordering at all; the validation runs at 10 ms for that reason.
"""

from __future__ import annotations

import math

from agents.base import BROADCAST_NOW, SUPPRESS, Action, ActionType, DecisionContext, Policy, Trigger


def cbf_timeout_s(distance_m: float, to_min_s: float, to_max_s: float, dist_max_m: float) -> float:
    """ETSI CBF buffering time for a receiver ``distance_m`` from the sender."""
    d = min(max(float(distance_m), 0.0), dist_max_m)
    return to_max_s + (to_min_s - to_max_s) * d / dist_max_m


class EtsiCbf(Policy):
    """Contention-based forwarding with a rectangular destination area along x."""

    name = "etsi_cbf"

    def __init__(self, to_min_ms: float = 1.0, to_max_ms: float = 100.0,
                 dist_max_m: float = 1000.0, area_x_min_m: float | None = None,
                 area_x_max_m: float | None = None) -> None:
        if not 0 < to_min_ms <= to_max_ms:
            raise ValueError("need 0 < to_min_ms <= to_max_ms")
        if dist_max_m <= 0:
            raise ValueError("dist_max_m must be positive")
        super().__init__(to_min_ms=to_min_ms, to_max_ms=to_max_ms, dist_max_m=dist_max_m,
                         area_x_min_m=area_x_min_m, area_x_max_m=area_x_max_m)
        self.to_min_s = float(to_min_ms) / 1000.0
        self.to_max_s = float(to_max_ms) / 1000.0
        self.dist_max_m = float(dist_max_m)
        self.area = (area_x_min_m, area_x_max_m)

    def in_area(self, x: float) -> bool:
        lo, hi = self.area
        return (lo is None or x >= lo) and (hi is None or x <= hi)

    def decide(self, ctx: DecisionContext) -> Action:
        if ctx.trigger is Trigger.ORIGINATE:
            return BROADCAST_NOW
        if not self.in_area(ctx.x):
            return SUPPRESS
        to = cbf_timeout_s(ctx.sender_distance_m, self.to_min_s, self.to_max_s, self.dist_max_m)
        steps = max(1, math.ceil(to / ctx.dt - 1e-9))
        return Action(ActionType.DEFER, delay_steps=int(steps), cancel_on_duplicates=1)
