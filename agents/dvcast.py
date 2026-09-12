"""DV-CAST-style store-carry-forward (Phase 4, baseline 7).

Every other baseline assumes the network is connected and only argues about how
many neighbours should relay. In sparse traffic that assumption fails outright:
there is frequently *no* neighbour in the propagation direction, and a scheme
that can only choose between "rebroadcast" and "suppress" loses the message
permanently. DV-CAST's contribution is recognising that the right third option
is to **hold the message and keep driving**.

Following Tonguz et al. (2010), a vehicle classifies its local neighbourhood
each time it has to decide, and the classification selects the behaviour:

===========================  ==========================================
Local connectivity           Action
===========================  ==========================================
Neighbours ahead             Well connected -- suppress via slotted
(same propagation direction) persistence, exactly as baseline 3.
Only opposing-direction      Sparsely connected -- rebroadcast now and
neighbours                   let oncoming traffic carry the message
                             back the other way.
No neighbours at all         Disconnected -- CARRY: hold the message and
                             re-evaluate every ``carry_recheck_epochs``
                             until a neighbour appears or the TTL expires.
===========================  ==========================================

The three cases correspond to DV-CAST's DFlg / ODN / MDC flags.

This is the baseline to beat in the sparse regime, and the one whose failure
mode is most interesting: carrying is *free* in transmissions but expensive in
time, and DV-CAST has no notion of how much time it has. It will happily carry
a message for 30 s towards a landslide that the at-risk vehicles will reach in
10 s, because nothing in the protocol knows the deadline. That is the opening
the risk field is designed to exploit -- and the reason TIR and the deadline
miss rate, not coverage, are where the difference should show up.
"""

from __future__ import annotations

import numpy as np

from agents.base import (
    BROADCAST_NOW,
    SUPPRESS,
    Action,
    ActionType,
    DecisionContext,
    Policy,
    Trigger,
)
from agents.persistence import distance_slot

# Connectivity classes, exposed so tests and figures can name them.
WELL_CONNECTED = "well_connected"
SPARSELY_CONNECTED = "sparsely_connected"
DISCONNECTED = "disconnected"


class DvCast(Policy):
    """Connectivity-aware broadcast with store-carry-forward."""

    name = "dvcast"
    # A carrying vehicle must revise its plan when it overhears the message
    # again -- that is how it learns the gap it was carrying across has closed.
    wants_duplicate_callbacks = True

    def __init__(
        self,
        n_slots: int = 5,
        slot_epochs: int = 1,
        cancel_on_duplicates: int = 1,
        carry_recheck_epochs: int = 10,
        use_opposite_direction_carriers: bool = True,
        opposite_heading_threshold_rad: float = np.pi / 2,
    ) -> None:
        super().__init__(
            n_slots=n_slots, slot_epochs=slot_epochs,
            cancel_on_duplicates=cancel_on_duplicates,
            carry_recheck_epochs=carry_recheck_epochs,
            use_opposite_direction_carriers=use_opposite_direction_carriers,
            opposite_heading_threshold_rad=opposite_heading_threshold_rad,
        )
        self.n_slots = int(n_slots)
        self.slot_epochs = int(slot_epochs)
        self.cancel_on_duplicates = int(cancel_on_duplicates)
        self.carry_recheck_epochs = int(carry_recheck_epochs)
        self.use_opposite_direction_carriers = bool(use_opposite_direction_carriers)
        self.opposite_heading_threshold_rad = float(opposite_heading_threshold_rad)

    # ------------------------------------------------------- classification --
    def classify(self, ctx: DecisionContext) -> str:
        """Which of DV-CAST's three connectivity regimes this vehicle is in."""
        if ctx.n_neighbours == 0:
            return DISCONNECTED

        # Neighbours that would carry the message further along its current
        # propagation direction. With no sender (origination, or a carry timer
        # firing) there is no direction yet, so any neighbour counts as ahead.
        if ctx.sender_index is None:
            has_ahead = True
        else:
            has_ahead = bool((ctx.progress() > 0.0).any())

        if has_ahead:
            return WELL_CONNECTED
        if self.use_opposite_direction_carriers and self._has_opposing(ctx):
            return SPARSELY_CONNECTED
        return DISCONNECTED

    def _has_opposing(self, ctx: DecisionContext) -> bool:
        """Any neighbour travelling against this vehicle's heading.

        Oncoming traffic is a transport resource: it is moving towards the
        region behind us, which on a hazard corridor is exactly where the
        upstream at-risk vehicles are.
        """
        own = np.array([np.cos(ctx.heading), np.sin(ctx.heading)])
        speed = np.hypot(ctx.neighbour_vx, ctx.neighbour_vy)
        moving = speed > 0.5
        if not moving.any():
            return False
        dot = (ctx.neighbour_vx * own[0] + ctx.neighbour_vy * own[1]) / np.maximum(speed, 1e-9)
        angle = np.arccos(np.clip(dot, -1.0, 1.0))
        return bool((moving & (angle > self.opposite_heading_threshold_rad)).any())

    # --------------------------------------------------------------- decide --
    def decide(self, ctx: DecisionContext) -> Action:
        if ctx.trigger is Trigger.ORIGINATE:
            return BROADCAST_NOW

        # Broadcast suppression, DV-CAST's first flag. Both of these checks are
        # load-bearing because this policy asks to be re-invoked on duplicate
        # receptions: without them a vehicle re-arms a rebroadcast every time it
        # overhears the message, and the "store-carry-forward" scheme that is
        # supposed to use the FEWEST transmissions instead uses forty times
        # more than blind flooding.
        if ctx.own_tx_count > 0:
            # Already relayed this message; relaying again adds nothing.
            return SUPPRESS
        if ctx.trigger is Trigger.RECEIVE and ctx.is_duplicate:
            # Heard it again from someone else: the neighbourhood is covered
            # and any gap we were carrying across has closed. Stop carrying.
            return SUPPRESS

        regime = self.classify(ctx)

        if regime == WELL_CONNECTED:
            # Plenty of relays available: fall back to distance-ordered
            # suppression so the neighbourhood does not storm.
            if ctx.trigger is Trigger.TIMER:
                # A carried message whose gap has closed: send it now, the wait
                # is already paid for.
                return BROADCAST_NOW
            slot = distance_slot(ctx.sender_distance_m, ctx.comm_range_m, self.n_slots)
            return Action(
                ActionType.DEFER,
                delay_steps=1 + slot * self.slot_epochs,
                cancel_on_duplicates=self.cancel_on_duplicates,
            )

        if regime == SPARSELY_CONNECTED:
            # Nobody ahead, but oncoming traffic can carry it: speak now.
            return BROADCAST_NOW

        # Disconnected: hold it and keep driving.
        return Action(ActionType.CARRY, delay_steps=self.carry_recheck_epochs)
