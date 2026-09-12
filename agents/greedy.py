"""Greedy farthest-relay selection (Phase 4, baseline 6).

Sender-side relay selection: instead of every receiver deciding for itself, the
transmitting vehicle reads its neighbour table (populated for free from CAM/BSM
beacons) and names the neighbour with the greatest progress along the
propagation direction as the relay. Everyone else stays quiet.

This is the strongest of the geometric baselines in dense traffic -- one relay
per hop, maximum geographic progress, no probabilistic waste -- and it is the
closest baseline in spirit to what AI-HARP does, which is why it matters that
it is implemented at full strength rather than as a straw man:

* **Bidirectional seeding.** A hazard message must travel both ways along a
  two-way corridor. The originator therefore designates the best relay on each
  side; designating one would leave half the at-risk set unreachable for
  reasons that have nothing to do with the relay rule.
* **Implicit-ACK fallback.** Broadcast is unacknowledged, so a designation can
  simply be lost -- if the designated relay does not decode the frame, the
  chain dies silently. Real greedy protocols handle this with a timeout: the
  other receivers arm a distance-ranked timer and take over if they do not hear
  the designated relay speak. Without that, this baseline would fail for a
  reason unrelated to relay selection, and beating it would prove nothing.

Its remaining weakness is the honest one: "greatest progress" is a purely
geometric criterion. The farthest neighbour may be the one driving away from
the hazard, in which case the message is propagating fast in the direction
nobody needs.
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


class GreedyFarthestRelay(Policy):
    """Designate the farthest-progress neighbour as the next relay."""

    name = "greedy_farthest"

    def __init__(
        self,
        fallback_enabled: bool = True,
        fallback_epochs: int = 3,
        fallback_n_slots: int = 4,
        seed_both_directions: bool = True,
    ) -> None:
        super().__init__(
            fallback_enabled=fallback_enabled,
            fallback_epochs=fallback_epochs,
            fallback_n_slots=fallback_n_slots,
            seed_both_directions=seed_both_directions,
        )
        self.fallback_enabled = bool(fallback_enabled)
        self.fallback_epochs = int(fallback_epochs)
        self.fallback_n_slots = int(fallback_n_slots)
        self.seed_both_directions = bool(seed_both_directions)

    # ------------------------------------------------------------ selection --
    def select_relays(self, ctx: DecisionContext) -> tuple[int, ...]:
        """Pick the relay(s) to designate.

        On origination there is no propagation direction yet, so the corridor
        is split by bearing and the farthest neighbour on each side is chosen.
        On a relay hop, only neighbours with positive progress (i.e. beyond
        this vehicle, away from the sender) are eligible.
        """
        if ctx.n_neighbours == 0:
            return ()

        if ctx.sender_index is None:
            if not self.seed_both_directions:
                return (int(ctx.neighbours[int(np.argmax(ctx.neighbour_distances))]),)
            # Split the neighbourhood by the sign of the along-road offset and
            # take the farthest on each side.
            axis = ctx.neighbour_dx if np.ptp(ctx.neighbour_dx) >= np.ptp(ctx.neighbour_dy) \
                else ctx.neighbour_dy
            chosen: list[int] = []
            for side in (axis > 0, axis < 0):
                if side.any():
                    local = np.flatnonzero(side)
                    best = local[int(np.argmax(ctx.neighbour_distances[local]))]
                    chosen.append(int(ctx.neighbours[best]))
            return tuple(chosen)

        progress = ctx.progress()
        ahead = progress > 0.0
        if not ahead.any():
            # Nothing beyond us: the message has reached the end of the chain
            # in this direction.
            return ()
        local = np.flatnonzero(ahead)
        best = local[int(np.argmax(progress[local]))]
        return (int(ctx.neighbours[best]),)

    # --------------------------------------------------------------- decide --
    def decide(self, ctx: DecisionContext) -> Action:
        originating = ctx.trigger is Trigger.ORIGINATE

        if originating or ctx.was_designated:
            relays = self.select_relays(ctx)
            if not relays:
                # No eligible relay: still transmit, so nearby vehicles are
                # warned even though the chain stops here.
                return BROADCAST_NOW
            return Action(ActionType.RELAY, relay_indices=relays)

        if not self.fallback_enabled:
            return SUPPRESS

        # Implicit-ACK fallback: wait past the designated relay's turn, ranked
        # by distance so the best alternative speaks first, and cancel the
        # moment the designated relay (or any other) is overheard.
        slot = distance_slot(ctx.sender_distance_m, ctx.comm_range_m, self.fallback_n_slots)
        return Action(
            ActionType.DEFER,
            delay_steps=self.fallback_epochs + slot,
            cancel_on_duplicates=1,
        )
