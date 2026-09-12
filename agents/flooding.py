"""Blind flooding -- the reference baseline (Phase 4, policy 1 of 7).

Every vehicle rebroadcasts exactly once, immediately, the first time it decodes
the message. No suppression, no deferral, no neighbour awareness.

This is the canonical broadcast-storm generator: in a dense network the number
of redundant transmissions grows with the number of informed vehicles, and the
resulting contention and interference destroy the very message being flooded.
It is included not because anyone would deploy it but because it upper-bounds
reachability in sparse networks and upper-bounds overhead in dense ones -- both
ends of the trade-off the paper is about.
"""

from __future__ import annotations

from agents.base import BROADCAST_NOW, Action, DecisionContext, Policy, Trigger


class BlindFlooding(Policy):
    """Rebroadcast once on first reception."""

    name = "flooding"
    uses_timers = False

    def decide(self, ctx: DecisionContext) -> Action:
        # ORIGINATE and the first RECEIVE both produce exactly one rebroadcast.
        # The engine only calls back on duplicates when
        # `wants_duplicate_callbacks` is set, which flooding does not, so this
        # cannot fire twice for one vehicle.
        if ctx.trigger in (Trigger.ORIGINATE, Trigger.RECEIVE):
            return BROADCAST_NOW
        return BROADCAST_NOW
