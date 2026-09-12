"""Probabilistic broadcast-storm mitigation (Phase 4, baselines 2-4).

All three suppress rebroadcasts, and all three differ only in *how they pick
the probability*:

* :class:`ProbabilisticPPersistence` -- a fixed probability ``p``, ignoring
  everything about the vehicle's situation.
* :class:`WeightedPPersistence` -- probability proportional to the distance from
  the sender, so receivers that would add more new coverage are likelier to
  relay.
* :class:`SlottedPersistence` -- deterministic ordering instead of chance: the
  farthest band of receivers waits the fewest slots and everyone nearer cancels
  on hearing it.

References are in ``configs/policies.yaml``; the parameters come from there.

What none of them can do
------------------------
Every one of these ranks candidate relays by **distance from the sender**.
Distance is a proxy for new radio coverage, and it is a reasonable one -- but
it is blind to who actually needs the message. A vehicle 500 m behind the
sender and driving *away* from the hazard is the most attractive relay these
schemes can see, because it maximises geographic progress. That is the gap
the risk field exists to close, and it is why the comparison in the paper is
against schemes that are strong at coverage rather than weak in general.
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


def distance_slot(
    sender_distance_m: float, comm_range_m: float, n_slots: int
) -> int:
    """Slot index for a receiver, farthest band first.

    .. math:: S_{ij} = \\left\\lfloor N_s\\left(1 - \\frac{\\min(D_{ij}, R)}{R}
                       \\right)\\right\\rfloor

    A receiver at the edge of the range gets slot 0 and speaks immediately; one
    right next to the sender gets slot ``N_s - 1`` and will almost certainly
    hear a duplicate first and cancel. Clamped to ``[0, N_s - 1]``.
    """
    if comm_range_m <= 0:
        return 0
    frac = min(max(sender_distance_m, 0.0), comm_range_m) / comm_range_m
    return int(np.clip(int(np.floor(n_slots * (1.0 - frac))), 0, n_slots - 1))


class ProbabilisticPPersistence(Policy):
    """Rebroadcast once with fixed probability ``p``, otherwise drop.

    The simplest storm mitigation there is. Its weakness is structural rather
    than parametric: because ``p`` is fixed, the *expected* number of relays
    scales with the number of receivers, so the value that prevents a storm at
    120 veh/km/lane silently breaks the relay chain at 5 veh/km/lane. No single
    ``p`` works across the density sweep, which is precisely what the sweep is
    for.
    """

    name = "p_persistence"
    uses_timers = False

    def __init__(self, p: float = 0.5) -> None:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1], got {p}")
        super().__init__(p=p)
        self.p = float(p)

    def decide(self, ctx: DecisionContext) -> Action:
        # The originator always speaks: suppressing it would mean the hazard is
        # detected and never announced.
        if ctx.trigger is Trigger.ORIGINATE:
            return BROADCAST_NOW
        return BROADCAST_NOW if ctx.rng.random() < self.p else SUPPRESS


class WeightedPPersistence(Policy):
    """Rebroadcast with probability ``p_ij = D_ij / R`` (distance-weighted).

    Receivers near the edge of the sender's range relay with probability near
    1; receivers right beside the sender almost never do. This concentrates
    relaying where it adds the most new coverage, without any coordination.
    """

    name = "weighted_p"
    uses_timers = False

    def __init__(self, min_p: float = 0.0, max_p: float = 1.0) -> None:
        super().__init__(min_p=min_p, max_p=max_p)
        self.min_p = float(min_p)
        self.max_p = float(max_p)

    def rebroadcast_probability(self, ctx: DecisionContext) -> float:
        if ctx.comm_range_m <= 0:
            return self.max_p
        p = ctx.sender_distance_m / ctx.comm_range_m
        return float(np.clip(p, self.min_p, self.max_p))

    def decide(self, ctx: DecisionContext) -> Action:
        if ctx.trigger is Trigger.ORIGINATE:
            return BROADCAST_NOW
        return (
            BROADCAST_NOW
            if ctx.rng.random() < self.rebroadcast_probability(ctx)
            else SUPPRESS
        )


class SlottedPersistence(Policy):
    """Slotted 1-persistence (``p = 1``) and slotted p-persistence.

    Replaces chance with ordering. The range is split into ``n_slots`` bands;
    a receiver waits ``slot * slot_epochs`` epochs, then rebroadcasts -- unless
    it has overheard ``cancel_on_duplicates`` copies in the meantime, which
    means a better-placed relay has already covered it.

    With ``p = 1`` this is the standard slotted 1-persistence scheme and is the
    strongest distance-based baseline in dense traffic: it gets close to
    one-relay-per-hop without any neighbour table. Its cost is latency -- every
    hop pays the slot wait -- and that cost is what TIR measures.
    """

    name = "slotted_1p"

    def __init__(
        self,
        p: float = 1.0,
        n_slots: int = 5,
        slot_epochs: int = 1,
        cancel_on_duplicates: int = 1,
    ) -> None:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1], got {p}")
        if n_slots < 1:
            raise ValueError("n_slots must be >= 1")
        super().__init__(
            p=p, n_slots=n_slots, slot_epochs=slot_epochs,
            cancel_on_duplicates=cancel_on_duplicates,
        )
        self.p = float(p)
        self.n_slots = int(n_slots)
        self.slot_epochs = int(slot_epochs)
        self.cancel_on_duplicates = int(cancel_on_duplicates)

    def slot_for(self, ctx: DecisionContext) -> int:
        return distance_slot(ctx.sender_distance_m, ctx.comm_range_m, self.n_slots)

    def decide(self, ctx: DecisionContext) -> Action:
        if ctx.trigger is Trigger.ORIGINATE:
            return BROADCAST_NOW
        if self.p < 1.0 and ctx.rng.random() >= self.p:
            return SUPPRESS
        slot = self.slot_for(ctx)
        return Action(
            ActionType.DEFER,
            # +1 because a rebroadcast can never leave in the same epoch as the
            # reception that triggered it.
            delay_steps=1 + slot * self.slot_epochs,
            cancel_on_duplicates=self.cancel_on_duplicates,
        )
