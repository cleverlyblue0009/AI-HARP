"""Distance-based counter scheme (Phase 4, baseline 5).

A hybrid of two schemes from Ni et al. (MobiCom 1999):

* **Counter-based.** Wait a random assessment delay. Every duplicate overheard
  during the wait is evidence that a neighbour has already covered this area.
  If the count reaches ``counter_threshold``, cancel.
* **Distance-based.** If the sender was close, this vehicle's transmission
  would cover almost the same area the sender already did, so relaying adds
  little. Suppress immediately when the sender is nearer than
  ``min_distance_fraction * R``.

The random assessment delay is what makes the counter work: without it every
receiver decides simultaneously, nobody has heard anybody, and the count is
always zero. It also decorrelates the relays' channel access, which reduces
backoff collisions -- a second-order benefit the MAC model captures.
"""

from __future__ import annotations

from agents.base import (
    BROADCAST_NOW,
    SUPPRESS,
    Action,
    ActionType,
    DecisionContext,
    Policy,
    Trigger,
)


class DistanceCounterPolicy(Policy):
    """Counter-based suppression with a distance admission test."""

    name = "counter_based"

    def __init__(
        self,
        counter_threshold: int = 3,
        max_delay_epochs: int = 5,
        min_distance_fraction: float = 0.35,
    ) -> None:
        if counter_threshold < 1:
            raise ValueError("counter_threshold must be >= 1")
        if max_delay_epochs < 1:
            raise ValueError("max_delay_epochs must be >= 1")
        super().__init__(
            counter_threshold=counter_threshold,
            max_delay_epochs=max_delay_epochs,
            min_distance_fraction=min_distance_fraction,
        )
        self.counter_threshold = int(counter_threshold)
        self.max_delay_epochs = int(max_delay_epochs)
        self.min_distance_fraction = float(min_distance_fraction)

    def passes_distance_test(self, ctx: DecisionContext) -> bool:
        """Would rebroadcasting cover meaningfully new ground?"""
        return ctx.sender_distance_m >= self.min_distance_fraction * ctx.comm_range_m

    def decide(self, ctx: DecisionContext) -> Action:
        if ctx.trigger is Trigger.ORIGINATE:
            return BROADCAST_NOW
        if not self.passes_distance_test(ctx):
            return SUPPRESS

        delay = 1 + int(ctx.rng.integers(0, self.max_delay_epochs))
        return Action(
            ActionType.DEFER,
            delay_steps=delay,
            # The counter counts *copies received*, of which the first is the
            # one that triggered this decision; so a threshold of C copies is
            # C - 1 duplicates.
            cancel_on_duplicates=self.counter_threshold - 1,
        )
