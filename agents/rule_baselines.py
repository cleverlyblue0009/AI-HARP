"""Revision-5 non-learning baselines with the learned policy's information (C4, C5).

Both are pre-registered in results/revision5/protocol.json (rules and grids) and go through the
same SEL -> EVAL selection protocol as every other method.

RAR -- relevance-aware rule
    Uses exactly the causal inputs the learned policy sees: the holder's causal relevance rho_c,
    its neighbour count n, its distance to the sender (slot ordering) and the message age (the
    engine's TTL applies to every policy). On first reception it rebroadcasts with probability
    p = clip(a*rho_c + b*(1 - n/n_max) + c, 0, 1), n_max = 30, deferred by distance into 5 slots
    (farthest first) and cancelled on hearing a duplicate, as slotted 1-persistence.

SCF-slotted -- store-carry-forward slotted
    Slotted 1-persistence when the holder has neighbours. When it has none it CARRIES the message
    and re-checks every second; it rebroadcasts as soon as it has a neighbour again, and gives up
    after ``carry_timeout_s``. Hearing a duplicate while carrying means the gap has closed: stop.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from agents.base import (BROADCAST_NOW, SUPPRESS, Action, ActionType, DecisionContext, Policy,
                         Trigger)
from agents.persistence import distance_slot


def _triple(v: Any, n: int) -> list[float]:
    vals = json.loads(v) if isinstance(v, str) else list(v)
    if len(vals) != n:
        raise ValueError(f"expected {n} values, got {vals!r}")
    return [float(x) for x in vals]


class RelevanceAwareRule(Policy):
    name = "rar"

    def __init__(self, abc: Any = (1.0, 0.5, 0.0), n_max: float = 30.0, n_slots: int = 5,
                 cancel_on_duplicates: int = 1) -> None:
        a, b, c = _triple(abc, 3)
        super().__init__(abc=[a, b, c], n_max=n_max, n_slots=n_slots,
                         cancel_on_duplicates=cancel_on_duplicates)
        self.a, self.b, self.c = a, b, c
        self.n_max = float(n_max)
        self.n_slots = int(n_slots)
        self.cancel_on_duplicates = int(cancel_on_duplicates)

    def rebroadcast_probability(self, ctx: DecisionContext) -> float:
        density_term = 1.0 - min(ctx.n_neighbours, self.n_max) / self.n_max
        return float(np.clip(self.a * ctx.relevance + self.b * density_term + self.c, 0.0, 1.0))

    def decide(self, ctx: DecisionContext) -> Action:
        if ctx.trigger is Trigger.ORIGINATE:
            return BROADCAST_NOW
        if ctx.own_tx_count > 0:
            return SUPPRESS
        if ctx.rng.random() >= self.rebroadcast_probability(ctx):
            return SUPPRESS
        slot = distance_slot(ctx.sender_distance_m, ctx.comm_range_m, self.n_slots)
        return Action(ActionType.DEFER, delay_steps=1 + slot,
                      cancel_on_duplicates=self.cancel_on_duplicates)


class StoreCarryForwardSlotted(Policy):
    name = "scf_slotted"
    wants_duplicate_callbacks = True

    def __init__(self, slots_timeout: Any = (5, 5.0), carry_recheck_epochs: int = 10,
                 cancel_on_duplicates: int = 1) -> None:
        n_slots, timeout = _triple(slots_timeout, 2)
        super().__init__(slots_timeout=[int(n_slots), timeout], carry_recheck_epochs=carry_recheck_epochs,
                         cancel_on_duplicates=cancel_on_duplicates)
        self.n_slots = int(n_slots)
        self.carry_timeout_s = float(timeout)
        self.carry_recheck_epochs = int(carry_recheck_epochs)
        self.cancel_on_duplicates = int(cancel_on_duplicates)
        self._carry_start: dict[int, float] = {}

    def reset(self, n_vehicles: int, rng: np.random.Generator) -> None:
        super().reset(n_vehicles, rng)
        self._carry_start = {}

    def decide(self, ctx: DecisionContext) -> Action:
        if ctx.trigger is Trigger.ORIGINATE:
            return BROADCAST_NOW
        if ctx.own_tx_count > 0:
            return SUPPRESS
        i = int(ctx.index)
        if ctx.trigger is Trigger.RECEIVE and ctx.is_duplicate:
            self._carry_start.pop(i, None)
            return SUPPRESS                       # covered by someone else / gap closed
        if ctx.n_neighbours > 0:
            if ctx.trigger is Trigger.TIMER:      # a carried message meets a neighbour: send now
                self._carry_start.pop(i, None)
                return BROADCAST_NOW
            slot = distance_slot(ctx.sender_distance_m, ctx.comm_range_m, self.n_slots)
            return Action(ActionType.DEFER, delay_steps=1 + slot,
                          cancel_on_duplicates=self.cancel_on_duplicates)
        start = self._carry_start.setdefault(i, ctx.time_s)
        if ctx.time_s - start >= self.carry_timeout_s:
            self._carry_start.pop(i, None)
            return SUPPRESS
        return Action(ActionType.CARRY, delay_steps=self.carry_recheck_epochs)
