"""The policy interface every dissemination scheme implements.

One interface covers all of Phase 4's baselines and Phase 5's learned agent, so
they run on byte-identical mobility traces, hazard instances and fading
realisations. Paired comparison is only meaningful if the *only* thing that
differs between two runs is the decision rule.

The engine invokes :meth:`Policy.decide` on three triggers:

``ORIGINATE`` the vehicle has just detected the hazard itself
``RECEIVE``   the vehicle has just decoded the message (first time or duplicate)
``TIMER``     a previously deferred/carried decision has come due

and applies the returned :class:`Action`. A policy holds no simulation state of
its own beyond what it stores per vehicle; everything it needs to decide is in
the :class:`DecisionContext`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class Trigger(str, Enum):
    ORIGINATE = "originate"
    RECEIVE = "receive"
    TIMER = "timer"


class ActionType(str, Enum):
    """The discrete action space (kept small; Phase 5 reuses it verbatim)."""

    SUPPRESS = "suppress"      # drop the message, never rebroadcast
    BROADCAST = "broadcast"    # rebroadcast in this decision epoch
    DEFER = "defer"            # rebroadcast in `delay_steps` epochs, unless cancelled
    CARRY = "carry"            # store-carry-forward: hold and re-evaluate later
    RELAY = "relay"            # broadcast, designating `relay_indices` to relay


@dataclass(frozen=True)
class Action:
    kind: ActionType
    delay_steps: int = 0
    #: Vehicles designated to rebroadcast on receipt. A tuple rather than a
    #: single index because a message originator on a two-way road must seed
    #: both propagation directions at once, and because Phase 5 designates
    #: relays from the top-k attended neighbours.
    relay_indices: tuple[int, ...] = ()
    # Set by deferring policies that cancel on hearing enough duplicates
    # (slotted / weighted p-persistence, counter schemes).
    cancel_on_duplicates: int | None = None

    def __post_init__(self) -> None:
        if self.delay_steps < 0:
            raise ValueError("delay_steps must be >= 0")
        if self.kind is ActionType.RELAY and not self.relay_indices:
            raise ValueError("RELAY requires at least one entry in relay_indices")


SUPPRESS = Action(ActionType.SUPPRESS)
BROADCAST_NOW = Action(ActionType.BROADCAST)


@dataclass
class DecisionContext:
    """Everything a policy may look at when deciding.

    Only locally observable quantities are exposed: a vehicle's own kinematics,
    what it has heard, and the neighbours it can currently hear. Global state
    (who else is informed anywhere in the network) is deliberately *not*
    reachable -- decentralised execution is a requirement, and a baseline that
    peeked at global state would not be a fair comparison.
    """

    step: int
    time_s: float
    dt: float
    index: int
    trigger: Trigger

    # Message state at this vehicle
    hop_count: int
    duplicate_count: int
    #: How many times *this* vehicle has already transmitted this message.
    #: Any policy that can be re-invoked (duplicate callbacks, carry timers)
    #: must check this, or it will relay the same message repeatedly.
    own_tx_count: int
    sender_index: int | None
    sender_distance_m: float
    max_sender_distance_m: float      # farthest sender heard so far (progress)
    age_s: float                      # since the message was originated

    # Own kinematics and risk
    x: float
    y: float
    speed_ms: float
    heading: float
    direction: int
    relevance: float
    eta_s: float
    vclass: str

    # Local neighbourhood (within nominal communication range).
    # Every one of these is obtainable from the CAM/BSM beacons a vehicle
    # already receives -- position, speed and heading are mandatory beacon
    # fields -- so using them costs no extra signalling.
    neighbours: np.ndarray            # [K] vehicle indices
    neighbour_distances: np.ndarray   # [K] metres
    neighbour_dx: np.ndarray          # [K] metres, neighbour minus self
    neighbour_dy: np.ndarray          # [K]
    neighbour_vx: np.ndarray          # [K] m/s
    neighbour_vy: np.ndarray          # [K]
    neighbour_relevance: np.ndarray   # [K]
    neighbour_informed: np.ndarray    # [K] bool -- only what this vehicle has
                                      # actually overheard, not ground truth
    comm_range_m: float
    rng: np.random.Generator

    #: Offset from this vehicle to the sender of the frame that triggered the
    #: decision (the sender's position rides in the frame header, as it does in
    #: ETSI GeoNetworking). Zero on ORIGINATE and TIMER triggers.
    sender_dx: float = 0.0
    sender_dy: float = 0.0
    #: True when an upstream relay named this vehicle as its designated relay.
    was_designated: bool = False
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def n_neighbours(self) -> int:
        return int(self.neighbours.size)

    @property
    def is_duplicate(self) -> bool:
        return self.duplicate_count > 0

    def progress(self) -> np.ndarray:
        """Each neighbour's progress along the message's propagation direction.

        The message travels *away* from the sender, so progress is the
        neighbour's displacement projected onto the unit vector pointing from
        the sender to this vehicle. Positive means the neighbour extends the
        message further; negative means it lies back towards the sender and
        relaying to it would be wasted.

        With no sender (origination), there is no propagation direction yet and
        every neighbour is equally useful, so all progress is zero.
        """
        norm = float(np.hypot(self.sender_dx, self.sender_dy))
        if norm < 1e-9:
            return np.zeros(self.neighbours.size)
        ux, uy = -self.sender_dx / norm, -self.sender_dy / norm
        return self.neighbour_dx * ux + self.neighbour_dy * uy


class Policy(ABC):
    """Base class for every dissemination policy."""

    #: Short name used in results rows and figures.
    name: str = "unnamed"
    #: Whether this policy needs a per-vehicle timer service from the engine.
    uses_timers: bool = True
    #: If True the engine calls :meth:`decide` again on duplicate receptions
    #: (needed by store-carry-forward and by the learned agent, which both
    #: revise their plan when they overhear the message again).
    wants_duplicate_callbacks: bool = False

    def __init__(self, **params: Any) -> None:
        self.params: dict[str, Any] = params

    def reset(self, n_vehicles: int, rng: np.random.Generator) -> None:
        """Called once per run before the first step."""

    @abstractmethod
    def decide(self, ctx: DecisionContext) -> Action:
        """Return the action for this vehicle at this decision epoch."""

    def on_transmit(self, index: int, step: int) -> None:
        """Notification that a scheduled transmission actually went out."""

    def describe(self) -> str:
        if not self.params:
            return self.name
        kv = ", ".join(f"{k}={v}" for k, v in sorted(self.params.items()))
        return f"{self.name}({kv})"
