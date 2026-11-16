"""STRUCT: a small structured learned dissemination policy that contains slotted_1p and RAR exactly (revision 8).

Pre-registered in ``results/revision8/ADDENDUM.md`` (section 2). Registry name ``struct``.

Decision rule (one decision per vehicle per message, on first reception)::

    ORIGINATE                -> BROADCAST_NOW
    own_tx_count > 0         -> SUPPRESS
    otherwise                   u = ctx.rng.random()                  (always exactly one draw, RAR's convention)
                                p = clip(w_p . x - 0.15 * delta, 0, 1)
                                u >= p -> SUPPRESS
                                else DEFER(delay = 1 + clip(round(w_d . x'), 0, 50), cancel_on_duplicates = k)
                                     k = 1 + argmax(W_k . x)  in {1, 2, 3}  (ties -> smaller k)

    x  = (1, rho_c, min(n, 30) / 30, min(max(d, 0), R) / R)
    x' = (x, s50),  s50 = distance_slot(d, R, 50)  (agents/persistence.py)

with rho_c = ctx.relevance (causal), n = ctx.n_neighbours, d = ctx.sender_distance_m, R = ctx.comm_range_m. All four
exist in the Veins bridge request (results/revision8/READING_NOTES.md section 6). ``wants_duplicate_callbacks`` is False:
duplicates act only through the engine's cancel counter, as for the slotted baselines. ``delta`` is the evaluation knob
(suppression bias, delta in {-2..3}; 0 in training).

**Parameter count: 21** (w_p 4 + w_d 5 + W_k 12), below the limit of 40.

Why clipped-linear heads, not sigmoids. The revision-8 instruction specified p = sigmoid(w.x) and
delay = 1 + round(50 * sigmoid(w.x')). Neither can contain the baselines exactly
(results/revision8/probe_heads.json, 1,698 RAR contexts on training-pool seeds): the best-fit sigmoid p head
mismatches RAR in 72 decisions, and a sigmoid delay head with one slot term mismatches slotted_1p n=5 in >= 484
decisions (n=50: 844 with the plain slot index). The clipped-linear heads mismatch in 0. Author decision, recorded in
ADDENDUM.md section 1.

Exact containment (tests/test_structured.py):
* slotted_1p(n_slots=50): w_p = (1,0,0,0), w_d = (0,0,0,0,1), k = 1.
* slotted_1p(n_slots=5):  w_d = (-0.45,0,0,0,0.1) (round(s50/10 - 0.45) = floor(s50/10) = floor(5(1-d/R))).
  Decisions are identical; STRUCT draws one uniform that slotted_1p (p = 1) does not, but nothing except the policy reads
  the policy stream, so whole engine runs are byte-identical (tested).
* RAR(a, b, c): w_p = (b + c, a, -b, 0), w_d as for n_slots=5, k = 1: identical decisions and identical draws.
* Generally: slotted_1p n in {1, 2, 5, 50} (n divides 50) and all 27 RAR settings.

Not exactly representable, and why:
* counter_based: its delay is random, 1 + U{0..4} (agents/counter.py:66), drawn with ``rng.integers``; STRUCT's delay is
  deterministic and it draws one uniform. Its distance test (p = 0 below 0.35 R, else 1) is representable with
  w_p = (-0.35 * M, 0, 0, M) only up to the measure-zero boundary, but the random delay is not.
* ETSI CBF: its destination-area test uses the absolute x coordinate, which is not a STRUCT input. Inside its area at
  the 100 ms epoch every timeout rounds to one epoch, so CBF equals slotted_1p with n_slots = 1, which STRUCT represents.
* slotted_1p n in {3, 8, 12, 20, 30}: floor(n(1 - d/R)) is not a function of the 50-slot index.
"""

from __future__ import annotations

import json
from typing import Any, Sequence

import numpy as np

from agents.base import BROADCAST_NOW, SUPPRESS, Action, ActionType, DecisionContext, Policy, Trigger
from agents.persistence import distance_slot

N_MAX = 30.0
D_MAX = 50
DELTA_STEP = 0.15
N_P, N_D, N_K = 4, 5, 12
N_PARAMS = N_P + N_D + N_K          # 21


def split(theta: Sequence[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = np.asarray(theta, dtype=float)
    if t.shape != (N_PARAMS,):
        raise ValueError(f"STRUCT expects {N_PARAMS} parameters, got shape {t.shape}")
    return t[:N_P], t[N_P:N_P + N_D], t[N_P + N_D:].reshape(3, N_P)


def features(ctx: DecisionContext) -> tuple[np.ndarray, float]:
    """x and the 50-slot index s50 for one context."""
    R = float(ctx.comm_range_m)
    d = float(ctx.sender_distance_m)
    dr = (min(max(d, 0.0), R) / R) if R > 0 else 0.0
    x = np.array([1.0, float(ctx.relevance), min(float(ctx.n_neighbours), N_MAX) / N_MAX, dr])
    return x, float(distance_slot(d, R, D_MAX))


def theta_slotted(n_slots: int) -> np.ndarray:
    """Equivalence parameters for slotted_1p(p=1, n_slots in {1, 2, 5, 50}, slot_epochs=1, cancel=1)."""
    if D_MAX % int(n_slots):
        raise ValueError(f"n_slots={n_slots} is not exactly representable (must divide {D_MAX})")
    m = int(n_slots) / D_MAX
    w_p = [1.0, 0.0, 0.0, 0.0]
    w_d = [0.0, 0.0, 0.0, 0.0, 1.0] if m == 1 else [-0.5 + 0.5 * m, 0.0, 0.0, 0.0, m]
    W_k = [1.0, 0.0, 0.0, 0.0] + [0.0] * 8
    return np.array(w_p + w_d + W_k, dtype=float)


def theta_rar(a: float, b: float, c: float) -> np.ndarray:
    """Equivalence parameters for RAR(abc = (a, b, c), n_max = 30, n_slots = 5, cancel = 1)."""
    t = theta_slotted(5)
    t[:N_P] = [b + c, a, -b, 0.0]
    return t


class StructuredPolicy(Policy):
    name = "struct"
    uses_timers = True
    wants_duplicate_callbacks = False

    def __init__(self, theta: Any = None, delta: float = 0.0) -> None:
        if theta is None:
            theta = theta_slotted(5)
        if isinstance(theta, str):
            theta = json.loads(theta)
        th = [float(v) for v in theta]
        super().__init__(theta=th, delta=float(delta))
        self.w_p, self.w_d, self.W_k = split(th)
        self.delta = float(delta)

    def probability(self, x: np.ndarray) -> float:
        return float(np.clip(float(self.w_p @ x) - DELTA_STEP * self.delta, 0.0, 1.0))

    def delay(self, x: np.ndarray, s50: float) -> int:
        z = float(self.w_d[:N_P] @ x) + float(self.w_d[N_P]) * s50
        return 1 + int(np.clip(np.round(z), 0, D_MAX))

    def cancel(self, x: np.ndarray) -> int:
        return 1 + int(np.argmax(self.W_k @ x))

    def decide(self, ctx: DecisionContext) -> Action:
        if ctx.trigger is Trigger.ORIGINATE:
            return BROADCAST_NOW
        if ctx.own_tx_count > 0:
            return SUPPRESS
        x, s50 = features(ctx)
        u = ctx.rng.random()
        if u >= self.probability(x):
            return SUPPRESS
        return Action(ActionType.DEFER, delay_steps=self.delay(x, s50), cancel_on_duplicates=self.cancel(x))
