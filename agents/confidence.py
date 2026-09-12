"""Phase 5.3: the confidence gate.

A named contribution, not a safety afterthought
-----------------------------------------------
The paper's safety claim is that a learned policy never *silently* fails in a
safety-critical broadcast. This module is where that claim is earned: the agent
estimates confidence in its own decision, and when confidence falls below
``tau`` the learned action is discarded and a known-good analytic policy
(``weighted_p`` by default) decides instead.

The fallback rate is therefore a first-class reported metric, not a diagnostic.
A gate that never fires proves nothing, and a gate that fires constantly means
the learned policy is not contributing -- both are results the paper must
state. The sweep over ``tau`` in the ablations is what maps that trade-off.

Torch-free by design
--------------------
This module takes a probability vector, not a network. It has no torch import,
so the gate's logic is unit-testable on any interpreter and the policy layer
can use it without pulling in the learning stack.

Two confidence estimators
-------------------------
``entropy``
    Normalised entropy of the action distribution:
    ``confidence = 1 - H(p) / log(n_actions)``. Cheap, needs one forward pass,
    and is exactly the quantity PPO's entropy bonus already regularises -- which
    is why ``entropy_coef`` must not be annealed to zero, or the gate loses its
    signal.

``ensemble``
    Disagreement across an ensemble: ``confidence = 1 - normalised spread of
    the members' action distributions``. More faithful (it separates *this
    model is unsure* from *the situation is genuinely ambiguous*) but costs N
    forward passes per decision, which matters at a 100 ms epoch.

A caveat the paper should state: for the Dueling DQN variant, the softmax over
Q-values is not a calibrated posterior, so its entropy is a dispersion signal
rather than a probability. The gate still works as a heuristic there, but the
calibrated reading belongs to PPO.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

CONFIDENCE_METHODS = ("entropy", "ensemble")


def normalised_entropy_confidence(probs: np.ndarray) -> float:
    """``1 - H(p)/log(n)``: 1.0 for a one-hot decision, 0.0 for uniform."""
    p = np.asarray(probs, dtype=float)
    p = np.clip(p / max(p.sum(), 1e-12), 1e-12, 1.0)
    n = p.size
    if n <= 1:
        return 1.0
    h = float(-(p * np.log(p)).sum())
    return float(np.clip(1.0 - h / math.log(n), 0.0, 1.0))


def ensemble_confidence(member_probs: Sequence[np.ndarray]) -> float:
    """``1 - max per-action spread`` across ensemble members.

    Uses the **maximum** per-action standard deviation, not the mean, scaled by
    the largest attainable spread for a probability (0.5, reached when half the
    members are certain of one action and half of another).

    The mean would dilute with the size of the action space, which is a real
    failure and not a cosmetic one: with nine actions and an ensemble split
    cleanly between two of them, the per-action standard deviations are
    ``[0.5, 0.5, 0, 0, 0, 0, 0, 0, 0]``. Their mean is 0.11, reporting
    confidence 0.78 for an ensemble that could not disagree more sharply. The
    maximum reports 0.0, which is the truth. Taking the max also means adding
    actions to the space never changes the reading.
    """
    arr = np.asarray(member_probs, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 2:
        return normalised_entropy_confidence(arr.reshape(-1))
    spread = float(arr.std(axis=0).max())
    return float(np.clip(1.0 - spread / 0.5, 0.0, 1.0))


@dataclass
class GateDecision:
    """One gate evaluation."""

    confidence: float
    threshold: float
    used_fallback: bool
    method: str

    @property
    def used_learned(self) -> bool:
        return not self.used_fallback


@dataclass
class ConfidenceGate:
    """Threshold a confidence estimate and count how often it fires."""

    tau: float = 0.5
    method: str = "entropy"
    enabled: bool = True
    n_decisions: int = 0
    n_fallbacks: int = 0
    confidences: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.method not in CONFIDENCE_METHODS:
            raise ValueError(
                f"unknown confidence method {self.method!r}; "
                f"expected one of {CONFIDENCE_METHODS}"
            )
        if not 0.0 <= self.tau <= 1.0:
            raise ValueError(f"tau must be in [0, 1], got {self.tau}")

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "ConfidenceGate":
        g = cfg.get("confidence_gate", {})
        return cls(
            tau=float(g.get("tau", 0.5)),
            method=str(g.get("method", "entropy")),
            enabled=bool(g.get("enabled", True)),
        )

    # ------------------------------------------------------------- evaluate --
    def confidence(
        self, probs: np.ndarray, member_probs: Sequence[np.ndarray] | None = None
    ) -> float:
        if self.method == "ensemble" and member_probs is not None:
            return ensemble_confidence(member_probs)
        return normalised_entropy_confidence(probs)

    def evaluate(
        self, probs: np.ndarray, member_probs: Sequence[np.ndarray] | None = None
    ) -> GateDecision:
        """Decide whether the learned action may execute.

        With the gate disabled (or ``tau == 0``) the learned action always
        executes, which is exactly the ``tau = 0`` ablation -- so that ablation
        needs no separate code path.
        """
        c = self.confidence(probs, member_probs)
        self.n_decisions += 1
        self.confidences.append(c)
        fallback = bool(self.enabled and self.tau > 0.0 and c < self.tau)
        if fallback:
            self.n_fallbacks += 1
        return GateDecision(confidence=c, threshold=self.tau,
                            used_fallback=fallback, method=self.method)

    # -------------------------------------------------------------- reporting --
    @property
    def fallback_rate(self) -> float:
        return self.n_fallbacks / self.n_decisions if self.n_decisions else float("nan")

    def stats(self) -> dict[str, float]:
        c = np.asarray(self.confidences, dtype=float)
        return {
            "gate_tau": self.tau,
            "gate_fallback_rate": self.fallback_rate,
            "gate_decisions": float(self.n_decisions),
            "gate_confidence_mean": float(c.mean()) if c.size else float("nan"),
            "gate_confidence_p05": float(np.percentile(c, 5)) if c.size else float("nan"),
            "gate_confidence_p50": float(np.percentile(c, 50)) if c.size else float("nan"),
        }

    def reset(self) -> None:
        self.n_decisions = 0
        self.n_fallbacks = 0
        self.confidences.clear()
