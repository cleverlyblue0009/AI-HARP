"""Constrained training objective: minimise transmissions subject to coverage.

Why the weighted-sum reward was replaced
----------------------------------------
It ranked near-total silence above every working scheme. On rural d=40,
evaluation seed 0: always-suppress scored +1.5 total reward at RWCR 0.020,
slotted_1p -829 at RWCR 0.954, flooding -9301. Its coverage and miss terms were
divided by the at-risk count and added to EVERY vehicle, so total failure
versus near-perfect coverage moved the reward by 0.016 per vehicle, against
>= 1.33 per transmission. Training run3 found that optimum (RWCR 0.034).

The objective
-------------
Per episode, with ``n`` the causal at-risk count and ``C`` causal coverage::

    maximise   -(transmissions) / n          -- the paper's headline cost axis
    subject to  C >= C*                      -- a per-cell coverage target

solved with a Lagrange multiplier::

    L      = -tx / n + lambda * (C - C*)
    lambda <- clip(lambda + eta * mean_e[(C*_e - C_e) / C*_e], 0, lambda_max)

A coverage shortfall raises lambda until relaying pays; a surplus lowers it
until suppression pays. Silence cannot be a resting point: it leaves C far
below C*, so lambda keeps rising.

Per-vehicle credit: a Shapley split along the dissemination tree
----------------------------------------------------------------
A multiplier alone does not fix the old failure, because the old coverage term
was spread evenly over all vehicles, silent or not; scaling a diluted term by
lambda leaves every relay's individual stake at noise level.

A vehicle ``u`` is warned only if EVERY vehicle on its ancestor path in the
``informed_by`` tree transmitted -- parent, grandparent, ..., originator. That
is a unanimity game, and the Shapley value of a unanimity game is an equal
share for each member. So each warned at-risk vehicle's relevance weight is
split equally along its ancestor path, giving

    sum_v credit_v = (coverage mass of warned, non-originator vehicles)

-- no over-counting -- and each relay a stake proportional to the coverage it
enabled downstream, not only the vehicles it reached directly. The per-vehicle
reward is

    r_v = -(tx_v + w_coll * coll_v) + lambda * credit_v

whose episode sum is ``n * L`` up to a constant, in units where one
transmission costs 1.

Causal only
-----------
Everything here uses the causal risk field; training may not see the oracle.
Coverage targets are therefore expressed in causal units
(``experiments/coverage_targets.py``), while evaluation still reports oracle
RWCR. Torch-free, so it is testable on any interpreter.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from common.logging_utils import get_logger

logger = get_logger("agents.constrained_reward")

NO_PARENT = -1


@dataclass
class ConstrainedObjective:
    """From ``configs/agent.yaml -> objective``."""

    lambda_init: float = 2.0
    lambda_lr: float = 1.0
    lambda_max: float = 50.0
    target_fraction: float = 0.95
    fallback_target: float = 0.80
    collision_weight: float = 0.0
    targets_path: str = "results/coverage_targets.json"

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "ConstrainedObjective":
        o = cfg.get("objective", {})
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in o.items() if k in known})


@dataclass
class EpisodeOutcome:
    """What one episode did against its constraint."""

    coverage: float
    target: float
    n_at_risk: int
    transmissions: int
    cost_per_at_risk: float
    shortfall: float            # (target - coverage) / target; negative = surplus
    credit_total: float

    def as_dict(self) -> dict[str, float]:
        return {k: float(v) for k, v in self.__dict__.items()}


# ---------------------------------------------------------------------------
# Coverage and credit (array level; no simulator objects needed)
# ---------------------------------------------------------------------------
def warned_at_risk(
    relevance: np.ndarray, informed_step: np.ndarray, at_risk_threshold: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(peak, at_risk, warned)`` from a causal ``[T, N]`` relevance matrix.

    ``warned`` means informed while still at risk -- the same indicator RWCR
    uses, so a warning delivered after the vehicle passed the hazard earns
    nothing.
    """
    peak = relevance.max(axis=0)
    at_risk = peak > at_risk_threshold
    informed = informed_step >= 0
    idx = np.flatnonzero(informed)
    rel_at_info = np.zeros(relevance.shape[1])
    rel_at_info[idx] = relevance[informed_step[idx], idx]
    warned = informed & at_risk & (rel_at_info > 0.0)
    return peak, at_risk, warned


def causal_coverage(peak: np.ndarray, at_risk: np.ndarray, warned: np.ndarray) -> float:
    """Relevance-weighted fraction of the at-risk set warned in time."""
    mass = float(peak[at_risk].sum())
    if mass <= 0:
        return float("nan")
    return float((peak * (warned & at_risk)).sum() / mass)


def ancestor_path(informed_by: np.ndarray, vehicle: int) -> list[int]:
    """Every transmitter ``vehicle``'s warning depended on, nearest first."""
    path: list[int] = []
    seen: set[int] = set()
    p = int(informed_by[vehicle])
    while p != NO_PARENT and p >= 0 and p not in seen:
        path.append(p)
        seen.add(p)
        p = int(informed_by[p])
    return path


def tree_credit(
    informed_by: np.ndarray, peak: np.ndarray, at_risk: np.ndarray, warned: np.ndarray
) -> np.ndarray:
    """Shapley credit per vehicle, in at-risk-vehicle equivalents.

    Each warned at-risk vehicle contributes ``peak / mean_peak`` split equally
    over its ancestor path. The originator's own coverage (it has no
    ancestors -- it detected the hazard) is assigned to nobody, because no
    dissemination decision produced it.
    """
    n = len(informed_by)
    credit = np.zeros(n)
    if not at_risk.any():
        return credit
    mean_w = float(peak[at_risk].mean())
    if mean_w <= 0:
        return credit
    for u in np.flatnonzero(warned & at_risk):
        path = ancestor_path(informed_by, int(u))
        if not path:
            continue
        share = (float(peak[u]) / mean_w) / len(path)
        for a in path:
            credit[a] += share
    return credit


# ---------------------------------------------------------------------------
# Episode rewards
# ---------------------------------------------------------------------------
def episode_rewards(
    result: Any, risk: Any, hazard: Any, lam: float, objective: ConstrainedObjective,
    target: float,
) -> tuple[dict[int, float], EpisodeOutcome]:
    """Per-vehicle rewards under the constrained objective, and the outcome."""
    rel = risk.relevance_matrix(result.trace, hazard)          # CAUSAL
    peak, at_risk, warned = warned_at_risk(rel, result.informed_step,
                                           risk.at_risk_threshold)
    n_ar = int(at_risk.sum())
    cov = causal_coverage(peak, at_risk, warned)
    credit = tree_credit(result.informed_by, peak, at_risk, warned)

    tx = result.tx_count.astype(float)
    coll = (np.asarray(result.collisions_caused, dtype=float)
            if getattr(result, "collisions_caused", None) is not None
            else np.zeros_like(tx))
    r = -(tx + objective.collision_weight * coll) + float(lam) * credit

    shortfall = ((target - cov) / target
                 if np.isfinite(cov) and target > 0 else 0.0)
    outcome = EpisodeOutcome(
        coverage=cov, target=float(target), n_at_risk=n_ar,
        transmissions=int(result.n_transmissions),
        cost_per_at_risk=float(tx.sum() / max(n_ar, 1)),
        shortfall=float(shortfall), credit_total=float(credit.sum()),
    )
    return {v: float(r[v]) for v in range(len(r))}, outcome


# ---------------------------------------------------------------------------
# The multiplier
# ---------------------------------------------------------------------------
@dataclass
class LagrangeMultiplier:
    """Dual ascent on the coverage constraint."""

    value: float
    lr: float
    max_value: float
    history: list[float] = field(default_factory=list)

    @classmethod
    def from_objective(cls, o: ConstrainedObjective) -> "LagrangeMultiplier":
        return cls(value=float(o.lambda_init), lr=float(o.lambda_lr),
                   max_value=float(o.lambda_max))

    def update(self, shortfalls: Iterable[float]) -> float:
        s = [float(x) for x in shortfalls if np.isfinite(x)]
        if s:
            self.value = float(np.clip(self.value + self.lr * np.mean(s),
                                       0.0, self.max_value))
        self.history.append(self.value)
        return self.value

    @property
    def saturated(self) -> bool:
        """At the cap: the constraint is not being met even at maximum price."""
        return self.value >= self.max_value - 1e-12

    def state_dict(self) -> dict[str, float]:
        return {"value": self.value, "lr": self.lr, "max_value": self.max_value}


# ---------------------------------------------------------------------------
# Per-cell coverage targets
# ---------------------------------------------------------------------------
class CoverageTargets:
    """``target_fraction`` x the best reference coverage measured in each cell.

    A single global target is infeasible in sparse cells (the best baseline
    reaches oracle RWCR 0.689 at rural d=2) and would drive lambda -- and the
    policy -- toward flooding there. Cells missing from the table fall back to
    ``fallback_target`` with a warning, so a gap is visible.
    """

    def __init__(self, table: dict[str, dict[str, Any]], fraction: float, fallback: float):
        self.table = table
        self.fraction = float(fraction)
        self.fallback = float(fallback)
        self._warned: set[str] = set()

    @staticmethod
    def key(scenario: str, density: float, weather: str, hazard_type: str) -> str:
        return f"{scenario}|{float(density):g}|{weather}|{hazard_type}"

    def target(self, scenario: str, density: float, weather: str, hazard_type: str) -> float:
        k = self.key(scenario, density, weather, hazard_type)
        entry = self.table.get(k)
        if entry is None or not np.isfinite(entry.get("ceiling", np.nan)):
            if k not in self._warned:
                logger.warning("No measured coverage ceiling for %s; using fallback "
                               "target %.2f", k, self.fallback)
                self._warned.add(k)
            return self.fallback
        return self.fraction * float(entry["ceiling"])

    @classmethod
    def load(cls, path: Path | str, fraction: float, fallback: float) -> "CoverageTargets":
        p = Path(path)
        if not p.exists():
            logger.warning("Coverage targets file %s not found; every cell uses the "
                           "fallback target %.2f", p, fallback)
            return cls({}, fraction, fallback)
        payload = json.loads(p.read_text(encoding="utf-8"))
        return cls(payload.get("cells", payload), fraction, fallback)


# ---------------------------------------------------------------------------
# What the trainer holds
# ---------------------------------------------------------------------------
@dataclass
class TrainingObjective:
    """Objective settings, the live multiplier and the per-cell targets."""

    objective: ConstrainedObjective
    multiplier: LagrangeMultiplier
    targets: CoverageTargets

    def target_for(self, scenario: str, density: float, weather: str,
                   hazard_type: str) -> float:
        return self.targets.target(scenario, density, weather, hazard_type)

    def update(self, shortfalls: Iterable[float]) -> float:
        """One dual-ascent step from a whole PPO update's episodes."""
        return self.multiplier.update(shortfalls)

    def state_dict(self) -> dict[str, Any]:
        return {"multiplier": self.multiplier.state_dict(),
                "objective": dict(self.objective.__dict__)}


def build_training_objective(cfg: dict[str, Any], project_root: Path | str) -> TrainingObjective:
    """Construct from ``configs/agent.yaml``; refuses anything but ``constrained``.

    The weighted-sum reward still sits in the config, marked broken, for the
    record. Refusing it here means it cannot be trained on by accident.
    """
    kind = cfg.get("objective", {}).get("kind")
    if kind != "constrained":
        raise ValueError(
            f"objective.kind is {kind!r}; only 'constrained' may be trained on. The "
            "weighted-sum reward ranks silence above every working scheme."
        )
    o = ConstrainedObjective.from_config(cfg)
    targets = CoverageTargets.load(Path(project_root) / o.targets_path,
                                   o.target_fraction, o.fallback_target)
    return TrainingObjective(objective=o, multiplier=LagrangeMultiplier.from_objective(o),
                             targets=targets)
