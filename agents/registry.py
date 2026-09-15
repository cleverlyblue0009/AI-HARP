"""Policy registry: name -> constructor, with defaults from ``configs/policies.yaml``.

Registration is explicit rather than by import-scanning so that a typo in a
sweep config fails loudly with the list of valid names.

Phase 5 registers ``ai_harp`` here alongside the baselines; because the sweep
addresses policies purely by name, nothing else in the codebase needs to change
when it does.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Callable

from agents.base import Policy
from agents.counter import DistanceCounterPolicy
from agents.dvcast import DvCast
from agents.flooding import BlindFlooding
from agents.greedy import GreedyFarthestRelay
from agents.persistence import (
    ProbabilisticPPersistence,
    SlottedPersistence,
    WeightedPPersistence,
)
from common.config import load_yaml

_REGISTRY: dict[str, Callable[..., Policy]] = {
    # 1. Blind flooding
    "flooding": BlindFlooding,
    # 2. Probabilistic p-persistence, at the three p values in the brief
    "p_persistence_03": ProbabilisticPPersistence,
    "p_persistence_05": ProbabilisticPPersistence,
    "p_persistence_07": ProbabilisticPPersistence,
    # 3. Slotted 1-persistence
    "slotted_1p": SlottedPersistence,
    # 4. Weighted p-persistence (distance-weighted)
    "weighted_p": WeightedPPersistence,
    # 5. Distance-based counter scheme
    "counter_based": DistanceCounterPolicy,
    # 6. Greedy farthest-relay selection
    "greedy_farthest": GreedyFarthestRelay,
    # 7. DV-CAST-style store-carry-forward
    "dvcast": DvCast,
}


def _build_ai_harp(**params: Any) -> Policy:
    """Phase 5 agent, constructed lazily.

    Imported inside the factory rather than at module scope for two reasons:
    it breaks the cycle (agents.ai_harp needs build_policy for its analytic
    fallback), and it keeps `import agents.registry` working on an interpreter
    with no torch, so the Phase 1-4 suite is unaffected by Phase 5.
    """
    from agents.ai_harp import AiHarpPolicy

    # A checkpoint path is how a trained agent travels through RunSpec, which
    # only carries serialisable params (and therefore enters the config hash).
    if "checkpoint" in params:
        return AiHarpPolicy.from_checkpoint(**params)
    return AiHarpPolicy(**params)


_REGISTRY["ai_harp"] = _build_ai_harp


def _build_etsi_cbf(**params: Any) -> Policy:
    from agents.cbf import EtsiCbf

    return EtsiCbf(**params)


# Simulator validation only (experiments/validate_amador.py); deliberately NOT
# in BASELINE_POLICIES, so no sweep or comparison picks it up.
_REGISTRY["etsi_cbf"] = _build_etsi_cbf

#: The seven baselines of Phase 4, in the order the brief lists them. Used as
#: the default policy set for sweeps and comparison tables.
BASELINE_POLICIES: tuple[str, ...] = (
    "flooding",
    "p_persistence_03",
    "p_persistence_05",
    "p_persistence_07",
    "slotted_1p",
    "weighted_p",
    "counter_based",
    "greedy_farthest",
    "dvcast",
)


@lru_cache(maxsize=1)
def _defaults() -> dict[str, dict[str, Any]]:
    return load_yaml("policies.yaml")


def register_policy(name: str, ctor: Callable[..., Policy]) -> None:
    if name in _REGISTRY:
        raise KeyError(f"Policy {name!r} is already registered.")
    _REGISTRY[name] = ctor


def available_policies() -> list[str]:
    return sorted(_REGISTRY)


def policy_defaults(name: str) -> dict[str, Any]:
    """Configured default kwargs for a policy (empty if it takes none)."""
    return dict(_defaults().get(name, {}))


def build_policy(name: str, **params: Any) -> Policy:
    """Construct a policy, merging explicit params over the YAML defaults."""
    try:
        ctor = _REGISTRY[name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown policy {name!r}. Available: {available_policies()}"
        ) from exc
    kwargs = {**policy_defaults(name), **params}
    policy = ctor(**kwargs)
    # Registry name wins over the class name, so the three p-persistence
    # variants stay distinguishable in results rows and figures.
    policy.name = name
    return policy
