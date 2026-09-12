"""Deterministic seeding.

Every stochastic component draws from its own named stream derived from one
master seed. Named streams matter for paired comparisons: with the same master
seed, two policies see *identical* mobility, identical hazard placement and
identical fading realisations, so the only difference between them is the
policy. That is what makes the Wilcoxon signed-rank test in Phase 7 legitimate.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

# Stream names used across the codebase. Adding a new one never perturbs the
# existing streams because each is keyed by the hash of its own name.
STREAMS = (
    "mobility",      # vehicle placement, desired speeds, driver noise
    "hazard",        # hazard type/severity/placement sampling
    "shadowing",     # per-link log-normal shadowing
    "fading",        # per-packet Nakagami-m fading
    "mac",           # backoff slot selection / collision draws
    "policy",        # policy-internal randomness (p-persistence, etc.)
    "torch",         # Phase 5 agent init / action sampling
)


def _stream_key(name: str) -> int:
    """Stable 32-bit key for a stream name (hash() is salted per process)."""
    return int.from_bytes(hashlib.sha256(name.encode("utf-8")).digest()[:4], "big")


def make_rng(master_seed: int, stream: str) -> np.random.Generator:
    """Independent PCG64 generator for ``stream`` under ``master_seed``."""
    seq = np.random.SeedSequence(entropy=int(master_seed), spawn_key=(_stream_key(stream),))
    return np.random.Generator(np.random.PCG64(seq))


@dataclass
class SeedBundle:
    """All RNG streams for one run, derived from a single master seed."""

    master_seed: int
    _cache: dict[str, np.random.Generator] = field(default_factory=dict, repr=False)

    def rng(self, stream: str) -> np.random.Generator:
        if stream not in STREAMS:
            raise KeyError(f"Unknown RNG stream {stream!r}; known: {STREAMS}")
        if stream not in self._cache:
            self._cache[stream] = make_rng(self.master_seed, stream)
        return self._cache[stream]

    def fresh(self, stream: str) -> np.random.Generator:
        """A generator for ``stream`` that ignores and resets any cached state."""
        gen = make_rng(self.master_seed, stream)
        self._cache[stream] = gen
        return gen


def set_global_determinism(seed: int) -> None:
    """Best-effort global determinism for Phase 5 (PyTorch is optional)."""
    import os
    import random

    random.seed(seed)
    np.random.seed(seed % (2**32))
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    try:  # pragma: no cover - torch is not a Phase 1-4 dependency
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
