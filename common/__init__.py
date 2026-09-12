"""Shared primitives: config loading, deterministic seeding, logging.

This package is not in the layout sketch in the build brief; it exists so that
``mobility``, ``sim``, ``hazard`` and ``agents`` can share YAML loading and RNG
management without importing each other.
"""

from common.config import (
    CONFIG_DIR,
    PROJECT_ROOT,
    RESULTS_DIR,
    config_hash,
    load_yaml,
    resolve_config_path,
)
from common.logging_utils import get_logger, log_banner
from common.seeding import SeedBundle, make_rng, set_global_determinism

__all__ = [
    "CONFIG_DIR",
    "PROJECT_ROOT",
    "RESULTS_DIR",
    "config_hash",
    "load_yaml",
    "resolve_config_path",
    "get_logger",
    "log_banner",
    "SeedBundle",
    "make_rng",
    "set_global_determinism",
]
