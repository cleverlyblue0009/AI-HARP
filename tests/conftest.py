"""Shared fixtures. Adds the project root to sys.path so tests run from anywhere."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.config import load_yaml  # noqa: E402
from sim.mac import build_mac  # noqa: E402
from sim.phy import build_phy  # noqa: E402


@pytest.fixture(scope="session")
def phy_cfg() -> dict:
    return load_yaml("phy.yaml")


@pytest.fixture(scope="session")
def hz_cfg() -> dict:
    return load_yaml("hazard.yaml")


@pytest.fixture(scope="session")
def phy(phy_cfg):
    return build_phy(phy_cfg, scenario="rural_highway", weather="clear", seed=7)


@pytest.fixture(scope="session")
def mac(phy_cfg, phy):
    return build_mac(phy_cfg, phy)
