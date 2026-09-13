"""Measure the per-cell coverage ceilings the constrained objective targets.

::

    D:/aiharp-env/python.exe -m experiments.coverage_targets            # all training cells
    D:/aiharp-env/python.exe -m experiments.coverage_targets --quick    # a few cells

For every training cell (scenario x curriculum density x training weather x
training hazard), runs the reference policies on a few training-pool seeds and
records CAUSAL coverage -- the quantity training can see. The ceiling is the
best reference policy's mean. The target used in training is
``objective.target_fraction`` of it.

Reference set: flooding, which maximises broadcast reach, and DV-CAST, which
beats flooding in the sparse regime by carrying messages across gaps (rural d=2:
oracle RWCR 0.666 against 0.607). Using flooding alone would set the sparse
targets too low.

Writes results/coverage_targets.json with provenance.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from typing import Any

import numpy as np

from agents.constrained_reward import CoverageTargets, causal_coverage, warned_at_risk
from agents.registry import build_policy
from common.config import PROJECT_ROOT, load_yaml
from common.logging_utils import get_logger
from common.seeding import SeedBundle
from hazard.model import hazard_from_config
from hazard.risk_field import build_risk_field
from mobility.generate import get_trace, load_scenario
from sim.engine import DisseminationEngine, SimSettings
from sim.mac import build_mac
from sim.phy import build_phy

logger = get_logger("experiments.coverage_targets")

REFERENCE_POLICIES = ("flooding", "dvcast")


def training_cells(cfg: dict[str, Any]) -> list[tuple[str, float, str, str]]:
    t = cfg["training"]
    densities = sorted({float(d) for st in t["curriculum"]["stages"] for d in st["densities"]})
    return [(sc, d, w, h) for sc in t["train_scenarios"] for d in densities
            for w in t["train_weather"] for h in t["train_hazards"]]


def cell_coverage(
    scenario: str, density: float, weather: str, hazard_type: str, seed: int,
    policy: str, cfgs: dict[str, Any],
) -> float:
    scenario_cfg = load_scenario(scenario)
    trace = get_trace(scenario_cfg, density, seed, weather=weather, phy_cfg=cfgs["phy"])
    hazard = hazard_from_config(cfgs["hazard"], trace.meta, overrides={"type": hazard_type})
    phy = build_phy(cfgs["phy"], scenario_cfg["name"], weather, seed, trace_meta=trace.meta)
    mac = build_mac(cfgs["phy"], phy)
    risk = build_risk_field(cfgs["hazard"], trace, hazard)
    res = DisseminationEngine(trace, phy, mac, risk, hazard, build_policy(policy),
                              SeedBundle(master_seed=seed),
                              SimSettings.from_config(cfgs["experiment"])).run()
    rel = risk.relevance_matrix(trace, hazard)
    peak, at_risk, warned = warned_at_risk(rel, res.informed_step, risk.at_risk_threshold)
    return causal_coverage(peak, at_risk, warned)


def build_targets(cfg: dict[str, Any], cfgs: dict[str, Any], n_seeds: int,
                  cells: list[tuple[str, float, str, str]]) -> dict[str, Any]:
    pool = cfg["training"].get("train_seed_pool", {"start": 100, "count": 32})
    seeds = list(range(int(pool["start"]), int(pool["start"]) + n_seeds))
    table: dict[str, Any] = {}
    t0 = time.time()
    for i, (sc, d, w, h) in enumerate(cells, 1):
        per_policy = {}
        for pol in REFERENCE_POLICIES:
            vals = [cell_coverage(sc, d, w, h, s, pol, cfgs) for s in seeds]
            vals = [v for v in vals if np.isfinite(v)]
            per_policy[pol] = float(np.mean(vals)) if vals else float("nan")
        finite = [v for v in per_policy.values() if np.isfinite(v)]
        key = CoverageTargets.key(sc, d, w, h)
        table[key] = {"ceiling": float(max(finite)) if finite else float("nan"),
                      "per_policy": per_policy, "seeds": seeds}
        logger.info("[%d/%d] %s ceiling=%.3f %s (%.0fs)", i, len(cells), key,
                    table[key]["ceiling"], per_policy, time.time() - t0)
    return table


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Measure per-cell coverage ceilings")
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--quick", action="store_true", help="4 cells, 1 seed")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    cfg = load_yaml("agent.yaml")
    cfgs = {"phy": load_yaml("phy.yaml"), "hazard": load_yaml("hazard.yaml"),
            "experiment": load_yaml("experiment.yaml")}
    objective = cfg.get("objective", {})
    cells = training_cells(cfg)
    n_seeds = int(args.seeds or objective.get("target_seeds", 2))
    if args.quick:
        cells, n_seeds = cells[:4], 1

    table = build_targets(cfg, cfgs, n_seeds, cells)
    out = PROJECT_ROOT / (args.out or objective.get("targets_path", "results/coverage_targets.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "provenance": {"reference_policies": list(REFERENCE_POLICIES),
                       "coverage": "causal", "n_seeds": n_seeds,
                       "n_cells": len(cells), "quick": bool(args.quick)},
        "cells": table,
    }, indent=1), encoding="utf-8")
    print(f"wrote {len(table)} cell ceilings -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
