"""Phase 4 baseline comparison.

::

    python -m experiments.compare                      # configured densities
    python -m experiments.compare --density 20 --seeds 10
    python -m experiments.compare --scenario urban_grid --weather heavy_rain

Runs every registered policy over the same seeds and reports mean +- std. The
comparison is **paired**: seed *k* gives every policy byte-identical mobility,
hazard placement, shadowing and fading, so differences between policies are
differences in the decision rule and nothing else. That is what makes the
Wilcoxon signed-rank test in Phase 7 legitimate, and this runner already emits
paired samples in the right shape for it.

Rows are appended to ``results/runs.csv`` keyed by config hash.
"""

from __future__ import annotations

import argparse
import logging
from typing import Any

import numpy as np

from agents.registry import BASELINE_POLICIES, available_policies
from analysis.metrics import METRIC_DIRECTION
from common.config import load_yaml
from common.logging_utils import get_logger
from experiments.results_io import append_runs
from experiments.run_sim import RunSpec, run_single

logger = get_logger("experiments.compare")


def run_cell(
    policies: list[str],
    seeds: list[int],
    *,
    scenario: str,
    density: float,
    weather: str,
    hazard_type: str | None,
    duration_s: float | None = None,
    corridor_length_m: float | None = None,
    cfgs: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Every (policy, seed) for one cell of the factorial."""
    cfgs = cfgs or {}
    phy_cfg = cfgs.get("phy") or load_yaml("phy.yaml")
    hz_cfg = cfgs.get("hazard") or load_yaml("hazard.yaml")
    exp_cfg = cfgs.get("experiment") or load_yaml("experiment.yaml")

    rows: list[dict[str, Any]] = []
    for policy in policies:
        for seed in seeds:
            spec = RunSpec(
                scenario=scenario, density_veh_km_lane=density, weather=weather,
                policy=policy, seed=seed, hazard_type=hazard_type,
                duration_s=duration_s, corridor_length_m=corridor_length_m,
            )
            m, _ = run_single(spec, phy_cfg=phy_cfg, hz_cfg=hz_cfg, exp_cfg=exp_cfg)
            rows.append(m)
    return rows


def _agg(rows: list[dict[str, Any]], policy: str, metric: str) -> tuple[float, float]:
    vals = np.array(
        [r[metric] for r in rows if r["policy"] == policy and metric in r], dtype=float
    )
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan"), float("nan")
    return float(vals.mean()), float(vals.std())


def format_table(rows: list[dict[str, Any]], metrics: list[str], title: str) -> str:
    """Console comparison table: mean +- std per policy, best value starred."""
    policies = [p for p in BASELINE_POLICIES if any(r["policy"] == p for r in rows)]
    short = {
        "rwcr": "RWCR", "tir_median_s": "TIRp50", "tir_p95_s": "TIRp95",
        "deadline_miss_rate": "MissRate", "transmissions": "Tx", "pdr": "PDR",
        "redundancy_ratio": "Redund", "at_risk_coverage": "AtRiskCov",
        "latency_mean_s": "Lat", "collisions_per_delivered": "Coll/Del",
    }
    widths = {m: max(len(short.get(m, m)), 15) for m in metrics}

    header = f"{'policy':<18}" + "".join(f"{short.get(m, m):>{widths[m]}}" for m in metrics)
    lines = ["=" * len(header), f" {title}", "=" * len(header), header, "-" * len(header)]

    stats = {p: {m: _agg(rows, p, m) for m in metrics} for p in policies}
    best: dict[str, str] = {}
    for m in metrics:
        direction = METRIC_DIRECTION.get(m, 1)
        finite = {p: stats[p][m][0] for p in policies if np.isfinite(stats[p][m][0])}
        if finite:
            best[m] = (max if direction > 0 else min)(finite, key=finite.get)

    for p in policies:
        cells = []
        for m in metrics:
            mu, sd = stats[p][m]
            if not np.isfinite(mu):
                cells.append(f"{'n/a':>{widths[m]}}")
                continue
            prec = 0 if m in ("transmissions",) else (2 if abs(mu) >= 10 else 3)
            txt = f"{mu:.{prec}f}+-{sd:.{prec}f}"
            if best.get(m) == p:
                txt = "*" + txt
            cells.append(f"{txt:>{widths[m]}}")
        lines.append(f"{p:<18}" + "".join(cells))

    lines.append("=" * len(header))
    lines.append(" * = best mean for that metric.  n = %d seeds, paired across policies."
                 % len({r["seed"] for r in rows}))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="AI-HARP Phase 4 baseline comparison")
    ap.add_argument("--scenario", default=None, choices=["rural_highway", "urban_grid"])
    ap.add_argument("--weather", default=None)
    ap.add_argument("--hazard", default=None)
    ap.add_argument("--density", type=float, action="append", default=None,
                    help="repeatable; defaults to configs/experiment.yaml compare block")
    ap.add_argument("--policies", nargs="*", default=None, choices=available_policies())
    ap.add_argument("--seeds", type=int, default=None, help="number of seeds (0..n-1)")
    ap.add_argument("--duration", type=float, default=None)
    ap.add_argument("--no-write", action="store_true", help="do not touch results/runs.csv")
    ap.add_argument("--quiet", action="store_true", help="suppress per-run logging")
    args = ap.parse_args(argv)

    if args.quiet:
        logging.getLogger("aiharp").setLevel(logging.WARNING)

    exp_cfg = load_yaml("experiment.yaml")
    cmp_cfg = exp_cfg["compare"]
    cfgs = {"phy": load_yaml("phy.yaml"), "hazard": load_yaml("hazard.yaml"),
            "experiment": exp_cfg}

    scenario = args.scenario or cmp_cfg["scenario"]
    weather = args.weather or cmp_cfg["weather"]
    hazard = args.hazard or cmp_cfg["hazard_type"]
    densities = args.density or cmp_cfg["densities_veh_km_lane"]
    policies = list(args.policies) if args.policies else list(BASELINE_POLICIES)
    seeds = list(range(args.seeds)) if args.seeds else list(cmp_cfg["seeds"])
    metrics = list(cmp_cfg["headline_metrics"])

    all_rows: list[dict[str, Any]] = []
    for density in densities:
        rows = run_cell(
            policies, seeds, scenario=scenario, density=density, weather=weather,
            hazard_type=hazard, duration_s=args.duration, cfgs=cfgs,
        )
        all_rows.extend(rows)
        title = (f"{scenario} | {density:g} veh/km/lane | {weather} | {hazard} "
                 f"| {len(seeds)} seeds")
        print()
        print(format_table(rows, metrics, title))

    if not args.no_write:
        append_runs(all_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
