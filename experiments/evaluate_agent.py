"""Phase 7: score a trained AI-HARP checkpoint against the baselines.

::

    D:/aiharp-env/python.exe -m experiments.evaluate_agent \\
        --checkpoint checkpoints/ckpt_latest.pt

The agent goes through the SAME paired path as every baseline -- ``run_single``,
so identical mobility, hazard placement, shadowing and fading per seed -- on the
SAME cells recorded in ``results/pareto_cells.json``. Its suppression bias is
swept to trace an operating curve, exactly as each baseline's knob was, and the
curve is scored by :func:`analysis.comparator.compare_policy`:

* **regret** against the per-cell oracle-best baseline (hindsight tuning);
* **margin** over the best single fixed baseline (what an engineer would ship).

Evaluation seeds are the comparator's (0-9); ``agents.train.check_seed_split``
guarantees the agent never trained on them. The confidence gate is ON here --
it was off during training because training must be on-policy -- and a separate
tau sweep records fallback rate against quality for the gate trade-off figure.

The report states how many updates the checkpoint was trained for. A
checkpoint from a short run is a pipeline check, not a result, and the output
says so rather than leaving it to be inferred.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from analysis.comparator import (
    COST_AXES, DEFAULT_MISS_MARGIN, Cell, compare_policy, load_cells, save_cells,
)
from analysis.pareto import sweep_policy
from common.config import PROJECT_ROOT, RESULTS_DIR, load_yaml
from common.logging_utils import get_logger

logger = get_logger("experiments.evaluate_agent")

DEFAULT_BIASES: tuple[float, ...] = (-2.0, -1.0, 0.0, 1.0, 2.0, 3.0)
DEFAULT_TAUS: tuple[float, ...] = (0.0, 0.1, 0.3, 0.5, 0.7)


def checkpoint_sha(path: Path | str) -> str:
    """Content hash, so a re-trained checkpoint at the same path never shares a
    config hash (and so never de-duplicates against) the old one's results."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def checkpoint_updates(path: Path | str) -> int:
    """How many PPO updates produced this checkpoint (0 if unreadable)."""
    try:
        import torch

        return int(torch.load(str(path), map_location="cpu", weights_only=False)
                   .get("update", 0))
    except Exception:  # pragma: no cover
        return 0


def agent_params(checkpoint: Path | str, **extra: Any) -> dict[str, Any]:
    return {"checkpoint": str(checkpoint), "checkpoint_sha": checkpoint_sha(checkpoint),
            **extra}


def agent_cells(
    checkpoint: Path, cells: Sequence[Cell], seeds: Sequence[int],
    biases: Sequence[float], tau: float, cfgs: dict[str, Any],
) -> list[Cell]:
    """Each cell with an ``ai_harp`` operating curve added beside the baselines."""
    out = []
    for cell in cells:
        k = cell.key
        logger.info("agent curve: %s (%d biases x %d seeds)", k, len(biases), len(seeds))
        curve = sweep_policy(
            "ai_harp", seeds, scenario=k.scenario, density=k.density,
            weather=k.weather, hazard_type=k.hazard_type, cfgs=cfgs,
            sweep=("suppression_bias", tuple(biases)),
            fixed_params=agent_params(checkpoint, tau=tau, deterministic=True),
        )
        out.append(Cell(key=k, curves={**cell.curves, "ai_harp": curve}))
    return out


def gate_sweep(
    checkpoint: Path, cells: Sequence[Cell], seeds: Sequence[int],
    taus: Sequence[float], cfgs: dict[str, Any],
) -> dict[str, list[dict[str, float]]]:
    """Fallback rate and quality against tau, at zero suppression bias."""
    out: dict[str, list[dict[str, float]]] = {}
    for cell in cells:
        k = cell.key
        curve = sweep_policy(
            "ai_harp", seeds, scenario=k.scenario, density=k.density,
            weather=k.weather, hazard_type=k.hazard_type, cfgs=cfgs,
            sweep=("tau", tuple(taus)),
            fixed_params=agent_params(checkpoint, suppression_bias=0.0, deterministic=True),
        )
        out[str(k)] = [{
            "tau": float(p.value), "rwcr": p.quality, "rwcr_std": p.quality_std,
            "cost": p.cost, "tir_median_s": p.extras.get("tir_median_s", float("nan")),
            "fallback_rate": p.extras.get("gate_fallback_rate", float("nan")),
            "confidence_mean": p.extras.get("gate_confidence_mean", float("nan")),
        } for p in sorted(curve.points, key=lambda p: p.value)]
    return out


def format_result(result, title: str) -> str:
    unit = COST_AXES.get(result.axis, result.axis)
    hdr = (f"{'cell':<38}{'agent':>9}{'oracle-best':>24}{'regret':>9}"
           f"{'fixed':>8}{'margin':>9}")
    lines = ["=" * len(hdr), f" {title}", f" {unit}", "=" * len(hdr), hdr, "-" * len(hdr)]
    for key, regret in result.regret_per_cell.items():
        a = result.agent_cost[key]
        o = result.oracle.cost(key)
        f = result.fixed.cost(key)
        m = result.margin_per_cell[key]
        fmt = lambda v, s="": "n/a" if not np.isfinite(v) else f"{v:.2f}{s}"  # noqa: E731
        lines.append(
            f"{str(key):<38}{fmt(a):>9}"
            f"{result.oracle.label(key) + ' ' + fmt(o):>24}"
            f"{fmt(regret * 100, '%') if np.isfinite(regret) else 'n/a':>9}"
            f"{fmt(f):>8}{fmt(m * 100, '%') if np.isfinite(m) else 'n/a':>9}"
        )
    lines += ["-" * len(hdr),
              f" fixed-best baseline : {result.fixed.label}",
              f" mean regret         : {result.mean_regret * 100:+.1f}%  "
              "(0% = matches per-cell hindsight tuning)",
              f" mean margin         : {result.mean_margin * 100:+.1f}%  "
              "(>0% = cheaper than the best fixed baseline)",
              f" cells where the agent never reached the target: {result.cells_failed}",
              " latency guard       : " + ("off (RWCR only)" if result.miss_margin is None
                                           else f"actionable miss <= cell best + "
                                                f"{result.miss_margin:g}"),
              "=" * len(hdr)]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate an AI-HARP checkpoint")
    ap.add_argument("--checkpoint", default="checkpoints/ckpt_latest.pt")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--target", type=float, default=0.95)
    ap.add_argument("--mode", default="relative", choices=["relative", "absolute"])
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--biases", type=float, nargs="*", default=list(DEFAULT_BIASES))
    ap.add_argument("--taus", type=float, nargs="*", default=list(DEFAULT_TAUS))
    ap.add_argument("--cells", type=int, default=None, help="first N cells only")
    ap.add_argument("--skip-gate", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--miss-margin", type=float, default=DEFAULT_MISS_MARGIN,
                    help="latency guard on matched quality: actionable-deadline miss "
                         "rate within this (absolute) of the cell's best baseline")
    ap.add_argument("--no-miss-guard", action="store_true",
                    help="RWCR-only matched quality (the pre-guard definition)")
    ap.add_argument("--out-dir", default=None,
                    help="where to write outputs (default results/); point a "
                         "partially trained checkpoint elsewhere so a pipeline "
                         "check never lands in results/")
    args = ap.parse_args(argv)
    if args.quiet:
        logging.getLogger("aiharp").setLevel(logging.WARNING)

    ckpt = Path(args.checkpoint)
    if not ckpt.is_absolute():
        ckpt = PROJECT_ROOT / ckpt
    if not ckpt.exists():
        print(f"checkpoint not found: {ckpt}")
        return 1

    agent_cfg = load_yaml("agent.yaml")
    tau = float(args.tau if args.tau is not None else agent_cfg["confidence_gate"]["tau"])
    cfgs = {"phy": load_yaml("phy.yaml"), "hazard": load_yaml("hazard.yaml"),
            "experiment": load_yaml("experiment.yaml")}
    cells = load_cells()
    if args.cells:
        cells = cells[: args.cells]
    seeds = list(range(args.seeds))

    updates = checkpoint_updates(ckpt)
    configured = int(agent_cfg["training"]["total_updates"])
    banner = (f"checkpoint {ckpt.name} sha={checkpoint_sha(ckpt)} | trained "
              f"{updates} of {configured} configured updates")
    if updates < configured:
        banner += " -- PARTIAL TRAINING: a pipeline check, not a result"

    out_dir = Path(args.out_dir) if args.out_dir else RESULTS_DIR
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    scored = agent_cells(ckpt, cells, seeds, args.biases, tau, cfgs)
    save_cells(scored, out_dir / "pareto_cells_agent.json")

    summary: dict[str, Any] = {"checkpoint": str(ckpt), "checkpoint_sha": checkpoint_sha(ckpt),
                               "updates": updates, "configured_updates": configured,
                               "tau": tau, "seeds": seeds, "target": args.target,
                               "mode": args.mode, "axes": {}}
    miss_margin = None if args.no_miss_guard else args.miss_margin
    summary["miss_margin"] = miss_margin
    print(banner)
    for axis in COST_AXES:
        res = compare_policy(scored, "ai_harp", args.target, axis, args.mode,
                             miss_margin=miss_margin)
        print()
        print(format_result(res, f"AI-HARP vs baselines -- {axis}, tau={tau}"))
        summary["axes"][axis] = {
            "mean_regret": res.mean_regret, "mean_margin": res.mean_margin,
            "cells_failed": res.cells_failed, "fixed_best": res.fixed.label,
            "per_cell": {str(k): {"agent": res.agent_cost[k],
                                  "oracle": res.oracle.cost(k),
                                  "oracle_policy": res.oracle.label(k),
                                  "fixed": res.fixed.cost(k),
                                  "regret": res.regret_per_cell[k],
                                  "margin": res.margin_per_cell[k]}
                         for k in res.regret_per_cell},
        }

    if not args.skip_gate:
        sweep = gate_sweep(ckpt, cells, seeds, args.taus, cfgs)
        (out_dir / "gate_sweep.json").write_text(json.dumps(sweep, indent=1),
                                                encoding="utf-8")
        summary["gate_sweep"] = str(out_dir / "gate_sweep.json")
        print("\nGATE SWEEP (fallback rate vs quality):")
        for cell, rows in sweep.items():
            print(f"  {cell}")
            for r in rows:
                print(f"    tau={r['tau']:.1f} fallback={r['fallback_rate']:.3f} "
                      f"rwcr={r['rwcr']:.3f} cost={r['cost']:.2f}")

    (out_dir / "agent_evaluation.json").write_text(
        json.dumps(summary, indent=1, default=float), encoding="utf-8")
    print(f"\n{banner}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
