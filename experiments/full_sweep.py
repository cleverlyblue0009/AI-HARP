"""The full factorial sweep -> results/runs.csv, the paper's evidence base.

::

    D:/aiharp-env/python.exe -m experiments.full_sweep --jobs 3                       # baselines
    D:/aiharp-env/python.exe -m experiments.full_sweep --jobs 3 \\
        --checkpoint checkpoints/campaign/ref/ckpt_final.pt --taus 0 0.5               # + agent

Grid: scenario {rural_highway, urban_nlos, urban_grid} x density {1, 2, 3, 5,
10, 20, 40, 80} x weather {clear, heavy_rain, moderate_rain, dense_fog} x
hazard {fog_bank, landslide, crash, waterlogging, black_ice} x policy x seeds
0-9. Baselines run at their registry defaults; the agent at each ``--taus``
value with its sampled policy (the headline evaluation mode).

Every row goes through ``experiments.run_sim.run_single`` -- the same paired
path as every other result -- so seed k gives every policy identical mobility,
hazard placement, shadowing and fading. Each row carries split labels:
``topology_split`` (urban_grid is never trained on), ``hazard_split`` and
``weather_split`` (training vs held-out, from configs/agent.yaml). Train and
held-out rows must never be pooled in an analysis.

Rows are appended as they finish and skipped on restart, so an interrupted
sweep resumes. Cheap cells (low density) run first.
"""

from __future__ import annotations

import argparse
import csv
import logging
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from agents.registry import BASELINE_POLICIES
from analysis.metrics import METRIC_DIRECTION, METRICS_VERSION
from common.config import PROJECT_ROOT, load_yaml

SCENARIOS = ("rural_highway", "urban_nlos", "urban_grid")
DENSITIES = (1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 40.0, 80.0)
WEATHERS = ("clear", "heavy_rain", "moderate_rain", "dense_fog")
HAZARDS = ("fog_bank", "landslide", "crash", "waterlogging", "black_ice")

#: ``density_veh_km_lane`` duplicates ``density`` under the name analysis/report.py
#: and analysis/tables.py read.
ID_FIELDS = ("scenario", "density", "density_veh_km_lane", "weather", "hazard_type",
             "policy", "tau", "seed",
             "topology_split", "hazard_split", "weather_split", "checkpoint_sha")
EXTRA_FIELDS = ("gate_fallback_rate", "gate_confidence_mean", "backend", "config_hash",
                "comm_range_m", "cs_range_m", "hazard_severity", "metrics_version")
FIELDS = ID_FIELDS + tuple(sorted(METRIC_DIRECTION)) + EXTRA_FIELDS
KEY_FIELDS = ("scenario", "density", "weather", "hazard_type", "policy", "tau", "seed",
              "checkpoint_sha")

_CFGS: dict[str, Any] = {}


def splits(scenario: str, hazard: str, weather: str, training: dict[str, Any]) -> dict[str, str]:
    """Train / held-out labels for one cell, from configs/agent.yaml -> training."""
    return {
        "topology_split": "held_out" if scenario in training["held_out_scenarios"] else "train",
        "hazard_split": "held_out" if hazard in training["held_out_hazards"] else "train",
        "weather_split": "held_out" if weather in training["held_out_weather"] else "train",
    }


def grid(policies: Sequence[str], taus: Sequence[float], seeds: Sequence[int],
         checkpoint: str | None, checkpoint_sha: str, scenarios: Sequence[str] = SCENARIOS,
         densities: Sequence[float] = DENSITIES, weathers: Sequence[str] = WEATHERS,
         hazards: Sequence[str] = HAZARDS) -> list[dict[str, Any]]:
    """Every run, cheapest first. Agent entries only when a checkpoint is given."""
    arms: list[tuple[str, str]] = [(p, "") for p in policies]
    if checkpoint:
        arms += [("ai_harp", f"{float(t):g}") for t in taus]
    return [{"scenario": sc, "density": float(d), "weather": w, "hazard_type": h,
             "policy": pol, "tau": tau, "seed": int(s),
             "checkpoint_sha": checkpoint_sha if pol == "ai_harp" else ""}
            for d in densities for sc in scenarios for w in weathers for h in hazards
            for pol, tau in arms for s in seeds]


def row_key(row: dict[str, Any]) -> tuple[str, ...]:
    return tuple(f"{float(row[k]):g}" if k == "density" else str(row[k]) for k in KEY_FIELDS)


def _init() -> None:
    logging.disable(logging.INFO)
    _CFGS.update(phy=load_yaml("phy.yaml"), hazard=load_yaml("hazard.yaml"),
                 experiment=load_yaml("experiment.yaml"), agent=load_yaml("agent.yaml"))


def run_task(task: dict[str, Any], checkpoint: str | None = None,
             duration_s: float | None = None) -> dict[str, Any]:
    from experiments.evaluate_agent import agent_params
    from experiments.run_sim import RunSpec, run_single

    if not _CFGS:
        _init()
    params: dict[str, Any] = {}
    if task["policy"] == "ai_harp":
        params = agent_params(checkpoint, tau=float(task["tau"]), deterministic=False)
    spec = RunSpec(scenario=task["scenario"], density_veh_km_lane=task["density"],
                   weather=task["weather"], policy=task["policy"], policy_params=params,
                   seed=task["seed"], hazard_type=task["hazard_type"], duration_s=duration_s)
    m, _ = run_single(spec, phy_cfg=_CFGS["phy"], hz_cfg=_CFGS["hazard"],
                      exp_cfg=_CFGS["experiment"])
    row = {k: m.get(k, np.nan) for k in FIELDS}
    row.update(task)
    row["density_veh_km_lane"] = task["density"]
    row.update(splits(task["scenario"], task["hazard_type"], task["weather"],
                      _CFGS["agent"]["training"]))
    row["metrics_version"] = METRICS_VERSION
    return row


def _run_star(args: tuple[dict[str, Any], str | None, float | None]) -> dict[str, Any]:
    return run_task(*args)


def sweep(out_csv: Path, tasks: list[dict[str, Any]], checkpoint: str | None, jobs: int,
          duration_s: float | None = None, log_every: int = 200) -> int:
    """Run every task not already in ``out_csv``; returns how many were run."""
    done: set[tuple[str, ...]] = set()
    if out_csv.exists():
        with out_csv.open(newline="", encoding="utf-8") as fh:
            done = {row_key(r) for r in csv.DictReader(fh)}
    todo = [t for t in tasks if row_key(t) not in done]
    print(f"{len(tasks)} runs in the grid, {len(tasks) - len(todo)} already in "
          f"{out_csv.name}, {len(todo)} to go", flush=True)
    if not todo:
        return 0
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    new = not out_csv.exists()
    t0 = time.time()
    work = [(t, checkpoint, duration_s) for t in todo]
    with out_csv.open("a", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        if new:
            wr.writeheader()
        if jobs > 1:
            from concurrent.futures import ProcessPoolExecutor

            ex = ProcessPoolExecutor(max_workers=jobs, initializer=_init)
            it = ex.map(_run_star, work, chunksize=4)
        else:
            _init()
            ex, it = None, map(_run_star, work)
        try:
            for i, row in enumerate(it, 1):
                wr.writerow(row)
                if i % log_every == 0 or i == len(todo):
                    fh.flush()
                    el = time.time() - t0
                    print(f"  {i}/{len(todo)}  {el:.0f}s  eta {el / i * (len(todo) - i):.0f}s  "
                          f"(last: {row['scenario']} d={row['density']:g} {row['policy']})",
                          flush=True)
        finally:
            if ex is not None:
                ex.shutdown(cancel_futures=True)
    return len(todo)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Full factorial sweep -> results/runs.csv")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--checkpoint", default=None, help="add the agent at each --taus value")
    ap.add_argument("--taus", nargs="*", type=float, default=[0.0, 0.5])
    ap.add_argument("--no-baselines", action="store_true")
    ap.add_argument("--scenarios", nargs="*", default=list(SCENARIOS))
    ap.add_argument("--densities", nargs="*", type=float, default=list(DENSITIES))
    ap.add_argument("--weathers", nargs="*", default=list(WEATHERS))
    ap.add_argument("--hazards", nargs="*", default=list(HAZARDS))
    ap.add_argument("--out", default="results/runs.csv")
    args = ap.parse_args(argv)

    sha = ""
    if args.checkpoint:
        from experiments.evaluate_agent import checkpoint_sha

        sha = checkpoint_sha(args.checkpoint)
    policies = [] if args.no_baselines else list(BASELINE_POLICIES)
    tasks = grid(policies, args.taus, range(args.seeds), args.checkpoint, sha,
                 args.scenarios, args.densities, args.weathers, args.hazards)
    sweep(PROJECT_ROOT / args.out, tasks, args.checkpoint, args.jobs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
