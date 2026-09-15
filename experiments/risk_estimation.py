"""Causal-vs-oracle risk estimation agreement over the full grid.

::

    D:/aiharp-env/python.exe -m experiments.risk_estimation --jobs 3

How well can risk be estimated from what a vehicle knows at time t
(hazard/oracle.py, ``estimation_agreement``)? The answer depends only on the
traffic, the hazard and the two risk fields -- not on any dissemination policy
-- so it is computed once per (scenario, density, weather, hazard, seed) rather
than per policy. It was missing from results/runs.csv (the sweep writes only
METRIC_DIRECTION metrics), and adding columns there would break the header
agent rows are appended under.

Uses the same trace cache as experiments/full_sweep.py, so after that sweep
every trace is a cache hit. Rows are appended and skipped on restart. Writes
results/risk_estimation.csv; train and held-out labels as in runs.csv.
"""

from __future__ import annotations

import argparse
import csv
import logging
import time
from pathlib import Path
from typing import Any

from common.config import PROJECT_ROOT, load_yaml
from experiments.full_sweep import DENSITIES, HAZARDS, SCENARIOS, WEATHERS, splits

FIELDS = ("scenario", "density", "density_veh_km_lane", "weather", "hazard_type", "seed",
          "topology_split", "hazard_split", "weather_split", "risk_est_precision",
          "risk_est_recall", "risk_est_f1", "risk_peak_corr", "n_at_risk_oracle",
          "n_at_risk_causal", "oracle_horizon_s", "backend")
KEY = ("scenario", "density", "weather", "hazard_type", "seed")
_CFGS: dict[str, Any] = {}


def _init() -> None:
    logging.disable(logging.INFO)
    _CFGS.update(phy=load_yaml("phy.yaml"), hazard=load_yaml("hazard.yaml"),
                 agent=load_yaml("agent.yaml"))


def agreement_row(task: tuple[str, float, str, str, int]) -> dict[str, Any]:
    from hazard.model import hazard_from_config
    from hazard.oracle import build_oracle_risk_field, estimation_agreement
    from hazard.risk_field import build_risk_field
    from mobility.generate import get_trace, load_scenario

    if not _CFGS:
        _init()
    sc, d, w, h, seed = task
    trace = get_trace(load_scenario(sc), d, seed, weather=w, phy_cfg=_CFGS["phy"])
    hazard = hazard_from_config(_CFGS["hazard"], trace.meta, overrides={"type": h})
    risk = build_risk_field(_CFGS["hazard"], trace, hazard)
    agree = estimation_agreement(trace, hazard, risk, build_oracle_risk_field(risk, hazard))
    return {"scenario": sc, "density": d, "density_veh_km_lane": d, "weather": w,
            "hazard_type": h, "seed": seed, **splits(sc, h, w, _CFGS["agent"]["training"]),
            **agree, "backend": trace.backend}


def _key(row: dict[str, Any]) -> tuple[str, ...]:
    return tuple(f"{float(row[k]):g}" if k == "density" else str(row[k]) for k in KEY)


def run(out_csv: Path, tasks: list[tuple[str, float, str, str, int]], jobs: int = 1) -> int:
    done: set[tuple[str, ...]] = set()
    if out_csv.exists():
        with out_csv.open(newline="", encoding="utf-8") as fh:
            done = {_key(r) for r in csv.DictReader(fh)}
    todo = [t for t in tasks if _key(dict(zip(KEY, t))) not in done]
    print(f"{len(tasks)} cells, {len(tasks) - len(todo)} done, {len(todo)} to go", flush=True)
    if not todo:
        return 0
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    new = not out_csv.exists()
    t0 = time.time()
    with out_csv.open("a", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        if new:
            wr.writeheader()
        if jobs > 1:
            from concurrent.futures import ProcessPoolExecutor

            ex = ProcessPoolExecutor(max_workers=jobs, initializer=_init)
            it = ex.map(agreement_row, todo, chunksize=8)
        else:
            _init()
            ex, it = None, map(agreement_row, todo)
        try:
            for i, row in enumerate(it, 1):
                wr.writerow(row)
                if i % 200 == 0 or i == len(todo):
                    fh.flush()
                    print(f"  {i}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
        finally:
            if ex is not None:
                ex.shutdown(cancel_futures=True)
    return len(todo)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--out", default="results/risk_estimation.csv")
    args = ap.parse_args(argv)
    tasks = [(sc, float(d), w, h, s) for d in DENSITIES for sc in SCENARIOS for w in WEATHERS
             for h in HAZARDS for s in range(args.seeds)]
    run(PROJECT_ROOT / args.out, tasks, args.jobs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
