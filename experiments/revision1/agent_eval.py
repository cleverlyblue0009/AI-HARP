"""Per-seed agent evaluation (C1-C4, B4/B5), parallel, through run_single.

::

    D:/aiharp-gpu/Scripts/python.exe -m experiments.revision1.agent_eval \
        --checkpoint checkpoints/campaign/ref/ckpt_final.pt --name ref_s0 --taus 0 0.5

Same evaluation as experiments/evaluate_agent.py (suppression bias swept
-2..+3, eval seeds 0-9, gate tau, sampled or argmax), but every seed is kept,
runs are parallel and resumable, and two controls are available:

* ``--random-gate-from ROWS``: C3(b). For each (cell, bias), replace the
  confidence gate with a coin whose fallback probability equals the learned
  gate's measured fallback rate at tau=0.5 in that (cell, bias) (from ROWS).

Rows go to results/revision1/agent_runs/<name>.csv.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from common.config import PROJECT_ROOT
from experiments.evaluate_agent import DEFAULT_BIASES, agent_params
from experiments.revision1.provenance import out_dir, rel, stamp, write_json
from experiments.revision1.runner import default_jobs, load_rows, run_tasks, task

HEADLINE = [("rural_highway", 2.0, "clear", "fog_bank"), ("rural_highway", 20.0, "clear", "fog_bank"),
            ("rural_highway", 80.0, "clear", "fog_bank"), ("urban_nlos", 20.0, "clear", "fog_bank")]
#: C4: held-out topology, held-out hazards, held-out weathers.
GENERALIZATION = (
    [("urban_grid", d, "clear", "fog_bank") for d in (2.0, 20.0, 80.0)]
    + [(sc, 20.0, "clear", h) for sc in ("rural_highway", "urban_nlos")
       for h in ("waterlogging", "black_ice")]
    + [(sc, 20.0, w, "fog_bank") for sc in ("rural_highway", "urban_nlos")
       for w in ("moderate_rain", "dense_fog")]
)
CELLSETS = {"headline": HEADLINE, "generalization": GENERALIZATION}


def gate_rates(rows_path: Path) -> dict[tuple, float]:
    """Mean learned-gate fallback rate per (scenario, density, weather, hazard, bias) at tau=0.5."""
    acc: dict[tuple, list[float]] = defaultdict(list)
    for r in load_rows(rows_path):
        p = json.loads(r["params_json"])
        if r["policy"] != "ai_harp" or float(p.get("tau", -1)) != 0.5 or p.get("deterministic") \
                or "random_gate_rate" in p:
            continue
        k = (r["scenario"], float(r["density"]), r["weather"], r["hazard_type"],
             float(p["suppression_bias"]))
        v = float(r["gate_fallback_rate"]) if r["gate_fallback_rate"] not in ("", "nan") else np.nan
        if np.isfinite(v):
            acc[k].append(v)
    return {k: float(np.mean(v)) for k, v in acc.items()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--taus", type=float, nargs="*", default=[0.0, 0.5])
    ap.add_argument("--modes", nargs="*", default=["sampled"], choices=["sampled", "argmax"])
    ap.add_argument("--biases", type=float, nargs="*", default=list(DEFAULT_BIASES))
    ap.add_argument("--cells", default="headline", choices=list(CELLSETS))
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--random-gate-from", default=None)
    ap.add_argument("--jobs", type=int, default=default_jobs())
    args = ap.parse_args(argv)

    ck = Path(args.checkpoint)
    ck = ck if ck.is_absolute() else PROJECT_ROOT / ck
    rates = gate_rates(Path(args.random_gate_from)) if args.random_gate_from else None
    tasks = []
    for sc, d, w, h in CELLSETS[args.cells]:
        for mode in args.modes:
            for tau in args.taus:
                for b in args.biases:
                    if rates is not None:
                        r = rates.get((sc, d, w, h, float(b)))
                        if r is None:
                            continue
                        fixed = agent_params(ck, deterministic=mode == "argmax", random_gate_rate=r)
                    else:
                        fixed = agent_params(ck, tau=float(tau), deterministic=mode == "argmax")
                    for s in range(args.seeds):
                        tasks.append(task(sc, d, "ai_harp", s, weather=w, hazard_type=h,
                                          param="suppression_bias", value=float(b), fixed=fixed))
                if rates is not None:
                    break  # tau is irrelevant for the random gate
    path = out_dir("agent_runs") / f"{args.name}.csv"
    meta = {"checkpoint": rel(ck), "taus": args.taus, "modes": args.modes, "biases": args.biases,
            "cells": [list(c) for c in CELLSETS[args.cells]],
            "random_gate_from": args.random_gate_from}
    prev = out_dir("agent_runs") / f"{args.name}_meta.json"
    stamps = json.loads(prev.read_text()) if prev.exists() else []
    stamps.append(stamp(f"agent_runs/{args.name}", meta, seeds=list(range(args.seeds)),
                        checkpoint_sha=agent_params(ck)["checkpoint_sha"], n_runs=len(tasks)))
    write_json(prev, stamps)
    rows = run_tasks(tasks, path, jobs=args.jobs, log_every=100)
    print(f"[agent_eval] {args.name}: {len(rows)} / {len(tasks)} rows -> {rel(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
