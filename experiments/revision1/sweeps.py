"""Per-seed baseline sweeps for Stage B (torch-free).

::

    python -m experiments.revision1.sweeps grid24   --jobs 8   # B1: 24 cells x 55 settings
    python -m experiments.revision1.sweeps headline --jobs 8   # B3/B4: 63 settings, eval seeds
    python -m experiments.revision1.sweeps train    --jobs 8   # B3: 63 settings, training seeds, causal
    python -m experiments.revision1.sweeps slots    --jobs 8   # B2: slot grids beyond 50

Every row goes through ``experiments.run_sim.run_single`` (the committed paired
path). Outputs (CSV, one row per run, resumable) under results/revision1/.
"""

from __future__ import annotations

import argparse

from agents.registry import BASELINE_POLICIES
from analysis.pareto import POLICY_SWEEPS
from experiments.revision1.provenance import out_dir, stamp, write_json
from experiments.revision1.runner import default_jobs, run_tasks, task

HEADLINE = [("rural_highway", 2.0), ("rural_highway", 20.0), ("rural_highway", 80.0),
            ("urban_nlos", 20.0)]
GRID24 = [(sc, d) for sc in ("rural_highway", "urban_nlos", "urban_grid")
          for d in (1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 40.0, 80.0)]
EVAL_SEEDS = list(range(10))
TRAIN_SEEDS = list(range(100, 132))
#: B2: beyond the committed n_slots <= 50 / fallback_n_slots <= 32.
EXTRA_SLOTS = {"slotted_1p": ("n_slots", (75, 100, 150, 200)),
               "dvcast": ("n_slots", (75, 100, 150, 200)),
               "greedy_farthest": ("fallback_n_slots", (64, 128))}


def knob_settings() -> list[tuple[str, str, object]]:
    """The 55 committed knob settings (analysis/pareto.py POLICY_SWEEPS)."""
    out = []
    for pol, (param, values) in POLICY_SWEEPS.items():
        out += [(pol, "", None)] if not values else [(pol, param, v) for v in values]
    return out


def all63() -> list[tuple[str, str, object]]:
    """The 63 settings of experiments/sparse_feasibility.py: 55 knob settings
    (flooding counted once) + each registry default except flooding's."""
    s = [(p, a, v) for p, a, v in knob_settings()]
    s += [(p, "", None) for p in BASELINE_POLICIES if p != "flooding"]
    return s


def build(name: str):
    if name == "grid24":
        cells, settings, seeds, causal = GRID24, knob_settings(), EVAL_SEEDS, False
        # headline cells first, so B1's four cells finish early
        cells = HEADLINE + [c for c in cells if c not in HEADLINE]
    elif name == "headline":
        cells, settings, seeds, causal = HEADLINE, all63(), EVAL_SEEDS, True
    elif name == "train":
        cells, settings, seeds, causal = HEADLINE[1:], all63(), TRAIN_SEEDS, True
    elif name == "gen":
        # C4 baseline curves for the generalization cells not already in grid24.
        from experiments.revision1.agent_eval import GENERALIZATION

        gen = list(GENERALIZATION)
        tasks = [task(sc, d, p, s, weather=w, hazard_type=h, param=a, value=v)
                 for sc, d, w, h in gen for p, a, v in knob_settings() for s in EVAL_SEEDS]
        meta = {"sweep": name, "cells": gen, "n_settings": len(knob_settings()),
                "note": "the urban_grid cells here are also grid24 cells; grid24 reuses these rows"}
        return tasks, meta, EVAL_SEEDS
    elif name == "slots":
        cells, seeds, causal = HEADLINE, EVAL_SEEDS, False
        settings = [(p, a, v) for p, (a, vs) in EXTRA_SLOTS.items() for v in vs]
    else:
        raise SystemExit(f"unknown sweep {name}")
    tasks = [task(sc, d, p, s, param=a, value=v, causal=causal)
             for sc, d in cells for p, a, v in settings for s in seeds]
    meta = {"sweep": name, "cells": cells, "settings": [f"{p}({a}={v})" if a else f"{p}(default)"
                                                       for p, a, v in settings],
            "n_settings": len(settings), "weather": "clear", "hazard_type": "fog_bank",
            "causal_coverage": causal}
    return tasks, meta, seeds


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("sweep", choices=["grid24", "headline", "train", "slots", "gen"])
    ap.add_argument("--jobs", type=int, default=default_jobs())
    args = ap.parse_args(argv)
    tasks, meta, seeds = build(args.sweep)
    d = out_dir("B_sweeps")
    write_json(d / f"{args.sweep}_meta.json", stamp(f"B_sweeps/{args.sweep}", meta, seeds=seeds,
                                                    n_runs=len(tasks)))
    also = [d / "gen_runs.csv"] if args.sweep == "grid24" else []
    rows = run_tasks(tasks, d / f"{args.sweep}_runs.csv", jobs=args.jobs, also=also)
    print(f"[sweeps] {args.sweep}: {len(rows)} / {len(tasks)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
