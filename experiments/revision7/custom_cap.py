"""Revision 7, Experiment A: the capped policy in the custom simulator (results/revision7/runs/custom_capped_rows.jsonl).

    python -m experiments.revision7.custom_cap --jobs 3

SUMO mobility, epoch mode, the same tasks as the revision-5 custom evaluation (experiments.revision1.runner.task, so a row's
task_hash names the matching committed row) with simulation.max_decisions_per_epoch = 1. Resumable by task_hash.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from common.config import PROJECT_ROOT, load_yaml

OUT = PROJECT_ROOT / "results" / "revision7" / "runs"
ROWS = OUT / "custom_capped_rows.jsonl"
RUNS = [f"ref_s{k}" for k in range(5)]
SEEDS = list(range(200, 210))
CELLS = (("rural_highway", 20.0), ("urban_nlos", 20.0))
BIASES = (-2.0, -1.0, 0.0, 1.0, 2.0, 3.0)
_CFG: dict = {}


def tasks() -> list[tuple[str, dict]]:
    from experiments.evaluate_agent import agent_params
    from experiments.revision1.runner import task
    from experiments.revision5.sweep import checkpoint_for
    out = []
    for sc, d in CELLS:
        for run in RUNS:
            fixed = agent_params(checkpoint_for(run), tau=0.5, deterministic=False)
            for s in SEEDS:
                for b in BIASES:
                    out.append((run, task(sc, d, "ai_harp", s, param="suppression_bias", value=b, fixed=fixed, backend="sumo")))
    return out


def run_one(run: str, t: dict) -> dict:
    from experiments.run_sim import RunSpec, run_single
    if not _CFG:
        _CFG.update(phy=load_yaml("phy.yaml"), hazard=load_yaml("hazard.yaml"), experiment=copy.deepcopy(load_yaml("experiment.yaml")))
        _CFG["experiment"]["simulation"]["max_decisions_per_epoch"] = 1
    w0 = time.time()
    spec = RunSpec(scenario=t["scenario"], density_veh_km_lane=t["density"], weather=t["weather"], policy=t["policy"],
                   policy_params=t["params"], seed=t["seed"], hazard_type=t["hazard_type"], backend="sumo")
    m, res = run_single(spec, phy_cfg=_CFG["phy"], hz_cfg=_CFG["hazard"], exp_cfg=_CFG["experiment"], return_result=True)
    return {"task_hash": t["task_hash"], "run": run, "scenario": t["scenario"], "density": t["density"], "seed": t["seed"],
            "bias": t["value"], "backend": m.get("backend"), "rwcr": m.get("rwcr"), "tx_per_at_risk_informed": m.get("tx_per_at_risk_informed"),
            "actionable_deadline_miss_rate": m.get("actionable_deadline_miss_rate"), "tir_p95_s": m.get("tir_p95_s"),
            "transmissions": m.get("transmissions"), "decisions": int(sum(res.action_counts.values())),
            "capped_callbacks": int(res.meta["capped_callbacks"]), "config_hash": m.get("config_hash"), "wall_s": round(time.time() - w0, 2)}


def main(argv=None) -> int:
    import ctypes

    import numpy

    from experiments.revision1.provenance import stamp, write_json
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--jobs", type=int, default=3)
    a = ap.parse_args(argv)
    print(f"[env] {sys.executable} numpy {numpy.__version__}", flush=True)
    os.environ["SUMO_HOME"] = os.environ.get("AIHARP_SUMO_HOME", r"D:\sumo\sumo-1.19.0")   # backend is explicit: "sumo"
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)      # idle sleep only; no power setting is changed
    OUT.mkdir(parents=True, exist_ok=True)
    ts = tasks()
    done = {json.loads(l)["task_hash"] for l in ROWS.open(encoding="utf-8")} if ROWS.exists() else set()
    todo = [(r, t) for r, t in ts if t["task_hash"] not in done]
    print(f"[custom_cap] {len(ts)} tasks, {len(todo)} to do, {a.jobs} jobs", flush=True)
    write_json(OUT / "custom_capped_meta.json", stamp("revision7/custom_cap", {"runs": RUNS, "cells": list(CELLS), "biases": BIASES,
                                                                                 "max_decisions_per_epoch": 1}, seeds=SEEDS, n_runs=len(ts)))
    n = 0
    with ProcessPoolExecutor(a.jobs) as ex, ROWS.open("a", encoding="utf-8") as fh:
        for f in as_completed([ex.submit(run_one, r, t) for r, t in todo]):
            try:
                row = f.result()
            except Exception as exc:  # noqa: BLE001 -- recorded; a rerun retries it
                print(f"[custom_cap] FAILED {exc!r}", flush=True)
                continue
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            n += 1
            if n % 50 == 0:
                print(f"{time.strftime('%H:%M:%S')} {n}/{len(todo)}", flush=True)
    left = len(ts) - len(done) - n
    print(f"[custom_cap] {n} new, {left} still not done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
