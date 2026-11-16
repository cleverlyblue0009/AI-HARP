"""CMA-ES trainer for STRUCT (revision 8, results/revision8/ADDENDUM.md section 4).

    python -m agents.train_structured --arm plain --seed 0 [--workers 4] [--generations 80]
    python -m agents.train_structured --arm rand  --seed 3            # resumes from its checkpoint if one exists

Training uses only the training seed pool (100-131), the four headline (scenario, density) pairs, clear weather and the
fog_bank hazard. Fitness is Deb's feasibility rule on CAUSAL coverage targets (results/coverage_targets.json, 32-seed
ceilings x 0.95): feasible (every cell group's pooled shortfall <= 0) beats infeasible; feasible candidates are ranked
by pooled transmissions per causal at-risk vehicle, infeasible ones by the summed positive group shortfall. No
Lagrangian, no oracle quantity (this module and everything it imports is checked by tests/test_oracle_isolation.py).

Outputs, per run, in results/revision8/train/<arm>_s<seed>/: history.jsonl (one line per generation), ckpt.pkl (exact
resume: CMA state, NumPy global RNG state used by cma's sampler, generation), theta.json (the CMA mean after the last
generation, which is the trained policy; no "best candidate" is selected on any data).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import pickle
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from agents.constrained_reward import CoverageTargets, causal_coverage, warned_at_risk
from agents.structured import N_PARAMS, StructuredPolicy
from common.config import PROJECT_ROOT, load_yaml

CELLS = (("rural_highway", 2.0), ("rural_highway", 20.0), ("rural_highway", 80.0), ("urban_nlos", 20.0))
POOL = tuple(range(100, 132))
SEEDS_PER_CELL = 3
GENERATIONS = 80
BATCH_SALT, AXES_SALT = 0x5752, 0x52
INFEASIBLE = 1e6
OUT = PROJECT_ROOT / "results" / "revision8" / "train"
H3 = PROJECT_ROOT / "results" / "revision5" / "analysis" / "h3.json"
WARM = PROJECT_ROOT / "results" / "revision8" / "warm_start.json"
SUMO_HOME_DEFAULT = None


def cma_stds() -> np.ndarray:
    """Initial per-coordinate scale (ADDENDUM section 4), sigma0 = 1."""
    return np.array([0.2] * 4 + [2.0, 2.0, 2.0, 2.0, 0.2] + [0.5] * 12)


# ------------------------------------------------------------------------------------------------- axes --
def r5_table() -> dict[str, dict]:
    """Per headline cell: signed gap PDR_veins - PDR_custom per 50 m bin (h3.json, epoch mode); bins with fewer than 10
    links in either simulator, or NaN, get gap 0."""
    nl = json.loads(H3.read_text(encoding="utf-8"))["network_layer"]
    out = {}
    for sc, d in CELLS:
        rows = nl[f"{sc} d={d:g}"]["pdr_vs_distance"]
        gap, pc = [], []
        for b in rows:
            pv, pcu = b["pdr_veins"], b["pdr_custom"]
            ok = (b["n_veins"] >= 10 and b["n_custom"] >= 10 and pv is not None and pcu is not None
                  and np.isfinite(pv) and np.isfinite(pcu))
            gap.append(float(pv - pcu) if ok else 0.0)
            pc.append(float(pcu) if pcu is not None and np.isfinite(pcu) else 0.5)
        out[f"{sc}|{d:g}"] = {"bin_m": 50.0, "gap": gap, "pdr_custom": pc}
    return out


def sample_axes(rng: np.random.Generator, cell: tuple[str, float], r5: dict) -> dict[str, Any]:
    """One independent draw of R1-R5 (ADDENDUM section 3). Always draws the same number of values."""
    u = rng.random(7)
    key = f"{cell[0]}|{cell[1]:g}"
    return {"dup_count_per_frame": bool(u[0] < 0.5),
            "nb_table_error": (0.5 * float(u[1]), 0.5 * float(u[2]), 1.15),
            "obs_lag_steps": int(u[3] < 0.5),
            "backend": "sumo" if u[4] < 0.5 else "fallback",
            "pdr_perturb": {**r5[key], "u": float(u[5])}}


def plain_axes() -> dict[str, Any]:
    return {"dup_count_per_frame": False, "nb_table_error": None, "obs_lag_steps": 0, "backend": "fallback",
            "pdr_perturb": None}


def batch(train_seed: int, gen: int, arm: str, r5: dict) -> list[dict]:
    """The generation's episodes: SEEDS_PER_CELL pool seeds per cell (without replacement), and their axes."""
    rng = np.random.default_rng([train_seed, gen, BATCH_SALT])
    eps = []
    for ci, cell in enumerate(CELLS):
        for ei, s in enumerate(rng.choice(POOL, size=SEEDS_PER_CELL, replace=False)):
            ax = (sample_axes(np.random.default_rng([train_seed, gen, ci, ei, AXES_SALT]), cell, r5) if arm == "rand"
                  else plain_axes())
            eps.append({"cell": cell, "seed": int(s), "axes": ax})
    return eps


# ---------------------------------------------------------------------------------------------- episodes --
_CACHE: dict = {}


def _world(scenario: str, density: float, seed: int, backend: str):
    """Trace, hazard, PHY, MAC, risk field and causal relevance matrix for one episode (memoised per worker),
    built exactly as experiments/run_sim.py builds them."""
    key = (scenario, density, seed, backend)
    if key in _CACHE:
        return _CACHE[key]
    from hazard.model import hazard_from_config
    from hazard.risk_field import build_risk_field
    from mobility.generate import get_trace, load_scenario
    from sim.mac import build_mac
    from sim.phy import build_phy
    phy_cfg, hz_cfg = load_yaml("phy.yaml"), load_yaml("hazard.yaml")
    sc = load_scenario(scenario)
    tr = get_trace(sc, density, seed, weather="clear", backend=backend, phy_cfg=phy_cfg)
    hz = hazard_from_config(hz_cfg, tr.meta, overrides={"type": "fog_bank"})
    phy = build_phy(phy_cfg, scenario=sc.get("phy_profile", sc["name"]), weather="clear", seed=seed,
                    trace_meta=tr.meta)
    mac = build_mac(phy_cfg, phy)
    risk = build_risk_field(hz_cfg, tr, hz)
    rel = risk.relevance_matrix(tr, hz)                       # CAUSAL
    if len(_CACHE) >= 24:                                   # memory bound (about 0.1 GB per d=80 entry); no effect on results
        _CACHE.clear()
    _CACHE[key] = (tr, hz, phy, mac, risk, rel)
    return _CACHE[key]


def run_episode(theta, ep: dict) -> tuple[float, int, int]:
    """(causal coverage, causal at-risk count, transmissions) of STRUCT(theta) on one episode."""
    from common.seeding import SeedBundle
    from sim.engine import DisseminationEngine, SimSettings
    ax = ep["axes"]
    sc, d = ep["cell"]
    tr, hz, phy, mac, risk, rel = _world(sc, float(d), int(ep["seed"]), ax["backend"])
    base = SimSettings.from_config(load_yaml("experiment.yaml"))
    st = dataclasses.replace(base, dup_count_per_frame=ax["dup_count_per_frame"], nb_table_error=ax["nb_table_error"],
                             obs_lag_steps=ax["obs_lag_steps"], pdr_perturb=ax["pdr_perturb"])
    pol = StructuredPolicy(theta=list(theta))
    res = DisseminationEngine(tr, phy, mac, risk, hz, pol, SeedBundle(master_seed=int(ep["seed"])), st).run()
    peak, at_risk, warned = warned_at_risk(rel, res.informed_step, risk.at_risk_threshold)
    return causal_coverage(peak, at_risk, warned), int(at_risk.sum()), int(res.n_transmissions)


def _job(args):
    import logging
    logging.disable(logging.WARNING)
    k, theta, ep = args
    return k, run_episode(theta, ep)


def fitness(outcomes: list[tuple[float, int, int]], eps: list[dict], targets: CoverageTargets) -> dict:
    """Deb's rule as a scalar: cost if feasible, else 1e6 + summed positive group shortfall."""
    groups: dict[str, list[float]] = {}
    tx = n_ar = 0
    for (cov, nar, ntx), ep in zip(outcomes, eps):
        sc, d = ep["cell"]
        g = f"{sc}|{d:g}"
        tgt = targets.target(sc, d, "clear", "fog_bank")
        t_c = groups.setdefault(g, [0.0, 0.0])
        if np.isfinite(cov) and nar > 0:
            t_c[0] += tgt
            t_c[1] += cov
        tx += ntx
        n_ar += nar
    short = {g: ((t - c) / t if t > 0 else 0.0) for g, (t, c) in groups.items()}
    cost = tx / n_ar if n_ar > 0 else float("inf")
    feasible = all(s <= 0 for s in short.values())
    pos = float(sum(max(s, 0.0) for s in short.values()))
    f = cost if feasible else INFEASIBLE + pos
    return {"f": float(f), "feasible": feasible, "cost": float(cost), "shortfall": short, "pos_shortfall": pos}


# ------------------------------------------------------------------------------------------------- train --
def warm_theta() -> np.ndarray:
    w = json.loads(WARM.read_text(encoding="utf-8"))
    th = np.asarray(w["theta"], dtype=float)
    assert th.shape == (N_PARAMS,)
    return th


def train(arm: str, seed: int, generations: int = GENERATIONS, workers: int = 4, out: Path | None = None,
          stop_after: int | None = None) -> Path:
    import cma
    out = out or OUT / f"{arm}_s{seed}"
    out.mkdir(parents=True, exist_ok=True)
    ck, hist = out / "ckpt.pkl", out / "history.jsonl"
    cfg = load_yaml("agent.yaml")["objective"]
    targets = CoverageTargets.load(PROJECT_ROOT / cfg["targets_path"], float(cfg["target_fraction"]),
                                   float(cfg["fallback_target"]))
    r5 = r5_table()
    if ck.exists():
        state = pickle.loads(ck.read_bytes())
        es, gen = state["es"], state["generation"]
        np.random.set_state(state["np_global"])
        n_hist = sum(1 for _ in hist.open()) if hist.exists() else 0
        if n_hist != gen:
            raise RuntimeError(f"{hist}: {n_hist} lines but checkpoint at generation {gen}")
        print(f"[train_structured] resume {arm}_s{seed} at generation {gen}", flush=True)
    else:
        if hist.exists():
            raise RuntimeError(f"{hist} exists without a checkpoint; refusing to overwrite")
        x0 = warm_theta()
        es = cma.CMAEvolutionStrategy(x0.tolist(), 1.0, {"seed": 1000 + seed, "CMA_stds": cma_stds().tolist(),
                                                         "verbose": -9})
        gen = 0
        meta = {"arm": arm, "seed": seed, "generations": generations, "popsize": es.popsize, "x0": x0.tolist(),
                "cma_version": cma.__version__, "numpy": np.__version__, "python": sys.version.split()[0],
                "executable": sys.executable, "platform": platform.platform(), "cells": CELLS, "pool": [POOL[0], POOL[-1]],
                "seeds_per_cell": SEEDS_PER_CELL, "warm_start": json.loads(WARM.read_text())}
        (out / "run_meta.json").write_text(json.dumps(meta, indent=1, default=str))
    from multiprocessing import get_context
    pool = get_context("fork").Pool(workers) if workers > 1 else None
    try:
        while gen < generations:
            t0, c0 = time.time(), time.process_time()
            eps = batch(seed, gen, arm, r5)
            X = es.ask()
            jobs = [((ci, ei), x, ep) for ci, x in enumerate(X) for ei, ep in enumerate(eps)]
            res = dict(pool.imap_unordered(_job, jobs, chunksize=2) if pool else map(_job, jobs))
            fits = [fitness([res[(ci, ei)] for ei in range(len(eps))], eps, targets) for ci in range(len(X))]
            es.tell(X, [f["f"] for f in fits])
            gen += 1
            fs = np.array([f["f"] for f in fits])
            b = int(np.argmin(fs))
            feas_costs = [f["cost"] for f in fits if f["feasible"]]
            line = {"generation": gen, "best_f": float(fs[b]), "median_f": float(np.median(fs)),
                    "feasibility_rate": float(np.mean([f["feasible"] for f in fits])),
                    "best_cost": fits[b]["cost"], "median_feasible_cost": (float(np.median(feas_costs))
                                                                          if feas_costs else None),
                    "best_shortfall_by_group": fits[b]["shortfall"], "mean": [float(v) for v in es.mean],
                    "sigma": float(es.sigma), "episodes": [[e["cell"][0], e["cell"][1], e["seed"]] for e in eps],
                    "wall_s": round(time.time() - t0, 2), "main_cpu_s": round(time.process_time() - c0, 2)}
            with hist.open("a") as fh:
                fh.write(json.dumps(line) + "\n")
            tmp = ck.with_suffix(".tmp")
            tmp.write_bytes(pickle.dumps({"es": es, "generation": gen, "np_global": np.random.get_state()}))
            os.replace(tmp, ck)
            print(f"[train_structured] {arm}_s{seed} gen {gen}/{generations} best_f {fs[b]:.4f} "
                  f"feasible {line['feasibility_rate']:.2f} wall {line['wall_s']}s", flush=True)
            if stop_after is not None and gen >= stop_after:
                break
    finally:
        if pool:
            pool.close()
            pool.join()
    if gen >= generations:
        (out / "theta.json").write_text(json.dumps({"arm": arm, "seed": seed, "generations": gen,
                                                    "theta": [float(v) for v in es.mean]}, indent=1))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", required=True, choices=["plain", "rand"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--generations", type=int, default=GENERATIONS)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--stop-after", type=int, default=None, help="stop cleanly after this generation (smoke/resume)")
    a = ap.parse_args(argv)
    print(f"[env] {sys.executable} numpy {np.__version__}", flush=True)
    if a.generations != GENERATIONS and a.out is None:
        raise SystemExit("the pre-registered budget is 80 generations; other values only with --out (smoke runs)")
    if a.arm == "rand" and not os.environ.get("SUMO_HOME"):
        raise SystemExit("STRUCT-rand samples the SUMO backend (R4): set SUMO_HOME")
    train(a.arm, a.seed, a.generations, a.workers, a.out, a.stop_after)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
