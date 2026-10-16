"""Per-seed run store shared by every revision-1 task.

``analysis.pareto.sweep_policy`` keeps only seed means; the reviewer tasks
(paired tests, hindsight envelopes, latency bounds) need every seed. Each task
here is one ``run_single`` call -- the same paired path as every committed
result -- and writes one CSV row carrying its config hash, metrics_version,
git commit and seed. Rows are appended as they finish and skipped on restart.

:func:`cells_from_rows` aggregates rows exactly as ``analysis.pareto._aggregate``
does (mean / population std over finite seeds), so the comparator sees the
same numbers it would have seen from ``sweep_policy``.
"""

from __future__ import annotations

import csv
import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from analysis.metrics import METRICS_VERSION
from common.config import config_hash, load_yaml
from experiments.revision1.provenance import git_commit

METRICS = ("rwcr", "tx_per_at_risk_informed", "tir_median_s", "tir_p95_s",
           "actionable_deadline_miss_rate", "deadline_miss_rate", "pdr", "transmissions",
           "collisions_per_delivered", "airtime_ms", "dissemination_cbr",
           "tir_uninformed_frac", "risk_est_precision", "risk_peak_corr",
           "gate_fallback_rate", "gate_confidence_mean")
FIELDS = ("task_hash", "scenario", "density", "weather", "hazard_type", "policy", "param",
          "value", "params_json", "seed", "causal_coverage") + METRICS + (
          "config_hash", "metrics_version", "git_commit", "backend")

_CFGS: dict[str, Any] = {}


def task(scenario: str, density: float, policy: str, seed: int, *, weather: str = "clear",
         hazard_type: str = "fog_bank", param: str = "", value: Any = None,
         fixed: dict[str, Any] | None = None, causal: bool = False) -> dict[str, Any]:
    params = dict(fixed or {})
    if param:
        params[param] = value
    t = {"scenario": scenario, "density": float(density), "weather": weather,
         "hazard_type": hazard_type, "policy": policy, "param": param,
         "value": value, "params": params, "seed": int(seed), "causal": bool(causal)}
    # checkpoint path is not identity; its sha is (agent_params carries both).
    ident = {k: v for k, v in t.items() if k != "params"}
    ident["params"] = {k: v for k, v in params.items() if k != "checkpoint"}
    t["task_hash"] = config_hash(ident, METRICS_VERSION)
    return t


def _init() -> None:
    logging.disable(logging.WARNING)
    _CFGS.update(phy=load_yaml("phy.yaml"), hazard=load_yaml("hazard.yaml"),
                 experiment=load_yaml("experiment.yaml"))
    try:  # one torch thread per worker process; the pool supplies parallelism
        import torch

        torch.set_num_threads(1)
    except Exception:
        pass


def run_task(t: dict[str, Any]) -> dict[str, Any]:
    if not _CFGS:
        _init()
    from experiments.run_sim import RunSpec, run_single

    spec = RunSpec(scenario=t["scenario"], density_veh_km_lane=t["density"],
                   weather=t["weather"], policy=t["policy"], policy_params=t["params"],
                   seed=t["seed"], hazard_type=t["hazard_type"])
    m, res = run_single(spec, phy_cfg=_CFGS["phy"], hz_cfg=_CFGS["hazard"],
                        exp_cfg=_CFGS["experiment"], return_result=t["causal"])
    cov = float("nan")
    if t["causal"]:
        from agents.constrained_reward import causal_coverage, warned_at_risk
        from hazard.model import hazard_from_config
        from hazard.risk_field import build_risk_field

        trace = res.trace
        hz = hazard_from_config(_CFGS["hazard"], trace.meta,
                                overrides={"type": t["hazard_type"]})
        risk = build_risk_field(_CFGS["hazard"], trace, hz)
        rel = risk.relevance_matrix(trace, hz)
        peak, at_risk, warned = warned_at_risk(rel, res.informed_step, risk.at_risk_threshold)
        cov = float(causal_coverage(peak, at_risk, warned))
    row = {k: t[k] for k in ("task_hash", "scenario", "density", "weather", "hazard_type",
                             "policy", "param", "seed")}
    row["value"] = "" if t["value"] is None else t["value"]
    row["params_json"] = json.dumps({k: v for k, v in t["params"].items() if k != "checkpoint"},
                                    sort_keys=True, default=str)
    row["causal_coverage"] = cov
    for k in METRICS:
        row[k] = m.get(k, float("nan"))
    row.update(config_hash=m.get("config_hash"), metrics_version=METRICS_VERSION,
               git_commit=_CFGS.setdefault("commit", git_commit()), backend=m.get("backend"))
    return row


def load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def run_tasks(tasks: Sequence[dict[str, Any]], path: Path, jobs: int = 4,
              log_every: int = 200) -> list[dict[str, Any]]:
    """Run every task not already in ``path``; return all rows for ``tasks``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    done = {r["task_hash"] for r in load_rows(path)}
    todo = [t for t in tasks if t["task_hash"] not in done]
    seen: set[str] = set()
    todo = [t for t in todo if not (t["task_hash"] in seen or seen.add(t["task_hash"]))]
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        if todo:
            print(f"[runner] {len(todo)} runs to do ({len(tasks) - len(todo)} cached) -> {path.name}",
                  flush=True)
        if jobs <= 1:
            for i, t in enumerate(todo, 1):
                w.writerow(run_task(t)); fh.flush()
                if i % log_every == 0:
                    print(f"[runner] {i}/{len(todo)}", flush=True)
        else:
            with ProcessPoolExecutor(max_workers=jobs, initializer=_init) as ex:
                futs = [ex.submit(run_task, t) for t in todo]
                for i, f in enumerate(as_completed(futs), 1):
                    w.writerow(f.result()); fh.flush()
                    if i % log_every == 0:
                        print(f"[runner] {i}/{len(todo)}", flush=True)
    want = {t["task_hash"] for t in tasks}
    return [r for r in load_rows(path) if r["task_hash"] in want]


def _num(x: Any) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _val(r: dict[str, Any]) -> Any:
    v = r["value"]
    if v in ("", None):
        return None
    f = _num(v)
    return int(f) if np.isfinite(f) and float(f).is_integer() and "." not in str(v) else f


def aggregate(rows: Iterable[dict[str, Any]]):
    """Rows of one setting -> an ``OperatingPoint`` (as analysis.pareto does)."""
    from analysis.pareto import OperatingPoint

    rows = list(rows)

    def agg(k: str) -> tuple[float, float]:
        v = np.array([_num(r.get(k)) for r in rows], dtype=float)
        v = v[np.isfinite(v)]
        return (float(v.mean()), float(v.std())) if v.size else (float("nan"), float("nan"))

    q, qs = agg("rwcr")
    c, cs = agg("tx_per_at_risk_informed")
    extras = {k: agg(k)[0] for k in METRICS if k not in ("rwcr", "tx_per_at_risk_informed")}
    extras["causal_coverage"] = agg("causal_coverage")[0]
    r0 = rows[0]
    return OperatingPoint(policy=r0["policy"], param=r0["param"], value=_val(r0), quality=q,
                          quality_std=qs, cost=c, cost_std=cs, n_seeds=len(rows), extras=extras)


def cells_from_rows(rows: Iterable[dict[str, Any]], policy_alias: dict[str, str] | None = None):
    """Group per-seed rows into comparator ``Cell``s (one point per setting)."""
    from analysis.comparator import Cell, CellKey
    from analysis.pareto import PolicyCurve

    alias = policy_alias or {}
    groups: dict[tuple, dict[tuple, list]] = {}
    for r in rows:
        ck = (r["scenario"], float(r["density"]), r["weather"], r["hazard_type"])
        name = alias.get(r["policy"], r["policy"])
        groups.setdefault(ck, {}).setdefault((name, r["param"], str(r["value"]), r["params_json"]),
                                             []).append(r)
    cells = []
    for ck, settings in groups.items():
        curves: dict[str, list] = {}
        for (name, *_), rs in settings.items():
            pt = aggregate(rs)
            pt.policy = name
            curves.setdefault(name, []).append(pt)
        cells.append(Cell(key=CellKey(*ck),
                          curves={n: PolicyCurve(n, pts) for n, pts in curves.items()}))
    return cells


def default_jobs() -> int:
    return max(1, (os.cpu_count() or 2) - 4)
