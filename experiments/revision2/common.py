"""Shared loaders for revision 2: per-seed rows -> comparator cells."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from common.config import PROJECT_ROOT
from experiments.revision1.runner import cells_from_rows, load_rows

R1 = PROJECT_ROOT / "results" / "revision1"
SWEEPS = R1 / "B_sweeps"
AGENT = R1 / "agent_runs"
HEADLINE_KEYS = [("rural_highway", 2.0), ("rural_highway", 20.0), ("rural_highway", 80.0),
                 ("urban_nlos", 20.0)]


def _clear_fog(r: dict[str, Any]) -> bool:
    return r["weather"] == "clear" and r["hazard_type"] == "fog_bank"


def _knob(r: dict[str, Any]) -> bool:
    """The 55 committed knob settings (registry defaults excluded, as in Algorithm 1)."""
    return bool(r["param"]) or r["policy"] == "flooding"


def baseline_rows(cellset: str = "headline") -> list[dict[str, Any]]:
    if cellset == "headline":
        return [r for r in load_rows(SWEEPS / "headline_runs.csv") if _knob(r)]
    rows = [r for p in ("grid24_runs.csv", "gen_runs.csv") for r in load_rows(SWEEPS / p)
            if _clear_fog(r) and _knob(r)]
    return rows


def agent_rows(names: list[str], tau: float, cellset: str = "headline") -> list[dict[str, Any]]:
    """Sampled-policy rows of one gate arm from the named agent_runs files (deduplicated)."""
    keys = set(HEADLINE_KEYS)
    out, seen = [], set()
    for n in names:
        p = AGENT / f"{n}.csv"
        for r in load_rows(p):
            q = json.loads(r["params_json"])
            if (float(q.get("tau", -1)) != tau or q.get("deterministic") or "random_gate_rate" in q
                    or not _clear_fog(r) or r["task_hash"] in seen):
                continue
            if cellset == "headline" and (r["scenario"], float(r["density"])) not in keys:
                continue
            seen.add(r["task_hash"])
            out.append(r)
    return out


#: Agent rows per training seed of the full model, per cell set.
SEED_FILES = {"headline": {s: [f"ref_s{s}"] for s in range(5)},
              "grid24": {0: ["ref_s0", "ref_s0_gen", "ref_s0_grid24"]}}


def available_seeds(cellset: str) -> list[int]:
    return [s for s, fs in SEED_FILES[cellset].items() if all((AGENT / f"{f}.csv").exists() for f in fs)
            and len(load_rows(AGENT / f"{fs[0]}.csv")) >= 480]


def cells_for(cellset: str, seed: int | None, tau: float = 0.5, base=None):
    base = base if base is not None else baseline_rows(cellset)
    rows = list(base)
    if seed is not None:
        rows += agent_rows(SEED_FILES[cellset][seed], tau, cellset)
    return cells_from_rows(rows)


def fin(x) -> float | None:
    return float(x) if x is not None and np.isfinite(x) else None
