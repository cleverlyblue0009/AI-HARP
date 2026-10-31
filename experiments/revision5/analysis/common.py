"""Revision-5 analysis: shared data loading, the selection rule and interval helpers (ANALYSIS_SPEC.md)."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from common.config import PROJECT_ROOT
from experiments.revision1.runner import load_rows

RUNS = PROJECT_ROOT / "results" / "revision5" / "runs"
OUT = PROJECT_ROOT / "results" / "revision5" / "analysis"
HEADLINE = [("rural_highway", 2.0), ("rural_highway", 20.0), ("rural_highway", 80.0), ("urban_nlos", 20.0)]
GRID24 = [(sc, float(d)) for sc in ("rural_highway", "urban_nlos", "urban_grid") for d in (1, 2, 3, 5, 10, 20, 40, 80)]
CAL, SEL, EVAL = list(range(300, 310)), list(range(400, 420)), list(range(200, 230))
D_R = {"rural_highway": 15.39, "urban_grid": 26.65, "urban_nlos": 1.89}
METRIC_COLS = {"rwcr": "rwcr", "cost": "tx_per_at_risk_informed", "miss": "actionable_deadline_miss_rate",
               "tir95": "tir_p95_s"}
EXTRA_COLS = ("tir_median_s", "stop_deadline_warned_share_actionable", "stop_deadline_warned_share",
              "deadline_miss_rate", "cpu_s", "wall_s", "gate_fallback_rate")
SEED_RNG = 20261031
B = 10_000


def num(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


@dataclass
class Table:
    """One cell's settings x seeds arrays."""
    settings: list            # setting keys, in a fixed order
    seeds: list[int]
    m: dict[str, np.ndarray]  # metric -> [S, N]

    def mean(self, metric: str, idx: np.ndarray | None = None) -> np.ndarray:
        a = self.m[metric] if idx is None else self.m[metric][:, idx]
        with np.errstate(invalid="ignore"):
            return a.mean(axis=1)


def tables(rows: list[dict], cells, seeds, key=lambda r: (r["policy"], r["param"], r["value"]),
           where=lambda r: True, extra=()) -> dict[tuple, Table]:
    """{cell: Table} from per-run rows; every (setting, seed) must be present exactly once."""
    sidx = {s: i for i, s in enumerate(seeds)}
    by: dict[tuple, dict] = {c: {} for c in cells}
    for r in rows:
        c = (r["scenario"], float(r["density"]))
        if c not in by or int(r["seed"]) not in sidx or not where(r):
            continue
        by[c].setdefault(key(r), {})[int(r["seed"])] = r
    out = {}
    for c, d in by.items():
        keys = sorted(d, key=lambda k: tuple(str(x) for x in k))
        m = {}
        for name, col in list(METRIC_COLS.items()) + [(e, e) for e in extra]:
            a = np.full((len(keys), len(seeds)), np.nan)
            for i, k in enumerate(keys):
                for s, r in d[k].items():
                    a[i, sidx[s]] = num(r.get(col))
            m[name] = a
        missing = int(sum(len(seeds) - len(d[k]) for k in keys))
        if missing:
            raise ValueError(f"{c}: {missing} (setting, seed) rows missing")
        out[c] = Table(keys, list(seeds), m)
    return out


def select(t: Table, q: float, m: float, D: float, idx=None, mask=None) -> int:
    """Selection rule on SEL: cheapest realised setting with mean RWCR >= q, mean miss <= m and mean p95 TIR <= D.
    Returns the setting index or -1 ("no pick"). Ties -> lower index."""
    rw, mi, co, tr = t.mean("rwcr", idx), t.mean("miss", idx), t.mean("cost", idx), t.mean("tir95", idx)
    ok = (rw >= q) & (mi <= m) & (tr <= D)
    if mask is not None:
        ok &= mask
    if not ok.any():
        return -1
    c = np.where(ok, co, np.inf)
    if not np.isfinite(c).any():
        return int(np.flatnonzero(ok)[0])
    return int(np.argmin(c))


def thresholds() -> dict[tuple, tuple[float, float]]:
    th = json.loads((PROJECT_ROOT / "results" / "revision5" / "thresholds.json").read_text())["cells"]
    return {(v["scenario"], float(v["density"])): (v["q_star"], v["m_star"]) for v in th.values()}


def cal_thresholds(t: Table, idx=None) -> tuple[float, float]:
    """q*, m* from a CAL table (0.95 x max mean RWCR; min mean miss + 0.05)."""
    return 0.95 * float(np.nanmax(t.mean("rwcr", idx))), float(np.nanmin(t.mean("miss", idx))) + 0.05


# ---- intervals and tests -------------------------------------------------------------------------------------
def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    from scipy.stats import beta
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def newcombe(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float]:
    """Newcombe hybrid-score interval for p1 - p2 (method 10)."""
    l1, u1 = wilson(k1, n1)
    l2, u2 = wilson(k2, n2)
    p1, p2 = k1 / n1, k2 / n2
    d = p1 - p2
    return d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)


def binom_lower_p(k: int, n: int, p0: float) -> float:
    from scipy.stats import binomtest
    return float(binomtest(k, n, p0, alternative="less").pvalue)


def holm(pvals: dict[str, float]) -> dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    out, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (len(items) - i) * p))
        out[k] = running
    return out


def finite(x) -> float | None:
    x = float(x)
    return x if math.isfinite(x) else None


def write(name: str, payload: dict, params: dict, seeds=None, n_runs=None) -> Path:
    from experiments.revision1.provenance import stamp, write_json
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / name
    write_json(p, {"provenance": stamp(f"revision5/analysis/{name}", params, seeds=seeds, n_runs=n_runs), **payload})
    return p


def rows(name: str) -> list[dict]:
    return load_rows(RUNS / name)
