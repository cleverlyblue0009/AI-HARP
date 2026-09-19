"""Paired significance tests over the full sweep (results/runs.csv).

::

    D:/aiharp-env/python.exe -m analysis.grid_stats

For every cell (scenario, density, weather, hazard), every headline metric and
every agent arm (tau), a Wilcoxon signed-rank test paired by seed of the agent
against two references:

* ``oracle_best`` -- the baseline with the best mean in THAT cell for THAT
  metric (hindsight tuning; the hardest reference);
* ``fixed_best`` -- ONE baseline for the whole grid (what an engineer would
  ship), chosen per metric by its mean over the fully-training cells only
  (training topology, hazard and weather), so held-out data never picks it.

Families: one per (comparison, tau, cell) -- Holm-Bonferroni across the four
METRICS tested in that cell, and nothing else. Train and held-out rows are
never pooled (a cell belongs to exactly one split).

The family must be stated because at this seed count it decides what can be
found at all. With 10 paired seeds the smallest two-sided Wilcoxon p is
2^-9 = 0.00195, so **no family larger than 25 tests can produce a
Holm-corrected p below 0.05 at any effect size**. Two larger families were
tried and both were unrejectable by construction:

* all four metrics x all cells of a split (128-384 tests): 4,078 of 7,073
  tests had raw p < 0.05 and none survived -- including the agent undercutting
  counter_based by 5.19 transmissions per informed vehicle at rural d=80 while
  losing every seed pair (p = 0.00195 -> 0.74);
* one metric x all cells of a split (32-96 tests): zero of 7,680 survived,
  since 0.00195 x 32 = 0.0625.

Per-cell families are 4 tests (0.00195 x 4 = 0.0078), so a consistent effect
in one cell can be significant. The price is stated rather than hidden: **no
correction is applied across the ~100 cells**, because 10 paired seeds cannot
support one. A marker therefore means "this cell's four metrics were corrected
together", not "corrected across the grid" (user decision).
**Effect sizes are the primary evidence**:
the rank-biserial correlation and the median paired difference are reported
next to p, and a significance marker without them means nothing at this seed
count. Writes results/stats/grid_tests.csv and grid_tests_summary.txt.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from analysis.metrics import METRIC_DIRECTION
from analysis.stats import holm_bonferroni, paired_test
from common.config import RESULTS_DIR

HEADLINE_METRICS = ("rwcr", "tx_per_at_risk_informed", "tir_median_s",
                    "actionable_deadline_miss_rate")
CELL = ["scenario", "density", "weather", "hazard_type"]
SPLITS = ["topology_split", "hazard_split", "weather_split"]
AGENT = "ai_harp"


def _score(values: Any, metric: str) -> float:
    """Mean in the metric's preferred direction (higher = better)."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) * METRIC_DIRECTION.get(metric, 1) if arr.size else -np.inf


def fixed_best_baselines(df, metrics: Sequence[str] = HEADLINE_METRICS) -> dict[str, str]:
    """Per metric, the single baseline with the best mean over fully-training cells."""
    base = df[df["policy"] != AGENT]
    train = base
    for s in SPLITS:
        train = train[train[s] == "train"]
    out = {}
    for m in metrics:
        # Mean of per-cell means, so dense cells with more finite rows don't dominate.
        per_cell = train.groupby(["policy"] + CELL)[m].mean().groupby("policy").mean()
        per_cell = per_cell.dropna()
        if per_cell.empty:
            continue
        out[m] = str((per_cell * METRIC_DIRECTION.get(m, 1)).idxmax())
    return out


def _paired(df_cell, a_policy: str, a_tau: str, b_policy: str, metric: str):
    a = df_cell[(df_cell["policy"] == a_policy) & (df_cell["tau"].astype(str) == a_tau)]
    b = df_cell[df_cell["policy"] == b_policy]
    joined = a[["seed", metric]].merge(b[["seed", metric]], on="seed", suffixes=("_a", "_b"))
    return joined[f"{metric}_a"].to_numpy(float), joined[f"{metric}_b"].to_numpy(float)


def grid_tests(df, metrics: Sequence[str] = HEADLINE_METRICS, alpha: float = 0.05):
    """One row per (comparison, tau, cell, metric), Holm-corrected within families."""
    import pandas as pd

    df = df.copy()
    df["tau"] = df["tau"].fillna("").astype(str)
    agent = df[df["policy"] == AGENT]
    if agent.empty:
        return pd.DataFrame()
    fixed = fixed_best_baselines(df, metrics)
    rows: list[dict[str, Any]] = []
    tests = []
    for key, cell_df in df.groupby(CELL + SPLITS, sort=True):
        cell = dict(zip(CELL + SPLITS, key))
        baselines = cell_df[cell_df["policy"] != AGENT]
        for tau in sorted(cell_df[cell_df["policy"] == AGENT]["tau"].unique()):
            for m in metrics:
                scores = {p: _score(g[m], m) for p, g in baselines.groupby("policy")}
                refs = {"oracle_best": max(scores, key=scores.get) if scores else None,
                        "fixed_best": fixed.get(m)}
                for comparison, ref in refs.items():
                    if ref is None:
                        continue
                    a, b = _paired(cell_df, AGENT, tau, ref, m)
                    t = paired_test(a, b, m)
                    tests.append(t)
                    rows.append({"comparison": comparison, "tau": tau, **cell, "metric": m,
                                 "reference_policy": ref, "test": t})
    out = pd.DataFrame(rows)
    # One family per cell: Holm across the metrics tested in it. Anything wider
    # is unrejectable at 10 seeds -- 0.00195 x 26 > 0.05 (see the docstring).
    fam_cols = ["comparison", "tau"] + CELL
    for _, fam in out.groupby(fam_cols):
        holm_bonferroni(list(fam["test"]), alpha)
    for col, attr in (("n_pairs", "n_pairs"), ("median_difference", "median_difference"),
                      ("mean_difference", "mean_difference"), ("rank_biserial", "rank_biserial"),
                      ("p_value", "p_value"), ("p_holm", "p_adjusted"),
                      ("significant", "significant"), ("agent_better", "candidate_better"),
                      ("note", "note")):
        out[col] = [getattr(t, attr) for t in out["test"]]
    out["marker"] = [t.marker for t in out["test"]]
    return out.drop(columns=["test"])


def summarise(tests) -> str:
    """Significant wins / losses per family and metric."""
    if tests.empty:
        return "no agent rows in runs.csv"
    lines = ["Agent vs references: significant (Holm) wins / losses / n.s., per split.",
             "Holm runs across the 4 metrics WITHIN each cell; [n] counts the cells",
             "summarised on that line, which are NOT corrected across (10 paired seeds",
             "cannot support it: 0.00195 x 26 > 0.05). Read r_rb (rank-biserial, +1 = the",
             "agent wins every seed) as the evidence and p as a filter.", ""]
    fam_cols = ["comparison", "tau"] + SPLITS
    for key, fam in tests.groupby(fam_cols):
        lines.append(" | ".join(f"{c}={v}" for c, v in zip(fam_cols, key)))
        for m, g in fam.groupby("metric"):
            sig = g[g["significant"]]
            win = int(sig["agent_better"].sum())
            loss = int(len(sig) - win)
            lines.append(f"   {m:32s} [{len(g):3d}] wins {win:3d}  losses {loss:3d}  "
                         f"n.s. {len(g) - len(sig):3d}  median r_rb "
                         f"{g['rank_biserial'].median():+.2f}  median |diff| "
                         f"{g['median_difference'].abs().median():.3f}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import pandas as pd

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", default=str(RESULTS_DIR / "runs.csv"))
    ap.add_argument("--out-dir", default=str(RESULTS_DIR / "stats"))
    args = ap.parse_args(argv)
    from analysis.metrics import METRICS_VERSION

    df = pd.read_csv(args.runs)
    df = df[df["metrics_version"] == METRICS_VERSION]
    tests = grid_tests(df)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tests.to_csv(out / "grid_tests.csv", index=False)
    text = summarise(tests)
    (out / "grid_tests_summary.txt").write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
