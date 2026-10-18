"""R8 rung A: is causal coverage (what training optimises) a usable surrogate for oracle
RWCR (what evaluation scores)? Training seeds only.

::

    python -m experiments.revision3.r8_rung_a_surrogate

Inputs (every run that has both quantities on training seeds 100-131):
results/revision3/R8/oracle_target_runs.csv (flooding + DV-CAST defaults, all 96 training
cells), results/revision1/B_sweeps/train_runs.csv (63 settings, rural d=20/80 and urban d=20,
clear/fog_bank), results/feasibility/sparse_feasibility_runs.csv (63 settings, rural/urban
d=2, 6 training cells). Per scenario: Pearson and Spearman correlation across runs, across
cell means, and WITHIN cells (setting means of the same cell -- the comparison training
actually has to make: which behaviour covers more), plus the scale gap.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from common.config import PROJECT_ROOT
from experiments.revision1.provenance import out_dir, stamp, write_json

R = PROJECT_ROOT / "results"


def load() -> pd.DataFrame:
    a = pd.read_csv(R / "revision3" / "R8" / "oracle_target_runs.csv")
    b = pd.read_csv(R / "revision1" / "B_sweeps" / "train_runs.csv")
    f = pd.read_csv(R / "feasibility" / "sparse_feasibility_runs.csv")
    f = f[f.split == "train"].rename(columns={"miss": "actionable_deadline_miss_rate"})
    cols = ["scenario", "density", "weather", "hazard_type", "policy", "param", "value", "seed",
            "causal_coverage", "rwcr"]
    df = pd.concat([x[cols].assign(source=n) for x, n in ((a, "oracle_targets"), (b, "train_sweep"),
                                                          (f, "sparse_feasibility"))])
    df["value"] = df["value"].astype(str)
    df["param"] = df["param"].fillna("").astype(str)
    return df.dropna(subset=["causal_coverage", "rwcr"]).drop_duplicates(
        subset=["scenario", "density", "weather", "hazard_type", "policy", "param", "value", "seed"])


def corr(x, y):
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return {"pearson": None, "spearman": None, "n": int(len(x))}
    return {"pearson": float(pearsonr(x, y)[0]), "spearman": float(spearmanr(x, y)[0]), "n": int(len(x))}


def main() -> int:
    df = load()
    out = {"n_runs": int(len(df)), "by_scenario": {}}
    for sc, g in df.groupby("scenario"):
        cell = ["density", "weather", "hazard_type"]
        cm = g.groupby(cell)[["causal_coverage", "rwcr"]].mean()
        within = []
        for _, gg in g.groupby(cell):
            sm = gg.groupby(["policy", "param", "value"])[["causal_coverage", "rwcr"]].mean()
            if len(sm) >= 5:
                c = corr(sm.causal_coverage.values, sm.rwcr.values)
                if c["spearman"] is not None:
                    within.append(c)
        by_d = {f"{d:g}": corr(gd.causal_coverage.values, gd.rwcr.values)["pearson"]
                for d, gd in g.groupby("density")}
        out["by_scenario"][sc] = {
            "across_runs": corr(g.causal_coverage.values, g.rwcr.values),
            "across_cell_means": corr(cm.causal_coverage.values, cm.rwcr.values),
            "within_cell_setting_means": {"n_cells": len(within),
                                          "median_spearman": float(np.median([w["spearman"] for w in within])) if within else None,
                                          "min_spearman": float(min(w["spearman"] for w in within)) if within else None,
                                          "median_pearson": float(np.median([w["pearson"] for w in within])) if within else None},
            "across_runs_by_density_pearson": by_d,
            "mean_causal_coverage": float(g.causal_coverage.mean()), "mean_rwcr": float(g.rwcr.mean())}
    out["provenance"] = stamp("R8/rungA", {"sources": ["results/revision3/R8/oracle_target_runs.csv",
                                                       "results/revision1/B_sweeps/train_runs.csv",
                                                       "results/feasibility/sparse_feasibility_runs.csv"]},
                              seeds=list(range(100, 132)))
    write_json(out_dir("R8", 3) / "r8_rung_a_surrogate.json", out)
    for sc, e in out["by_scenario"].items():
        print(sc, "runs", e["across_runs"], "| cell means", e["across_cell_means"], "| within cells",
              e["within_cell_setting_means"], "| mean causal %.3f oracle %.3f" % (e["mean_causal_coverage"], e["mean_rwcr"]))
        print("    by density pearson", {k: (None if v is None else round(v, 2)) for k, v in e["across_runs_by_density_pearson"].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
