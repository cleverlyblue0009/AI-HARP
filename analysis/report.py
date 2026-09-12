"""Phase 8: regenerate every figure and table from committed results.

``python -m analysis.report``

Reads only ``results/runs.csv`` and ``results/pareto_cells.json`` -- never
re-runs the simulator -- so the paper's artefacts are a pure function of the
recorded evidence. Anything that cannot be built from those files is skipped
with a stated reason rather than silently omitted or faked.

Refuses to mix metric eras: rows whose ``metrics_version`` differs from the
current one are dropped, loudly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from analysis.metrics import METRICS_VERSION
from common.config import RESULTS_DIR
from common.logging_utils import get_logger

logger = get_logger("analysis.report")


def load_runs_current() -> Any:
    """Load results/runs.csv, keeping only the current metrics era."""
    import pandas as pd

    path = RESULTS_DIR / "runs.csv"
    if not path.exists():
        logger.warning("no results/runs.csv; density figures will be skipped")
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "metrics_version" not in df.columns:
        logger.warning(
            "results/runs.csv has no metrics_version column: every row predates "
            "the oracle/causal split and is NOT comparable. Skipping. "
            "Regenerate with experiments/compare.py."
        )
        return pd.DataFrame()
    keep = df[df["metrics_version"] == METRICS_VERSION]
    if len(keep) != len(df):
        logger.warning("dropped %d row(s) from older metric versions",
                       len(df) - len(keep))
    return keep


def build_all(skip_agent: bool = False) -> dict[str, list[str]]:
    from analysis import figures as F
    from analysis import tables as T

    made: dict[str, list[str]] = {"figures": [], "tables": [], "skipped": []}
    F.apply_ieee_style()

    # --- cells: Pareto + reference points ---------------------------------
    cells_path = RESULTS_DIR / "pareto_cells.json"
    cells = None
    if cells_path.exists():
        from analysis.comparator import load_cells

        cells = load_cells()
        for i, cell in enumerate(cells):
            nm = f"fig02_pareto_{cell.key.scenario}_d{cell.key.density:g}"
            F.fig_pareto(cells, cell_index=i, name=nm)
            made["figures"].append(nm)
        made["tables"].append(T.reference_table(cells).stem)
    else:
        made["skipped"].append("fig02 Pareto + reference table: no pareto_cells.json")

    # --- density sweeps ----------------------------------------------------
    df = load_runs_current()
    if not df.empty:
        for scenario in sorted(df["scenario"].unique()):
            sub = df[df["scenario"] == scenario]
            if sub["density_veh_km_lane"].nunique() < 2:
                continue
            for metric, ylabel, nm in (
                ("rwcr", "RWCR", "fig03_rwcr_density"),
                ("tir_median_s", r"TIR median (s)", "fig04_tir_median_density"),
                ("tir_p95_s", r"TIR p95 (s)", "fig04b_tir_p95_density"),
            ):
                if metric not in sub:
                    continue
                name = f"{nm}_{scenario}"
                F.fig_metric_vs_density(df, metric, scenario, ylabel, name)
                made["figures"].append(name)

        # Weather: SECONDARY, traffic-mediated. The caption must say so --
        # there is no channel-degradation claim in this project.
        if df["weather"].nunique() > 1:
            for scenario in sorted(df["scenario"].unique()):
                name = f"fig05_weather_{scenario}"
                _fig_weather(F, df, scenario, name)
                made["figures"].append(name)
        else:
            made["skipped"].append("fig05 weather: only one weather condition in runs.csv")

        metrics = [m for m in ("rwcr", "tir_median_s", "tx_per_at_risk_informed",
                               "actionable_deadline_miss_rate", "pdr") if m in df]
        d = float(sorted(df["density_veh_km_lane"].unique())[len(df["density_veh_km_lane"].unique()) // 2])
        sc = sorted(df["scenario"].unique())[0]
        made["tables"].append(T.main_comparison_table(df, metrics, sc, d).stem)
    else:
        made["skipped"].append("fig03/04/05 + main table: no current-era rows in runs.csv")

    # --- agent-dependent artefacts ----------------------------------------
    hist = RESULTS_DIR.parent / "checkpoints" / "history.jsonl"
    if not skip_agent and hist.exists() and hist.stat().st_size:
        rows = [json.loads(line) for line in hist.read_text().splitlines() if line]
        F.fig_training_curves([rows])
        made["figures"].append("fig07_training")
    else:
        made["skipped"].append("fig07 training curves: no checkpoints/history.jsonl")

    made["skipped"].append("fig06 attention heatmap: needs a trained checkpoint")
    made["skipped"].append("fig08 ablation + ablation table: needs the ablation sweep")
    made["skipped"].append("fig09 gate trade-off: needs the tau sweep")
    made["skipped"].append("fig10 validation: needs Phase 7b (SUMO + published curve)")

    # --- always available --------------------------------------------------
    made["tables"].append(T.parameter_table().stem)
    made["tables"].append(T.notation_table().stem)
    return made


def _fig_weather(F, df, scenario: str, name: str) -> None:
    """RWCR by weather -- traffic-mediated, and the caption says so."""
    import matplotlib.pyplot as plt

    F.apply_ieee_style()
    fig, ax = plt.subplots(figsize=(F.SINGLE_COL, F.SINGLE_COL * 0.72))
    sub = df[df["scenario"] == scenario]
    order = ["clear", "moderate_rain", "heavy_rain", "dense_fog"]
    conditions = [w for w in order if w in set(sub["weather"])]
    panel = [p for p in F.FAMILIES["topology-aware"] if p in set(sub["policy"])]

    for pol in panel:
        d = sub[sub["policy"] == pol]
        mu = [d[d["weather"] == w]["rwcr"].mean() for w in conditions]
        sd = [d[d["weather"] == w]["rwcr"].std() for w in conditions]
        st = F.style_for(pol, panel)
        ax.errorbar(range(len(conditions)), mu, yerr=sd, label=F.label_for(pol), **st)

    ax.set_xticks(range(len(conditions)), [w.replace("_", " ") for w in conditions],
                  rotation=15, ha="right")
    ax.set_ylabel("RWCR")
    ax.set_xlabel("Weather (effect is traffic-mediated, not channel)")
    ax.legend(loc="best")
    F.save(fig, name)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Regenerate paper figures and tables")
    ap.add_argument("--skip-agent", action="store_true")
    args = ap.parse_args(argv)

    made = build_all(skip_agent=args.skip_agent)
    print(f"\nfigures ({len(made['figures'])}):")
    for f in made["figures"]:
        print(f"  {f}")
    print(f"tables ({len(made['tables'])}):")
    for t in made["tables"]:
        print(f"  {t}")
    if made["skipped"]:
        print(f"\nnot generated ({len(made['skipped'])}) -- stated, not silently omitted:")
        for s in made["skipped"]:
            print(f"  - {s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
