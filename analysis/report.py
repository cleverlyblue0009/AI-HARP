"""Phase 8: regenerate every figure and table from committed results.

``python -m analysis.report``

Reads only committed artefacts under ``results/`` (plus the reference training
history) -- it never re-runs the simulator -- so the paper's figures and tables
are a pure function of the recorded evidence. Anything that cannot be built
from those files is skipped with a stated reason rather than silently omitted
or quietly faked.

Numbering
---------
``fig01``..``fig10`` are the paper's ten figures, in the order the campaign
brief numbers them:

1.  overhead vs RWCR Pareto front, per cell (headline)
2.  per-cell oracle-best vs best-fixed headroom
3.  agent regret and margin across all cells
4.  causal vs oracle estimation agreement
5.  confidence-gate trade-off: cost and fallback rate vs tau
6.  training curves: shortfall and lambda per group
7.  attention over one real dissemination event
8.  ablation bar chart
9.  latency / overhead inversion
10. simulator validation against a published curve

Everything else is named ``figS*``. Supporting plots are useful, but they are
not paper figures, and a shared numbering would let one be mistaken for the
other.

Refuses to mix metric eras: rows whose ``metrics_version`` differs from the
current one are dropped, loudly.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from analysis.metrics import METRICS_VERSION
from common.config import RESULTS_DIR, load_yaml
from common.logging_utils import get_logger

logger = get_logger("analysis.report")

#: Retrained ablations, in the order the ablation figure and table list them.
#: Keys are the result directory names; values are display labels.
ABLATIONS: dict[str, str] = {
    "ref": "full",
    "gcn": "GCN encoder",
    "mlp": "MLP encoder",
    "star": "star graph",
    "heads1": "1 attention head",
    "heads8": "8 attention heads",
    "k4": "neighbour cap 4",
    "no_relevance": "no causal relevance",
    "long_wait": "long deferrals",
}


def load_runs_current() -> Any:
    """Load results/runs.csv, keeping only the current metrics era."""
    import pandas as pd

    path = RESULTS_DIR / "runs.csv"
    if not path.exists():
        logger.warning("no results/runs.csv; density figures will be skipped")
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    if "metrics_version" not in df.columns:
        logger.warning(
            "results/runs.csv has no metrics_version column: every row predates "
            "the oracle/causal split and is NOT comparable. Skipping. "
            "Regenerate with experiments/full_sweep.py."
        )
        return pd.DataFrame()
    keep = df[df["metrics_version"] == METRICS_VERSION]
    if len(keep) != len(df):
        logger.warning("dropped %d row(s) from older metric versions",
                       len(df) - len(keep))
    return keep


def _json(path: Path) -> Any | None:
    """Parse a JSON artefact, or None if it is absent or unreadable."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _agent_eval(variant: str = "ref", mode: str = "sampled") -> Any | None:
    return _json(RESULTS_DIR / f"agent_{variant}" / mode / "agent_evaluation.json")


def _lambda_cap() -> float | None:
    """The multiplier cap from configs/agent.yaml, for the training figure."""

    def walk(node: Any) -> float | None:
        if isinstance(node, dict):
            if "lambda_max" in node:
                try:
                    return float(node["lambda_max"])
                except (TypeError, ValueError):
                    return None
            for value in node.values():
                got = walk(value)
                if got is not None:
                    return got
        return None

    try:
        return walk(load_yaml("agent.yaml"))
    except Exception:                                   # pragma: no cover
        return None


def _slug(cell_key: str) -> str:
    return (cell_key.replace("/", "_").replace("=", "").replace(".", "p")
            .replace(" ", ""))


def _markers_for_cell(scenario: str, density: float,
                      comparison: str = "fixed_best") -> dict[str, dict[str, str]]:
    """Significance markers for the agent, from results/stats/grid_tests.csv.

    Read from the artefact grid_stats.py wrote rather than recomputed here: two
    independent computations of one test are two chances to disagree, and the
    paper would then contain both answers.
    """
    import pandas as pd

    path = RESULTS_DIR / "stats" / "grid_tests.csv"
    if not path.exists():
        return {}
    tests = pd.read_csv(path)
    sub = tests[(tests["scenario"] == scenario) & (tests["comparison"] == comparison)]
    col = "density_veh_km_lane" if "density_veh_km_lane" in sub.columns else "density"
    sub = sub[sub[col] == density]
    out: dict[str, dict[str, str]] = {}
    for _, row in sub.iterrows():
        marker = row.get("marker")
        if isinstance(marker, str) and marker:
            key = f"ai_harp@{float(row['tau']):g}"
            out.setdefault(key, {})[str(row["metric"])] = marker
    return out


def _attention_matrix(
    payload: dict[str, Any], max_rows: int = 12, max_cols: int = 14,
) -> tuple[Any, list[str], list[str], int] | None:
    """Attention over the successive decisions of one dissemination event.

    Columns are the event's network-driven decisions in time order; rows are
    the neighbour RANKS within each decision, best-attended first.

    Rank, not vehicle id, is the axis that survives: each decision is taken by
    a different holder with a different neighbour set, so a row keyed by
    vehicle would be empty almost everywhere. Keyed by rank, a column shows how
    sharply that decision concentrated its attention, and the figure shows
    whether the policy picks a clear relay or spreads thin.

    Gate fallbacks are excluded. On those decisions the analytic policy chose
    the action and the attention did not select anything, so plotting them
    would credit the network with decisions it did not make.
    """
    decisions = [d for d in (payload.get("decisions") or [])
                 if not d.get("used_fallback")]
    if not decisions:
        return None

    ranked: list[tuple[int, int, list[float]]] = []
    for d in decisions:
        weights = sorted((float(a) for a in d["attention"] if np.isfinite(a)),
                         reverse=True)
        if weights:
            ranked.append((int(d["step"]), int(d["holder"]), weights))
    if not ranked:
        return None

    ranked.sort(key=lambda r: (r[0], r[1]))
    if len(ranked) > max_cols:                  # sample evenly across the event
        take = np.linspace(0, len(ranked) - 1, max_cols).round().astype(int)
        ranked = [ranked[i] for i in dict.fromkeys(take)]

    depth = min(max_rows, max(len(w) for _, _, w in ranked))
    matrix = np.full((depth, len(ranked)), np.nan)
    for j, (_, _, weights) in enumerate(ranked):
        for i in range(min(depth, len(weights))):
            matrix[i, j] = weights[i]

    row_labels = [f"rank {i + 1}" for i in range(depth)]
    # Several decisions can fall in one simulation step. Label the first column
    # of each step and leave the rest blank: repeating "t52" three times reads
    # as three identical columns rather than three decisions within one step.
    col_labels, previous = [], None
    for step, _, _ in ranked:
        col_labels.append("" if step == previous else f"t{step}")
        previous = step
    return matrix, row_labels, col_labels, len(ranked)


def _ablation_rows(mode: str = "sampled", axis: str = "cost") -> list[dict[str, Any]]:
    """Per-variant regret/margin summaries from the ablation evaluations.

    The spread is ACROSS CELLS, not across seeds: each evaluation records one
    operating point per cell, so a seed-level band does not exist here. The
    figure's axis label says which spread it shows.
    """
    rows: list[dict[str, Any]] = []
    for variant, label in ABLATIONS.items():
        payload = _agent_eval(variant, mode)
        if not payload:
            continue
        per_cell = (payload.get("axes", {}).get(axis, {}) or {}).get("per_cell", {})
        regret = np.array([float(v.get("regret", np.nan)) for v in per_cell.values()])
        margin = np.array([float(v.get("margin", np.nan)) for v in per_cell.values()])
        finite_regret = regret[np.isfinite(regret)]
        finite_margin = margin[np.isfinite(margin)]
        rows.append({
            "variant": label,
            "regret": ((float(finite_regret.mean()), float(finite_regret.std()))
                       if finite_regret.size else (np.nan, np.nan)),
            "margin": ((float(finite_margin.mean()), float(finite_margin.std()))
                       if finite_margin.size else (np.nan, np.nan)),
            "cells_matched": (float(finite_regret.size), float("nan")),
            "n_cells": len(per_cell),
        })
    return rows


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


def build_all(skip_agent: bool = False) -> dict[str, list[str]]:
    from analysis import figures as F
    from analysis import tables as T

    made: dict[str, list[str]] = {"figures": [], "tables": [], "skipped": []}
    F.apply_ieee_style()

    # --- 1, 2: Pareto fronts and per-cell headroom -------------------------
    cells_path = RESULTS_DIR / "pareto_cells.json"
    if cells_path.exists():
        from analysis.comparator import load_cells

        cells = load_cells()
        for i, cell in enumerate(cells):
            name = f"fig01_pareto_{cell.key.scenario}_d{cell.key.density:g}"
            F.fig_pareto(cells, cell_index=i, name=name)
            made["figures"].append(name)
        F.fig_headroom(cells)
        made["figures"].append("fig02_headroom")
        made["tables"].append(T.reference_table(cells).stem)
    else:
        made["skipped"].append("fig01/fig02 + reference table: no pareto_cells.json")

    # --- 3: agent regret and margin over every cell ------------------------
    evaluation = None if skip_agent else _agent_eval("ref")
    if evaluation:
        per_cell = (evaluation.get("axes", {}).get("cost", {}) or {}).get("per_cell", {})
        if per_cell:
            F.fig_regret_margin(per_cell)
            made["figures"].append("fig03_regret_margin")
        else:
            made["skipped"].append("fig03 regret/margin: evaluation has no per-cell costs")
    else:
        made["skipped"].append("fig03 regret/margin: no results/agent_ref/sampled "
                               "evaluation")

    # --- 4: causal vs oracle estimation agreement --------------------------
    risk_path = RESULTS_DIR / "risk_estimation.csv"
    if risk_path.exists():
        import pandas as pd

        F.fig_estimation_agreement(pd.read_csv(risk_path))
        made["figures"].append("fig04_estimation_agreement")
    else:
        made["skipped"].append("fig04 estimation agreement: no risk_estimation.csv")

    # --- 5: confidence gate, cost and fallback rate vs tau -----------------
    sweep = None if skip_agent else _json(
        RESULTS_DIR / "agent_ref" / "sampled" / "gate_sweep.json")
    if sweep:
        for key, points in sweep.items():
            name = f"fig05_gate_{_slug(key)}"
            F.fig_gate_tradeoff(
                [p["tau"] for p in points], [p["fallback_rate"] for p in points],
                [p["cost"] for p in points],
                perf_label="Tx per at-risk informed", name=name,
            )
            made["figures"].append(name)
        headline = next((k for k in sweep if "d=20" in k), next(iter(sweep)))
        points = sweep[headline]
        F.fig_gate_tradeoff([p["tau"] for p in points],
                            [p["fallback_rate"] for p in points],
                            [p["cost"] for p in points],
                            perf_label="Tx per at-risk informed")
        made["figures"].append("fig05_gate")
    else:
        made["skipped"].append("fig05 gate trade-off: no gate_sweep.json")

    # --- 6: training curves, shortfall and lambda per group ----------------
    history_path = (RESULTS_DIR.parent / "checkpoints" / "campaign" / "ref"
                    / "history.jsonl")
    if not skip_agent and history_path.exists() and history_path.stat().st_size:
        rows = [json.loads(line) for line
                in history_path.read_text(encoding="utf-8").splitlines() if line]
        F.fig_training_constraint(rows, lambda_cap=_lambda_cap())
        made["figures"].append("fig06_training")
        F.fig_training_curves([rows])
        made["figures"].append("figS4_training_reward")
    else:
        made["skipped"].append("fig06 training curves: no campaign/ref/history.jsonl")

    # --- 7: attention over one real dissemination event --------------------
    event = None if skip_agent else _json(RESULTS_DIR / "attention_event.json")
    built = _attention_matrix(event) if event else None
    if built:
        matrix, row_labels, col_labels, n_decisions = built
        F.fig_attention_heatmap(matrix, row_labels, col_labels,
                                xlabel="Decision, in time order",
                                ylabel="Neighbour rank")
        made["figures"].append("fig07_attention")
        logger.info("attention figure: %d ranks x %d decisions (of %d network "
                    "decisions in the event)", matrix.shape[0], matrix.shape[1],
                    n_decisions)
    else:
        made["skipped"].append("fig07 attention: no results/attention_event.json "
                               "(regenerate with experiments/attention_event.py)")

    # --- 8: ablations ------------------------------------------------------
    ablations = [] if skip_agent else _ablation_rows()
    if len(ablations) >= 2:
        full = next((r for r in ablations if r["variant"] == "full"), None)
        F.fig_ablation(
            [r["variant"] for r in ablations],
            [r["regret"][0] for r in ablations], [r["regret"][1] for r in ablations],
            ylabel="Mean regret vs per-cell oracle-best (spread across cells)",
            baseline=full["regret"][0] if full else None,
            # Each mean covers only the cells that variant matched, so the
            # counts must travel with the bars: a variant that matched more
            # cells is doing better even at higher regret.
            notes=[f"{int(r['cells_matched'][0])}/{r['n_cells']} cells matched"
                   for r in ablations],
        )
        made["figures"].append("fig08_ablation")
        made["tables"].append(
            T.ablation_table(ablations, ["regret", "margin", "cells_matched"]).stem)
    else:
        made["skipped"].append(
            f"fig08 ablation + ablation table: only {len(ablations)} scored "
            "variant(s); run experiments/evaluate_ablations.py")

    # --- 9, plus the supporting density and weather plots ------------------
    df = load_runs_current()
    if not df.empty:
        head_scenario, head_density = "rural_highway", 20.0
        has_headline = bool(((df["scenario"] == head_scenario)
                             & (df["density_veh_km_lane"] == head_density)).any())
        if has_headline:
            F.fig_latency_cost_inversion(df, head_scenario, head_density)
            made["figures"].append("fig09_latency_cost_inversion")
        else:
            made["skipped"].append("fig09 latency/cost inversion: headline cell absent")

        for scenario in sorted(df["scenario"].unique()):
            sub = df[df["scenario"] == scenario]
            if sub["density_veh_km_lane"].nunique() < 2:
                continue
            for metric, ylabel, stem in (
                ("rwcr", "RWCR", "figS1_rwcr_density"),
                ("tir_median_s", r"TIR median (s)", "figS2_tir_median_density"),
                ("tir_p95_s", r"TIR p95 (s)", "figS2b_tir_p95_density"),
            ):
                if metric not in sub:
                    continue
                name = f"{stem}_{scenario}"
                F.fig_metric_vs_density(df, metric, scenario, ylabel, name)
                made["figures"].append(name)

        # Weather: SECONDARY, traffic-mediated. The caption must say so --
        # there is no channel-degradation claim in this project.
        if df["weather"].nunique() > 1:
            for scenario in sorted(df["scenario"].unique()):
                name = f"figS3_weather_{scenario}"
                _fig_weather(F, df, scenario, name)
                made["figures"].append(name)
        else:
            made["skipped"].append("figS3 weather: only one weather in runs.csv")

        # Main comparison: one table per cell, markers from the committed tests.
        metrics = [m for m in ("rwcr", "tir_median_s", "tx_per_at_risk_informed",
                               "actionable_deadline_miss_rate", "pdr") if m in df]
        for (scenario, density), _ in df.groupby(["scenario", "density_veh_km_lane"]):
            name = f"table_main_{scenario}_d{float(density):g}"
            made["tables"].append(T.main_comparison_table(
                df, metrics, scenario, float(density),
                tests=_markers_for_cell(scenario, float(density)), name=name).stem)
        if has_headline:
            made["tables"].append(T.main_comparison_table(
                df, metrics, head_scenario, head_density,
                tests=_markers_for_cell(head_scenario, head_density)).stem)
    else:
        made["skipped"].append("fig09 + supporting figures + main tables: "
                               "no current-era rows in runs.csv")

    # --- 10: simulator validation against a published curve ----------------
    validation = _json(RESULTS_DIR / "validation" / "amador2022.json")
    if validation and validation.get("per_density"):
        per_density = validation["per_density"]
        keys = sorted(per_density, key=float)
        xs = [float(k) for k in keys]
        ours = [float(per_density[k]["pdr_mean"]) for k in keys]
        published = [float(per_density[k]["paper"]) for k in keys]
        F.fig_validation(xs, ours, published, xlabel="Vehicle density (veh/km)",
                         ylabel="Packet delivery ratio",
                         source=str(validation.get("reference", "published"))[:40])
        made["figures"].append("fig10_validation")
    else:
        made["skipped"].append("fig10 validation: no results/validation/amador2022.json")

    # --- always available --------------------------------------------------
    made["tables"].append(T.parameter_table().stem)
    made["tables"].append(T.notation_table().stem)
    return made


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
