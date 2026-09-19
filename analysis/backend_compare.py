"""Does a conclusion flip between the fallback mobility backend and SUMO?

::

    D:/aiharp-env/python.exe -m analysis.backend_compare

Reads two committed sweeps -- ``results/runs.csv`` (fallback) and
``results/runs_sparse_sumo.csv`` (SUMO) -- and compares them on the cells they
share. Nothing is re-simulated, so this is a statement about the evidence the
paper ships.

What is compared, and why
-------------------------
Absolute numbers are *expected* to differ: the fallback pre-places vehicles
and has no lane changing, so its traffic is more evenly spread and its sparse
corridors are better connected (see the platooning measurement in the README).
Every claim in the paper is comparative -- regret against a per-cell
oracle-best, margin over a fixed baseline -- so the question that decides
whether those claims survive is whether the **ordering of policies** is
preserved, not whether RWCR matches.

So this reports, per cell and metric:

* the best policy under each backend, and whether it changed;
* Kendall's tau between the two policy orderings (+1 identical, -1 reversed);
* the paired per-seed difference for the policy each backend prefers.

Splits are never pooled. Held-out hazards and weathers are reported as their
own strata, because a conclusion that holds on the training split and fails on
the held-out one is exactly the thing pooling would hide.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from analysis.metrics import METRIC_DIRECTION, METRICS_VERSION
from common.config import RESULTS_DIR
from common.logging_utils import get_logger

logger = get_logger("analysis.backend_compare")

#: Reported metrics. Cost and coverage first: they carry the paper's claims.
METRICS: tuple[str, ...] = (
    "tx_per_at_risk_informed", "rwcr", "tir_median_s",
    "actionable_deadline_miss_rate",
)

CELL = ["scenario", "density_veh_km_lane"]
SPLITS = ["hazard_split", "weather_split"]


def kendall_tau(a: Sequence[float], b: Sequence[float]) -> float:
    """Rank correlation between two orderings of the same items.

    Written out rather than imported: the project avoids a SciPy dependency in
    the analysis path, and with nine policies the quadratic form is free.
    Ties contribute zero to the numerator and are excluded from both
    denominators (tau-b).
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    keep = np.isfinite(a) & np.isfinite(b)
    a, b = a[keep], b[keep]
    if a.size < 2:
        return float("nan")
    concordant = discordant = ties_a = ties_b = 0
    for i, j in itertools.combinations(range(a.size), 2):
        da, db = a[i] - a[j], b[i] - b[j]
        if da == 0 and db == 0:
            continue
        if da == 0:
            ties_a += 1
        elif db == 0:
            ties_b += 1
        elif (da > 0) == (db > 0):
            concordant += 1
        else:
            discordant += 1
    n0 = concordant + discordant
    denom = np.sqrt((n0 + ties_a) * (n0 + ties_b))
    return float((concordant - discordant) / denom) if denom else float("nan")


def _load(fallback_path: Path, sumo_path: Path):
    import pandas as pd

    fb = pd.read_csv(fallback_path, low_memory=False)
    if "metrics_version" in fb.columns:
        fb = fb[fb["metrics_version"] == METRICS_VERSION]
    su = pd.read_csv(sumo_path, low_memory=False)
    if "metrics_version" in su.columns:
        su = su[su["metrics_version"] == METRICS_VERSION]

    # Only the cells and policies both backends actually ran. The SUMO sweep
    # carries baselines only, so the agent is not in this comparison.
    cells = set(map(tuple, su[CELL].drop_duplicates().to_numpy()))
    fb = fb[[tuple(r) in cells for r in fb[CELL].to_numpy()]]
    policies = sorted(set(fb["policy"]) & set(su["policy"]))
    fb = fb[fb["policy"].isin(policies)]
    su = su[su["policy"].isin(policies)]
    return fb, su, policies


def compare(fallback_path: Path, sumo_path: Path) -> tuple[Any, str]:
    """Per (cell, split, metric) ordering agreement between the backends."""
    import pandas as pd

    fb, su, policies = _load(fallback_path, sumo_path)
    keys = CELL + SPLITS
    rows: list[dict[str, Any]] = []

    for key, f_group in fb.groupby(keys, sort=True):
        s_group = su
        for col, value in zip(keys, key):
            s_group = s_group[s_group[col] == value]
        if s_group.empty:
            continue
        for metric in METRICS:
            if metric not in f_group or metric not in s_group:
                continue
            f_mean = f_group.groupby("policy")[metric].mean()
            s_mean = s_group.groupby("policy")[metric].mean()
            shared = [p for p in policies if p in f_mean.index and p in s_mean.index]
            if len(shared) < 2:
                continue
            direction = METRIC_DIRECTION.get(metric, 1)
            pick = (max if direction > 0 else min)
            f_best = pick(shared, key=lambda p: f_mean[p])
            s_best = pick(shared, key=lambda p: s_mean[p])
            rows.append({
                **dict(zip(keys, key)), "metric": metric,
                "n_policies": len(shared),
                "fallback_best": f_best, "sumo_best": s_best,
                "best_changed": f_best != s_best,
                "kendall_tau": kendall_tau([f_mean[p] for p in shared],
                                           [s_mean[p] for p in shared]),
                "fallback_best_value": float(f_mean[f_best]),
                "sumo_same_policy_value": float(s_mean[f_best]),
                "sumo_best_value": float(s_mean[s_best]),
                # How much the flip is worth, judged on SUMO: what the new
                # winner gains over the old one under the backend that changed
                # its mind. RWCR saturates, so two policies can swap places on
                # a difference of 0.001 -- that is noise being relabelled as a
                # conclusion, and it must not be counted as one.
                "flip_gain": abs(float(s_mean[s_best]) - float(s_mean[f_best])),
                "flip_gain_rel": (abs(float(s_mean[s_best]) - float(s_mean[f_best]))
                                  / max(abs(float(s_mean[f_best])), 1e-9)),
            })
    return pd.DataFrame(rows), summarise(pd.DataFrame(rows))


#: A flip counts as material when the new winner beats the old one by more
#: than this, relative, under the backend that changed its mind. Saturated
#: metrics swap their top two on differences far below any effect this paper
#: would report.
MATERIAL_REL = 0.05


def summarise(df, material_rel: float = MATERIAL_REL) -> str:
    """Whether the comparative conclusions survive the backend change."""
    if df.empty:
        return "no shared (cell, split, metric) strata to compare"
    df = df.copy()
    df["material"] = df["best_changed"] & (df["flip_gain_rel"] > material_rel)

    out: list[str] = []
    out.append("=" * 78)
    out.append(" BACKEND COMPARISON -- fallback vs SUMO, sparse cells")
    out.append("=" * 78)
    out.append("Absolute values are expected to differ (the fallback spreads vehicles")
    out.append("evenly; SUMO platoons them). What decides the paper's comparative")
    out.append("claims is whether the ORDERING of policies survives the change.")
    out.append("")

    for split_key, group in df.groupby(["hazard_split", "weather_split"], sort=True):
        hz, wx = split_key
        changed = int(group["best_changed"].sum())
        material = int(group["material"].sum())
        out.append(f"--- hazard_split={hz}, weather_split={wx} "
                   f"({len(group)} strata) ".ljust(78, "-"))
        out.append(f"  best policy changed in {changed} of {len(group)} "
                   f"(cell, metric) strata; {material} by more than "
                   f"{material_rel:.0%}")
        tau = group["kendall_tau"].to_numpy(float)
        tau = tau[np.isfinite(tau)]
        if tau.size:
            out.append(f"  Kendall tau of the policy ordering: median {np.median(tau):+.2f}, "
                       f"min {tau.min():+.2f}, max {tau.max():+.2f}")
        for metric, sub in group.groupby("metric"):
            flips = sub[sub["material"]]
            if len(flips):
                names = ", ".join(
                    f"{r['scenario'].replace('_highway','')} d={r['density_veh_km_lane']:g}: "
                    f"{r['fallback_best']} -> {r['sumo_best']} ({r['flip_gain_rel']:.0%})"
                    for _, r in flips.iterrows())
                out.append(f"    {metric:32s} {len(flips)}/{len(sub)} material | {names}")
            else:
                out.append(f"    {metric:32s} 0/{len(sub)} material "
                           f"({int(sub['best_changed'].sum())} flips, all below "
                           f"{material_rel:.0%})")
        out.append("")

    total_changed = int(df["best_changed"].sum())
    total_material = int(df["material"].sum())
    tau_all = df["kendall_tau"].to_numpy(float)
    tau_all = tau_all[np.isfinite(tau_all)]
    out.append("-" * 78)
    out.append(f" OVERALL: best policy changed in {total_changed} of {len(df)} strata "
               f"({100 * total_changed / len(df):.0f}%)")
    out.append(f" OVERALL: {total_material} of those flips exceed {material_rel:.0%} "
               f"({100 * total_material / len(df):.0f}% of all strata); the rest swap "
               f"near-tied policies")
    if tau_all.size:
        out.append(f" OVERALL: median Kendall tau {np.median(tau_all):+.2f} "
                   f"({int((tau_all > 0).sum())} of {tau_all.size} strata positive)")
    out.append("-" * 78)
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fallback", default=str(RESULTS_DIR / "runs.csv"))
    ap.add_argument("--sumo", default=str(RESULTS_DIR / "runs_sparse_sumo.csv"))
    ap.add_argument("--out", default=str(RESULTS_DIR / "backend_sparse_comparison"))
    args = ap.parse_args(argv)

    df, text = compare(Path(args.fallback), Path(args.sumo))
    print(text)
    out = Path(args.out)
    df.to_csv(out.with_suffix(".csv"), index=False)
    out.with_suffix(".txt").write_text(text + "\n", encoding="utf-8")
    logger.info("wrote %s.{csv,txt}", out.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
