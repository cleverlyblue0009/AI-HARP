"""B4: the statistics Section VI promises, in the four headline cells.

::

    python -m experiments.revision1.b4_stats bootstrap
    python -m experiments.revision1.b4_stats wilcoxon --agent-rows results/revision1/agent_runs/ref_s0.csv

wilcoxon
    Per headline cell: the agent against (a) the best single fixed baseline and
    (b) the per-cell oracle-best, each at the setting Algorithm 1 selects, on
    the 10 paired eval seeds. Wilcoxon signed-rank (analysis.stats.paired_test,
    the same call analysis/grid_stats.py makes), Holm across the four metrics
    within each (cell, comparison) family, matched-pairs rank-biserial effect
    size. The agent's setting is its cheapest qualifying bias (Algorithm 1's
    upper anchor; the matched cost itself is an interpolation and has no
    per-seed values); in a cell it never reaches, the bias with the highest
    mean RWCR. Baseline settings are the qualifying knob values k*.

bootstrap
    Paired-difference bootstrap 95% CIs of mean RWCR between every pair of the
    nine default baselines at rural d=20: (i) clear/fog_bank, pairs = seeds
    (n = 10); (ii) Table 2's pooled view, pairs = (weather, hazard, seed)
    (n = 200). 10,000 resamples of pairs, percentile intervals, fixed RNG seed.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.comparator import best_fixed_baseline, per_cell_oracle_best
from analysis.stats import holm_bonferroni, paired_test
from common.config import PROJECT_ROOT
from experiments.revision1.provenance import out_dir, rel, stamp, write_json
from experiments.revision1.runner import cells_from_rows, load_rows

R = PROJECT_ROOT / "results"
METRICS = ("rwcr", "tx_per_at_risk_informed", "tir_median_s", "actionable_deadline_miss_rate")
N_BOOT = 10_000
BOOT_SEED = 20261016


def bootstrap_ci(d: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    idx = rng.integers(0, d.size, size=(N_BOOT, d.size))
    means = d[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def part_bootstrap() -> dict:
    runs = pd.read_csv(R / "runs.csv", low_memory=False)
    base = runs[(runs.scenario == "rural_highway") & (runs.density_veh_km_lane == 20)
                & (runs.policy != "ai_harp")]
    rng = np.random.default_rng(BOOT_SEED)
    out = {}
    for view, sub in (("clear_fog_bank_n10", base[(base.weather == "clear")
                                                  & (base.hazard_type == "fog_bank")]),
                      ("table2_pooled_n200", base)):
        wide = sub.pivot_table(index=["weather", "hazard_type", "seed"], columns="policy",
                               values="rwcr")
        pols = sorted(wide.columns)
        sd_unpaired = {p: float(wide[p].std(ddof=1)) for p in pols}
        pairs = []
        seeds = wide.index.get_level_values("seed").to_numpy()
        useeds = np.unique(seeds)
        for a, b in itertools.combinations(pols, 2):
            d = (wide[a] - wide[b]).to_numpy()
            lo, hi = bootstrap_ci(d, rng)
            e = {"a": a, "b": b, "n": int(d.size), "mean_diff": float(d.mean()),
                 "sd_paired_diff": float(d.std(ddof=1)), "ci95": [lo, hi],
                 "ci_excludes_0": bool(lo > 0 or hi < 0)}
            if len(useeds) < d.size:
                # The 200 pooled pairs share 10 mobility seeds: resample seeds
                # (clusters of 20 conditions) as the honest interval.
                sums = np.array([d[seeds == s].sum() for s in useeds])
                cnts = np.array([(seeds == s).sum() for s in useeds])
                idx = rng.integers(0, len(useeds), size=(N_BOOT, len(useeds)))
                m = sums[idx].sum(axis=1) / cnts[idx].sum(axis=1)
                clo, chi = float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))
                e.update(ci95_seed_cluster=[clo, chi], cluster_excludes_0=bool(clo > 0 or chi < 0))
            pairs.append(e)
        out[view] = {"per_policy_sd": sd_unpaired, "pairs": pairs,
                     "n_pairs_ci_excludes_0": sum(p["ci_excludes_0"] for p in pairs),
                     "n_pairs_cluster_ci_excludes_0": sum(p.get("cluster_excludes_0", False)
                                                          for p in pairs),
                     "n_pairs": len(pairs)}
    return out


def _setting_rows(rows, policy, value):
    v = "" if value is None else str(value)
    sel = [r for r in rows if r["policy"] == policy and str(r["value"]) == v]
    return sorted(sel, key=lambda r: int(r["seed"]))


def _qualifying(cell, policy, tq, mb):
    pts = [p for p in cell.curves[policy].points if p.quality >= tq and cell.within_miss_bound(p, mb)]
    return min(pts, key=lambda p: p.cost) if pts else None


def part_wilcoxon(agent_rows_path: Path, tau: float, mode: str) -> dict:
    base_rows = load_rows(R / "revision1" / "B_sweeps" / "headline_runs.csv")
    base_rows = [r for r in base_rows if r["param"] or r["policy"] == "flooding"]  # 55 knob settings
    agent_rows = []
    for r in load_rows(agent_rows_path):
        p = json.loads(r["params_json"])
        if float(p.get("tau", -1)) == tau and bool(p.get("deterministic")) == (mode == "argmax") \
                and "random_gate_rate" not in p:
            agent_rows.append(r)
    cells = cells_from_rows(base_rows + agent_rows)
    oracle = per_cell_oracle_best(cells, 0.95, "cost", exclude=("ai_harp",))
    fixed = best_fixed_baseline(cells, 0.95, "cost", exclude=("ai_harp",))
    out = {"fixed_best": fixed.label, "tau": tau, "mode": mode, "cells": {}}
    for c in cells:
        tq = c.resolve_target(0.95, "relative", {"ai_harp"})
        mb = oracle.miss_bound[c.key]
        ag = _qualifying(c, "ai_harp", tq, mb)
        reaches = ag is not None
        if ag is None:
            ag = max(c.curves["ai_harp"].points, key=lambda p: p.quality)
        a_rows = [r for r in agent_rows if (r["scenario"], float(r["density"])) ==
                  (c.key.scenario, c.key.density) and float(r["value"]) == float(ag.value)]
        a_rows = sorted(a_rows, key=lambda r: int(r["seed"]))
        cell_rows = [r for r in base_rows if (r["scenario"], float(r["density"])) ==
                     (c.key.scenario, c.key.density)]
        refs = {"fixed_best": (fixed.policy, fixed.value)}
        op = oracle.label(c.key)
        if op != "none":
            q = _qualifying(c, op, tq, mb)
            refs["oracle_best"] = (op, None if q is None else q.value)
        entry = {"agent_bias": ag.value, "agent_reaches_target": reaches, "comparisons": {}}
        for name, (pol, val) in refs.items():
            b_rows = _setting_rows(cell_rows, pol, val)
            tests = []
            for m in METRICS:
                a = [float(r[m]) if r[m] not in ("", "nan") else np.nan for r in a_rows]
                b = [float(r[m]) if r[m] not in ("", "nan") else np.nan for r in b_rows]
                tests.append(paired_test(a, b, m))
            holm_bonferroni(tests)
            entry["comparisons"][name] = {
                "reference": f"{pol}({val})" if val is not None else pol,
                "tests": [{"metric": t.metric, "n": t.n_pairs, "median_diff": t.median_difference,
                           "mean_diff": t.mean_difference, "rank_biserial": t.rank_biserial,
                           "p": t.p_value, "p_holm": t.p_adjusted, "significant": t.significant,
                           "agent_better": bool(t.candidate_better), "note": t.note}
                          for t in tests]}
        out["cells"][str(c.key)] = entry
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("part", choices=["bootstrap", "wilcoxon"])
    ap.add_argument("--agent-rows", default=str(R / "revision1" / "agent_runs" / "ref_s0.csv"))
    ap.add_argument("--taus", type=float, nargs="*", default=[0.5, 0.0])
    ap.add_argument("--mode", default="sampled", choices=["sampled", "argmax"])
    args = ap.parse_args(argv)
    d = out_dir("B4")
    if args.part == "bootstrap":
        res = part_bootstrap()
        res["provenance"] = stamp("B4/bootstrap", {"n_boot": N_BOOT, "rng_seed": BOOT_SEED,
                                                   "source": "results/runs.csv"},
                                  seeds=list(range(10)))
        write_json(d / "b4_bootstrap.json", res)
        rows = [{"view": v, **{k: p[k] for k in ("a", "b", "n", "mean_diff", "sd_paired_diff")},
                 "ci_lo": p["ci95"][0], "ci_hi": p["ci95"][1], "excludes_0": p["ci_excludes_0"],
                 "cluster_ci_lo": p.get("ci95_seed_cluster", [np.nan])[0],
                 "cluster_ci_hi": p.get("ci95_seed_cluster", [np.nan, np.nan])[1],
                 "cluster_excludes_0": p.get("cluster_excludes_0")}
                for v in ("clear_fog_bank_n10", "table2_pooled_n200") for p in res[v]["pairs"]]
        pd.DataFrame(rows).to_csv(d / "b4_bootstrap.csv", index=False)
        for v in ("clear_fog_bank_n10", "table2_pooled_n200"):
            print(v, res[v]["n_pairs_ci_excludes_0"], "/", res[v]["n_pairs"])
    else:
        res = {f"tau={t:g}": part_wilcoxon(Path(args.agent_rows), t, args.mode) for t in args.taus}
        res["provenance"] = stamp("B4/wilcoxon", {"agent_rows": rel(args.agent_rows),
                                                  "baseline_rows": "results/revision1/B_sweeps/headline_runs.csv",
                                                  "metrics": METRICS, "holm_family": "4 metrics per (cell, comparison)"},
                                  seeds=list(range(10)))
        write_json(d / f"b4_wilcoxon_{args.mode}.json", res)
        flat = []
        for tk, r in res.items():
            if tk == "provenance":
                continue
            for ck, e in r["cells"].items():
                for cn, comp in e["comparisons"].items():
                    for t in comp["tests"]:
                        flat.append({"tau": tk, "cell": ck, "agent_bias": e["agent_bias"],
                                     "agent_reaches": e["agent_reaches_target"], "comparison": cn,
                                     "reference": comp["reference"], **t})
        pd.DataFrame(flat).to_csv(d / f"b4_wilcoxon_{args.mode}.csv", index=False)
        print(pd.DataFrame(flat)[["tau", "cell", "comparison", "reference", "metric", "median_diff",
                                  "rank_biserial", "p", "p_holm", "significant"]].to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
