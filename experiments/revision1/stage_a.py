"""Stage A: fact extraction from committed results -- no new simulation.

::

    python -m experiments.revision1.stage_a

Reads results/pareto_cells.json, results/agent_ref/*, results/runs.csv,
results/coverage_targets.json, results/feasibility/*, results/validation/* and
the (git-ignored) training histories under checkpoints/. Writes
results/revision1/A/stage_a_facts.json; REVISION_REPORT.md quotes from it.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analysis.comparator import compare_policy, load_cells, per_cell_oracle_best
from common.config import PROJECT_ROOT, load_yaml
from experiments.revision1.provenance import out_dir, rel, stamp, write_json

R = PROJECT_ROOT / "results"
CK = PROJECT_ROOT / "checkpoints"
HEADLINE = [("rural_highway", 2.0), ("rural_highway", 20.0), ("rural_highway", 80.0),
            ("urban_nlos", 20.0)]


def _f(x: Any) -> float:
    return float(x) if x is not None else float("nan")


# --------------------------------------------------------------------- A1 ---
def a1_targets() -> dict[str, Any]:
    out: dict[str, Any] = {}
    agent_cfg = load_yaml("agent.yaml")
    # (1) Evaluation / Algorithm 1 / Fig. 2: 0.95 x best RWCR of ANY baseline
    # setting in the cell (oracle RWCR, eval seeds 0-9), agent excluded, plus
    # the miss guard.
    cells = load_cells(R / "pareto_cells.json")
    oracle = per_cell_oracle_best(cells, 0.95, "cost", exclude=("ai_harp",))
    ev = {}
    for c in cells:
        ceil = c.achievable_quality(("ai_harp",))
        best_setting = max(((p.quality, p.label) for cu in c.curves.values()
                            for p in cu.points), key=lambda t: t[0])
        ev[str(c.key)] = {"ceiling_oracle_rwcr": ceil, "ceiling_setting": best_setting[1],
                          "target_0.95x": 0.95 * ceil,
                          "miss_bound": oracle.miss_bound[c.key],
                          "best_miss_rate": c.best_miss_rate(("ai_harp",)),
                          "n_settings_in_ceiling": sum(len(cu.points) for cu in c.curves.values())}
    out["evaluation_target"] = {
        "definition": "q* = 0.95 x max over every baseline knob setting in the cell of mean "
                      "ORACLE RWCR on eval seeds 0-9 (agent excluded); a setting also needs "
                      "actionable-miss <= cell best + 0.05",
        "code": "analysis/comparator.py Cell.resolve_target / resolve_miss_bound",
        "source_file": "results/pareto_cells.json", "per_cell": ev}

    # (2) Training q*_g: 0.95 x max(flooding, dvcast default) CAUSAL coverage,
    # per (scenario, density, weather, hazard) cell, 32 training seeds.
    ct = json.loads((R / "coverage_targets.json").read_text())
    frac = float(agent_cfg["objective"]["target_fraction"]) if "target_fraction" in \
        agent_cfg["objective"] else 0.95
    groups: dict[str, list[float]] = defaultdict(list)
    for k, v in ct["cells"].items():
        sc, d, w, h = k.split("|")
        groups[f"{sc}|{d}"].append(v["ceiling"])
    tr = {}
    for g, ceils in sorted(groups.items(), key=lambda kv: (kv[0].split("|")[0],
                                                           float(kv[0].split("|")[1]))):
        tr[g] = {"mean_ceiling_causal": float(np.mean(ceils)),
                 "mean_target": frac * float(np.mean(ceils)),
                 "min_target": frac * float(min(ceils)), "max_target": frac * float(max(ceils)),
                 "n_cells": len(ceils)}
    for sc, d in HEADLINE:
        k = f"{sc}|{d:g}|clear|fog_bank"
        if k in ct["cells"]:
            tr[f"{sc}|{d:g}"]["headline_cell_clear_fog_bank_target"] = frac * ct["cells"][k]["ceiling"]
    out["training_target"] = {
        "definition": f"q*_g = {frac} x max(flooding default, dvcast default) mean CAUSAL "
                      "coverage per (scenario, density, weather, hazard) cell on training "
                      "seeds 100-131; the multiplier for group (scenario, density) is stepped "
                      "on pooled shortfall (sum target - sum coverage)/sum target",
        "code": "agents/constrained_reward.py CoverageTargets; experiments/coverage_targets.py",
        "source_file": "results/coverage_targets.json",
        "provenance": ct["provenance"], "target_fraction": frac, "per_group": tr}

    # (3) Feasibility sweep (Sec. VII-F): training side used targets_path with
    # target_seeds recorded in its provenance; evaluation side used 0.95 x best
    # of 63 settings (55 knob settings + 8 registry defaults).
    fz = json.loads((R / "feasibility" / "sparse_feasibility.json").read_text())
    out["feasibility_target"] = {
        "definition_training": "same as training target, but computed from the ceiling file "
                               "AS IT STOOD when the sweep was summarised",
        "definition_evaluation": "0.95 x best mean oracle RWCR over 63 settings (eval seeds 0-9)",
        "source_file": "results/feasibility/sparse_feasibility.json",
        "provenance": fz["provenance"],
        "training_mean_target": {g: v["mean_target"] for g, v in fz["training"].items()},
        "training_settings_meeting": {g: len(v["settings_meeting_target"])
                                      for g, v in fz["training"].items()},
        "evaluation": {k: {"ceiling": v["ceiling"], "target": v["target"],
                           "n_meeting": v["n_meeting"], "n_settings": v["n_settings"]}
                       for k, v in fz["evaluation"].items()},
    }
    cur = {g: tr[g]["mean_target"] for g in fz["training"]}
    out["feasibility_target"]["current_32seed_mean_target"] = cur
    return out


# --------------------------------------------------------------------- A2 ---
def a2_matched_cost() -> dict[str, Any]:
    cells = load_cells(R / "agent_ref" / "sampled" / "pareto_cells_agent.json")
    res = compare_policy(cells, "ai_harp", 0.95, "cost", "relative", miss_margin=0.05)
    cell = next(c for c in cells if c.key.scenario == "rural_highway" and c.key.density == 20)
    tq = cell.resolve_target(0.95, "relative", {"ai_harp"})
    mb = res.oracle.miss_bound[cell.key]
    pts = sorted(cell.curves["ai_harp"].points, key=lambda p: p.value)
    reaching = [p for p in pts if p.quality >= tq and cell.within_miss_bound(p, mb)]
    best = min(reaching, key=lambda p: p.cost)
    short = [p for p in pts if p.quality < tq and p.cost < best.cost]
    lo = max(short, key=lambda p: p.cost)
    frac = (tq - lo.quality) / (best.quality - lo.quality)
    interp = lo.cost + frac * (best.cost - lo.cost)

    runs = pd.read_csv(R / "runs.csv", low_memory=False)
    sel = runs[(runs.scenario == "rural_highway") & (runs.density_veh_km_lane == 20)
               & (runs.weather == "clear") & (runs.hazard_type == "fog_bank")
               & (runs.policy == "ai_harp")]
    tab = {f"tau={t:g}": {"cost_mean": float(g.tx_per_at_risk_informed.mean()),
                          "rwcr_mean": float(g.rwcr.mean()), "n": int(len(g)),
                          "checkpoint_sha": str(g.checkpoint_sha.iloc[0])}
           for t, g in sel.groupby("tau")}
    return {
        "matched_cost": interp, "agent_evaluation_json_value": res.agent_cost[cell.key],
        "target_rwcr": tq, "miss_bound": mb,
        "swept_knob": "suppression_bias in {-2,-1,0,1,2,3} (experiments/evaluate_agent.py "
                      "DEFAULT_BIASES), gate tau=0.5, sampled policy, eval seeds 0-9",
        "curve": [{"bias": p.value, "rwcr": p.quality, "cost": p.cost,
                   "miss": p.extras.get("actionable_deadline_miss_rate"),
                   "tir_median_s": p.extras.get("tir_median_s"),
                   "fallback_rate": p.extras.get("gate_fallback_rate")} for p in pts],
        "upper_anchor": {"bias": best.value, "rwcr": best.quality, "cost": best.cost},
        "lower_anchor": {"bias": lo.value, "rwcr": lo.quality, "cost": lo.cost},
        "interp_fraction": frac,
        "computation": "PolicyCurve.overhead_at: cheapest setting meeting target and miss "
                       "bound (upper anchor), linearly interpolated toward the most expensive "
                       "setting that falls short and is cheaper (lower anchor). Anchors are "
                       "chosen by COST, not by adjacency in the bias knob.",
        "table2_rows_runs_csv": tab,
        "table2_source": "results/runs.csv rows policy=ai_harp at suppression_bias 0 "
                         "(single setting, not matched); results/tables/table_main.tex",
    }


# --------------------------------------------------------------------- A3 ---
def a3_spread() -> dict[str, Any]:
    runs = pd.read_csv(R / "runs.csv", low_memory=False)
    sel = runs[(runs.scenario == "rural_highway") & (runs.density_veh_km_lane == 20)
               & (runs.weather == "clear") & (runs.hazard_type == "fog_bank")
               & (runs.policy != "ai_harp")]
    m = sel.groupby("policy").rwcr.mean().sort_values()
    nine = list(load_yaml("experiment.yaml")["sweep"]["policies"])
    wo = m.drop("p_persistence_03")
    # Table 2 (analysis/tables.py main_comparison_table) pools EVERY weather x
    # hazard at the density: 4 x 5 x 10 = 200 rows per scheme.
    pool = runs[(runs.scenario == "rural_highway") & (runs.density_veh_km_lane == 20)]
    pb = pool[pool.policy != "ai_harp"]
    mp = pb.groupby("policy").rwcr.mean().sort_values()
    wp = mp.drop("p_persistence_03")
    agent = pool[pool.policy == "ai_harp"].groupby("tau")[
        ["rwcr", "tx_per_at_risk_informed", "actionable_deadline_miss_rate",
         "tir_median_s"]].mean()
    table2 = {"rows_per_scheme": int(pb.groupby("policy").size().iloc[0]),
              "rwcr_mean_by_policy": mp.to_dict(),
              "spread_all_nine": [float(mp.min()), float(mp.max()), float(mp.max() - mp.min())],
              "spread_without_p_persistence_03": [float(wp.min()), float(wp.max()),
                                                  float(wp.max() - wp.min())],
              "miss_actionable_by_policy": pb.groupby("policy")
              .actionable_deadline_miss_rate.mean().to_dict(),
              "agent_by_tau": agent.to_dict("index")}
    return {"table2_pooled_view": table2,
            "nine_baselines_in_code": nine, "code": "configs/experiment.yaml sweep.policies",
            "rwcr_mean_by_policy": m.to_dict(),
            "spread_all_nine": [float(m.min()), float(m.max()), float(m.max() - m.min())],
            "spread_without_p_persistence_03": [float(wo.min()), float(wo.max()),
                                                float(wo.max() - wo.min())],
            "default_p_persistence_is": "p_persistence_03 (p = 0.3)",
            "source_file": "results/runs.csv (rural_highway d=20 clear fog_bank, seeds 0-9)"}


# --------------------------------------------------------------------- A4 ---
def a4_miss() -> dict[str, Any]:
    runs = pd.read_csv(R / "runs.csv", low_memory=False)
    rur = runs[runs.scenario == "rural_highway"]
    keys = ["density_veh_km_lane", "weather", "hazard_type", "policy", "tau"]
    g = rur.fillna({"tau": -1}).groupby(keys)[["deadline_miss_rate",
                                                "actionable_deadline_miss_rate"]].mean()
    g = g.reset_index()
    near = g[((g.deadline_miss_rate - 0.240).abs() < 0.0005)
             & ((g.actionable_deadline_miss_rate - 0.102).abs() < 0.0005)]
    near_un = g[(g.deadline_miss_rate - 0.240).abs() < 0.0005]
    # Pooled-over-everything views.
    pooled = {}
    for name, sub in [("rural_all_rows", rur), ("rural_baselines", rur[rur.policy != "ai_harp"]),
                      ("rural_d20_clear_fog_bank_baselines",
                       rur[(rur.density_veh_km_lane == 20) & (rur.weather == "clear")
                           & (rur.hazard_type == "fog_bank") & (rur.policy != "ai_harp")])]:
        pooled[name] = {"deadline_miss_rate": float(sub.deadline_miss_rate.mean()),
                        "actionable_deadline_miss_rate": float(sub.actionable_deadline_miss_rate.mean())}
    by_density = rur[rur.policy != "ai_harp"].groupby("density_veh_km_lane")[
        ["deadline_miss_rate", "actionable_deadline_miss_rate"]].mean()
    return {"exact_matches_0.240_and_0.102": near.to_dict("records"),
            "matches_0.240_unrestricted_only": near_un.head(20).to_dict("records"),
            "pooled": pooled, "rural_baselines_by_density": by_density.to_dict("index"),
            "table2_miss_column": "actionable_deadline_miss_rate, rural d=20 clear fog_bank "
                                  "(results/tables/table_main.tex)",
            "definitions": "analysis/metrics.py deadline_miss_rate (all at-risk) vs "
                           "actionable_deadline_miss_rate (at-risk with time to act)"}


# --------------------------------------------------------------------- A5 ---
def _hist(path: Path) -> list[dict[str, Any]]:
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def a5_lambda() -> dict[str, Any]:
    runs = {"run8_cap50": CK / "run8" / "history.jsonl",
            "armA_cap500": CK / "feas_A_cap500" / "history.jsonl",
            "armB_init500": CK / "feas_B_init500" / "history.jsonl",
            "campaign_ref_cap500": CK / "campaign" / "ref" / "history.jsonl"}
    out: dict[str, Any] = {}
    for name, p in runs.items():
        if not p.exists():
            out[name] = {"missing": rel(p)}
            continue
        h = _hist(p)
        ft = [r for r in h if r.get("stage") == "finetune"] or h
        series: dict[str, list[tuple[int, float]]] = defaultdict(list)
        for r in h:
            for g, lam in (r.get("lambda_next_by_group") or {}).items():
                series[g].append((int(r["update"]), float(lam)))
        last_upd = ft[-1]["update"]
        per = {}
        for g, s in sorted(series.items()):
            u = np.array([a for a, _ in s]); v = np.array([b for _, b in s])
            last100 = v[u > last_upd - 100]
            last200 = v[u > last_upd - 200]
            per[g] = {"peak": float(v.max()), "peak_update": int(u[v.argmax()]),
                      "final": float(v[-1]),
                      "plateau_mean_last100": float(last100.mean()) if last100.size else None,
                      "plateau_mean_last200": float(last200.mean()) if last200.size else None,
                      "n_updates_seen": int(len(v))}
        sat = [r for r in h if r.get("lambda_saturated")]
        out[name] = {"history": rel(p), "updates": int(h[-1]["update"]),
                     "finetune_updates": len(ft), "updates_with_any_lambda_saturated": len(sat),
                     "lambda_max_cfg": None, "per_group": per,
                     "global_peak": max(v["peak"] for v in per.values())}
    return out


# --------------------------------------------------------------------- A7 ---
def a7_validation() -> dict[str, Any]:
    v = json.loads((R / "validation" / "amador2022.json").read_text())
    return {"validation_json_keys": list(v)[:20], "validation": v}


# --------------------------------------------------------------------- A8 ---
def a8_regret_margin() -> dict[str, Any]:
    d = json.loads((R / "agent_ref" / "sampled" / "agent_evaluation.json").read_text())
    pc = d["axes"]["cost"]["per_cell"]["rural_highway/d=20/clear/fog_bank"]
    a, o, f = pc["agent"], pc["oracle"], pc["fixed"]
    return {"agent": a, "oracle": o, "fixed": f,
            "regret_code_a_over_o_minus_1": a / o - 1, "margin_code_f_over_a_minus_1": f / a - 1,
            "margin_alt_(f-a)/f": (f - a) / f, "regret_alt_(a-o)/a": (a - o) / a,
            "f_minus_a": f - a, "a_minus_o": a - o,
            "reviewer_0.013/0.730": 0.013 / 0.730,
            "source_file": "results/agent_ref/sampled/agent_evaluation.json",
            "code": "analysis/comparator.py compare_policy: regret = a/o - 1, margin = f/a - 1"}


def main() -> int:
    facts = {"A1": a1_targets(), "A2": a2_matched_cost(), "A3": a3_spread(), "A4": a4_miss(),
             "A5": a5_lambda(), "A7": a7_validation(), "A8": a8_regret_margin()}
    facts["provenance"] = stamp("A", {"inputs": ["results/pareto_cells.json",
                                                  "results/agent_ref", "results/runs.csv",
                                                  "results/coverage_targets.json",
                                                  "results/feasibility",
                                                  "results/validation/amador2022.json",
                                                  "checkpoints/*/history.jsonl"]})
    p = write_json(out_dir("A") / "stage_a_facts.json", facts)
    print(f"wrote {rel(p)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
