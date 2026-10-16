"""B3: feasibility of the 63 fixed settings under every target definition (A1).

::

    python -m experiments.revision1.b3_feasibility

Definitions (A1):

* ``E``   Algorithm 1: 0.95 x best mean oracle RWCR over the 55 knob settings
          (results/pareto_cells.json), eval seeds 0-9; reported with and
          without the deadline guard (miss <= cell best + 0.05).
* ``F63`` Sec. VII-F evaluation side: 0.95 x best mean oracle RWCR over all 63
          settings, eval seeds 0-9.
* ``T32`` training target now in force: 0.95 x max(flooding, dvcast default)
          causal coverage, 32 training seeds (results/coverage_targets.json).
* ``T2``  the stale 2-seed training target the feasibility sweep used
          (results/coverage_targets.json as of a378bcf).

Per definition and headline cell: how many of the 63 settings meet the target
on their seed mean; the per-episode hindsight envelope (best setting chosen
separately for every seed) and its pooled shortfall
(sum target - sum envelope)/sum target (negative = reachable with hindsight).
Each of the agent's misses is then classified:
``target unreachable`` -- no fixed setting meets it (so even per-cell tuning
of a fixed scheme cannot); ``policy failure`` -- at least one does.
Training-side definitions use the agent's training history (pooled group
shortfall over the last 200 finetune updates), the only causal-coverage
measurement of the agent on training seeds.
"""

from __future__ import annotations

import csv
import json
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np

from analysis.comparator import load_cells
from common.config import PROJECT_ROOT
from experiments.revision1.provenance import out_dir, stamp, write_json
from experiments.revision1.runner import load_rows

R = PROJECT_ROOT / "results"
HEADLINE = [("rural_highway", 2.0), ("rural_highway", 20.0), ("rural_highway", 80.0),
            ("urban_nlos", 20.0)]
FRAC = 0.95


def _f(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def label(r) -> str:
    return f"{r['policy']}({r['param']}={r['value']})" if r["param"] else f"{r['policy']}(default)"


def per_setting(rows, metric):
    """setting -> {seed: value}."""
    out: dict[str, dict[int, float]] = defaultdict(dict)
    for r in rows:
        out[label(r)][int(r["seed"])] = _f(r[metric])
    return out


def feasibility(table: dict[str, dict[int, float]], target_of_seed: dict[int, float],
                miss: dict[str, dict[int, float]] | None = None, miss_bound: float = np.inf) -> dict:
    seeds = sorted(target_of_seed)
    means = {s: float(np.nanmean([v[k] for k in seeds if k in v])) for s, v in table.items()}
    mtarget = float(np.mean([target_of_seed[k] for k in seeds]))
    ok = []
    for s, m in means.items():
        if m < mtarget:
            continue
        if miss is not None and np.isfinite(miss_bound):
            mm = float(np.nanmean([miss[s][k] for k in seeds if k in miss[s]]))
            if not mm <= miss_bound:
                continue
        ok.append(s)
    env = {k: max(v.get(k, -np.inf) for v in table.values()) for k in seeds}
    tsum = sum(target_of_seed[k] for k in seeds)
    best = max(means, key=means.get)
    return {"n_settings": len(table), "n_meeting": len(ok), "meeting": sorted(ok),
            "mean_target": mtarget, "best_setting": best, "best_setting_mean": means[best],
            "best_fixed_pooled_shortfall": (tsum - sum(table[best].get(k, np.nan) for k in seeds)) / tsum,
            "envelope_mean": float(np.mean(list(env.values()))),
            "envelope_pooled_shortfall": (tsum - sum(env.values())) / tsum,
            "n_seeds": len(seeds)}


def old_targets() -> dict:
    txt = subprocess.check_output(["git", "show", "a378bcf:results/coverage_targets.json"],
                                  cwd=PROJECT_ROOT, text=True)
    return json.loads(txt)


def agent_eval_best() -> dict:
    cells = load_cells(R / "agent_ref" / "sampled" / "pareto_cells_agent.json")
    return {(c.key.scenario, c.key.density): max(p.quality for p in c.curves["ai_harp"].points)
            for c in cells}


def agent_train_shortfall() -> dict:
    p = PROJECT_ROOT / "checkpoints" / "campaign" / "ref" / "history.jsonl"
    if not p.exists():
        return {}
    h = [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]
    ft = [r for r in h if r.get("stage") == "finetune"][-200:]
    acc = defaultdict(list)
    for r in ft:
        for g, v in (r.get("shortfall_by_group") or {}).items():
            acc[g].append(v)
    return {g: float(np.mean(v)) for g, v in acc.items()}


def main() -> int:
    ev_rows = load_rows(R / "revision1" / "B_sweeps" / "headline_runs.csv")
    tr_rows = load_rows(R / "revision1" / "B_sweeps" / "train_runs.csv")
    with (R / "feasibility" / "sparse_feasibility_runs.csv").open(encoding="utf-8") as fh:
        fz = [r for r in csv.DictReader(fh)]
    t32 = json.loads((R / "coverage_targets.json").read_text())["cells"]
    t2 = old_targets()["cells"]
    eval_cells = {(c.key.scenario, c.key.density): c for c in load_cells(R / "pareto_cells.json")}
    a_eval, a_train = agent_eval_best(), agent_train_shortfall()

    out: dict = {}
    for sc, d in HEADLINE:
        key = f"{sc}|{d:g}|clear|fog_bank"
        e_rows = [r for r in ev_rows if r["scenario"] == sc and float(r["density"]) == d]
        rw = per_setting(e_rows, "rwcr")
        miss = per_setting(e_rows, "actionable_deadline_miss_rate")
        cell = eval_cells[(sc, d)]
        qE = FRAC * cell.achievable_quality(("ai_harp",))
        mb = cell.best_miss_rate(("ai_harp",)) + 0.05
        best63 = max(float(np.nanmean(list(v.values()))) for v in rw.values())
        seeds = range(10)
        res = {
            "E_no_guard": feasibility(rw, {k: qE for k in seeds}),
            "E_with_guard": feasibility(rw, {k: qE for k in seeds}, miss, mb),
            "F63": feasibility(rw, {k: FRAC * best63 for k in seeds}),
        }
        if sc == "rural_highway" and d == 2.0:
            t_rows = [r for r in fz if r["scenario"] == sc and float(r["density"]) == d
                      and r["split"] == "train" and r["weather"] == "clear"
                      and r["hazard_type"] == "fog_bank"]
            t_source = "results/feasibility/sparse_feasibility_runs.csv"
        else:
            t_rows = [r for r in tr_rows if r["scenario"] == sc and float(r["density"]) == d]
            t_source = "results/revision1/B_sweeps/train_runs.csv"
        cov = per_setting(t_rows, "causal_coverage")
        tseeds = sorted({int(r["seed"]) for r in t_rows})
        res["T32"] = feasibility(cov, {k: FRAC * t32[key]["ceiling"] for k in tseeds})
        res["T2"] = feasibility(cov, {k: FRAC * t2[key]["ceiling"] for k in tseeds})
        for k in ("T32", "T2"):
            res[k]["source"] = t_source
        # classification of the agent's result under each definition
        cls = {}
        ab = a_eval[(sc, d)]
        for name in ("E_no_guard", "E_with_guard", "F63"):
            tgt = res[name]["mean_target"]
            miss_ = ab < tgt
            cls[name] = {"agent_best_rwcr": ab, "target": tgt, "agent_misses": bool(miss_),
                         "class": None if not miss_ else
                         ("policy failure" if res[name]["n_meeting"] > 0 else "target unreachable")}
        g = a_train.get(f"{sc}|{d:g}")
        for name in ("T32",):
            miss_ = g is not None and g > 0
            cls[name] = {"agent_group_shortfall_last200": g,
                         "note": "group pooled over 6 training weather x hazard cells; "
                                 "feasibility counted on the clear/fog_bank cell",
                         "agent_misses": bool(miss_), "class": None if not miss_ else
                         ("policy failure" if res[name]["n_meeting"] > 0 else "target unreachable")}
        res["agent_classification"] = cls
        out[key] = res
    out["provenance"] = stamp("B3", {"definitions": ["E_no_guard", "E_with_guard", "F63", "T32", "T2"],
                                     "eval_rows": "results/revision1/B_sweeps/headline_runs.csv",
                                     "train_rows": "results/revision1/B_sweeps/train_runs.csv",
                                     "rural_d2_train_rows": "results/feasibility/sparse_feasibility_runs.csv",
                                     "old_targets": "git show a378bcf:results/coverage_targets.json"},
                              seeds={"eval": list(range(10)), "train": list(range(100, 132))})
    write_json(out_dir("B3") / "b3_feasibility.json", out)
    for k, v in out.items():
        if k == "provenance":
            continue
        print(k)
        for name in ("E_no_guard", "E_with_guard", "F63", "T32", "T2"):
            x = v[name]
            print(f"  {name:13s} target {x['mean_target']:.4f}  meeting {x['n_meeting']:2d}/{x['n_settings']}"
                  f"  best {x['best_setting']} {x['best_setting_mean']:.4f}"
                  f"  envelope {x['envelope_mean']:.4f} shortfall {x['envelope_pooled_shortfall']:+.4f}")
        print("  agent:", {n: c["class"] for n, c in v["agent_classification"].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
