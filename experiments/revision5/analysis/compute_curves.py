"""Performance vs compute (C5). Writes results/revision5/analysis/compute_curves.json.

    python -m experiments.revision5.analysis.compute_curves

Classical random-search tuning: for k in {5, 10, 20, 40, 55} settings x s in {5, 10, 20} SEL seeds, 200 repetitions
each: a random subset of k of the 55 classical settings and s of the 20 SEL seeds; the selection rule on that subset
(frozen q*, m*); the pick scored on all 30 EVAL seeds. Tuning compute = the summed cpu_s of the k x s SEL runs used.
Policy: the full model's cost and qualification at the budget checkpoints (1,000-2,000 updates, D4) against training
CPU-hours (history.jsonl main_cpu_s + rollout_cpu_s, cumulative). Seeds 0-2's first 1,000 updates predate CPU logging
(revision 5): their CPU is the mean cumulative CPU of ref_s5-s9 (same configuration), labelled as an estimate.
"""

from __future__ import annotations

import json
import math

import numpy as np

from experiments.revision5.analysis.common import (EVAL, HEADLINE, SEED_RNG, SEL, finite, rows, select, tables,
                                                  thresholds, write)

R1 = __import__("common.config", fromlist=["PROJECT_ROOT"]).PROJECT_ROOT / "results" / "revision1" / "checkpoints"
KS, SS, REPS = (5, 10, 20, 40, 55), (5, 10, 20), 200


def cum_cpu(run: str) -> list[float]:
    H = [json.loads(l) for l in (R1 / f"{run}_history.jsonl").open(encoding="utf-8")]
    c, out = 0.0, []
    for h in H:
        c += float(h.get("main_cpu_s", 0.0)) + float(h.get("rollout_cpu_s", 0.0))   # absent before revision-5 logging
        out.append(c)
    return out


def main() -> int:
    th = thresholds()
    sel = tables(rows("classical__SEL__grid24__default__fallback.csv"), HEADLINE, SEL, extra=("cpu_s",))
    ev = tables(rows("classical__EVAL__grid24__default__fallback.csv"), HEADLINE, EVAL)
    rng = np.random.default_rng(SEED_RNG)
    classical = {}
    for c in HEADLINE:
        q, m = th[c]
        res = []
        for k in KS:
            for s in SS:
                costs, quals, cpus = [], [], []
                for _ in range(REPS):
                    sub = rng.choice(len(sel[c].settings), k, replace=False)
                    seeds = rng.choice(len(SEL), s, replace=False)
                    mask = np.zeros(len(sel[c].settings), bool)
                    mask[sub] = True
                    j = select(sel[c], q, m, math.inf, seeds, mask)
                    cpus.append(float(np.nansum(sel[c].m["cpu_s"][np.ix_(sub, seeds)])))
                    if j < 0:
                        quals.append(False)
                        continue
                    e = ev[c].settings.index(sel[c].settings[j])
                    rw, mi, co = (float(ev[c].mean(x)[e]) for x in ("rwcr", "miss", "cost"))
                    ok = rw >= q and mi <= m
                    quals.append(ok)
                    if ok:
                        costs.append(co)
                res.append({"k_settings": k, "s_sel_seeds": s, "reps": REPS,
                            "tuning_cpu_hours_mean": float(np.mean(cpus)) / 3600,
                            "eval_qualification_rate": float(np.mean(quals)),
                            "eval_cost_median_qualifying": finite(np.median(costs)) if costs else None,
                            "eval_cost_iqr_qualifying": [finite(np.quantile(costs, 0.25)), finite(np.quantile(costs, 0.75))] if costs else None})
        classical[f"{c[0]} d={c[1]:g}"] = res

    est = [cum_cpu(f"ref_s{k}") for k in range(5, 10)]
    base1000 = [e[999] for e in est if len(e) >= 1000]
    base_mean, base_sd = float(np.mean(base1000)), float(np.std(base1000, ddof=1))
    policy = {}
    sec = (R1.parents[1] / "revision5" / "analysis" / "secondary.json")
    budget = json.loads(sec.read_text()).get("budget", {}) if sec.exists() else {}
    for k in range(3):
        ext = cum_cpu(f"ref_s{k}_u2000")          # 2,000 lines; updates 1,001-2,000 carry CPU fields
        ext_part = [e - ext[999] for e in ext]    # CPU spent in the extension only (exact)
        policy[f"ref_s{k}"] = {f"u{u}": {"training_cpu_hours_estimate": (base_mean + (ext_part[u - 1] if u > 1000 else 0.0)) / 3600}
                               for u in (1000, 1200, 1500, 1700, 2000)}
    payload = {"classical_random_search": classical,
               "policy_budget_cpu": policy,
               "policy_cpu_note": {"first_1000_updates_cpu_hours_estimate_mean": base_mean / 3600,
                                   "sd": base_sd / 3600, "source": "ref_s5-s9 histories (same configuration)",
                                   "extension_updates_cpu": "exact, from ref_s0-s2_u2000 histories"},
               "policy_budget_results": budget if "status" not in budget else {"status": budget["status"]}}
    print(write("compute_curves.json", payload, {"k": KS, "s": SS, "reps": REPS}))
    for c, r in classical.items():
        print(c, [(x["k_settings"], x["s_sel_seeds"], round(x["tuning_cpu_hours_mean"], 2), round(x["eval_qualification_rate"], 2)) for x in r][:4])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
