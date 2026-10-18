"""Pre-registration precision justification (computed from EXISTING records, before any new run).

::

    python -m experiments.revision5.precision

Paired per-seed relative cost difference at rural d=20 between the full-model policy (training
seeds 0-4, each at the suppression bias that was its cheapest qualifying setting on the legacy
eval seeds) and the adaptive classical pick slotted_1p(n_slots=30) (R4), on legacy eval seeds 0-9:
r_ij = cost_policy(train i, eval j) / cost_adaptive(eval j) - 1.
Variance components by one-way random-effects ANOVA: between training seeds (sigma_t^2) and within
(sigma_e^2). For n_train x n_eval: SE = sqrt(sigma_t^2/n_t + sigma_e^2/(n_t n_e)); 95% CI half-width
= 1.96 SE; minimum detectable effect (80% power, two-sided alpha 0.05) = (1.96 + 0.8416) SE.
Caveat recorded in the output: the policy's bias here was chosen on the same legacy seeds (the new
protocol separates them), and only 5 training seeds inform sigma_t.
"""

from __future__ import annotations

import json

import numpy as np

from experiments.revision1.provenance import out_dir, stamp, write_json
from experiments.revision1.runner import load_rows
from experiments.revision2.common import AGENT, SWEEPS


def main() -> int:
    r1 = json.loads((AGENT.parent.parent / "revision2" / "R1" / "r1_tail_latency.json").read_text())
    cell = r1["headline"]["bounds"]["unbounded"]["cells"]["rural_highway/d=20/clear/fog_bank"]["agent"]
    bias = {int(s): v["agent_bias"] for s, v in cell.items()}
    adapt = {int(r["seed"]): float(r["tx_per_at_risk_informed"]) for r in load_rows(SWEEPS / "headline_runs.csv")
             if r["scenario"] == "rural_highway" and float(r["density"]) == 20.0
             and r["policy"] == "slotted_1p" and r["value"] == "30"}
    R = []
    for s, b in sorted(bias.items()):
        rows = [r for r in load_rows(AGENT / f"ref_s{s}.csv")
                if r["scenario"] == "rural_highway" and float(r["density"]) == 20.0
                and float(json.loads(r["params_json"]).get("tau", -1)) == 0.5
                and not json.loads(r["params_json"]).get("deterministic") and float(r["value"]) == b]
        R.append([float(r["tx_per_at_risk_informed"]) / adapt[int(r["seed"])] - 1
                  for r in sorted(rows, key=lambda r: int(r["seed"]))])
    R = np.array(R)                                  # [n_train=5, n_eval=10]
    nt, ne = R.shape
    msb = ne * np.var(R.mean(axis=1), ddof=1)
    msw = np.mean(np.var(R, axis=1, ddof=1))
    s2e = msw
    s2t = max((msb - msw) / ne, 0.0)
    grid = {}
    for n_e in (10, 20, 30):
        for n_t in (5, 10):
            se = np.sqrt(s2t / n_t + s2e / (n_t * n_e))
            grid[f"n_eval={n_e}|n_train={n_t}"] = {"se": float(se), "ci95_half_width": float(1.96 * se),
                                                    "mde_80pct_power": float((1.96 + 0.8416) * se)}
    target = grid["n_eval=30|n_train=10"]["ci95_half_width"]
    out = {"per_train_seed_bias": bias, "paired_relative_diff_mean": float(R.mean()),
           "per_train_seed_mean": R.mean(axis=1).tolist(), "sigma_t": float(np.sqrt(s2t)), "sigma_e": float(np.sqrt(s2e)),
           "grid": grid, "half_width_30x10": target, "meets_5pct": bool(target <= 0.05),
           "caveats": ["policy bias chosen on the same legacy seeds (the new protocol separates selection and scoring)",
                       "sigma_t estimated from only 5 training seeds", "rural d=20 only; other cells differ"],
           "provenance": stamp("revision5/precision", {"cell": "rural_highway d=20", "adaptive": "slotted_1p(n_slots=30)"},
                               seeds={"eval_legacy": list(range(10)), "train": list(range(5))})}
    write_json(out_dir("precision", 5) / "precision.json", out)
    print(json.dumps({k: v for k, v in out.items() if k != "provenance"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
