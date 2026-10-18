"""R12: random-gate control for full-model seeds 0-4 (C3 extended).

::

    python -m experiments.revision3.r12_random_gate

Per training seed: the learned confidence gate (tau = 0.5), the network alone (tau = 0),
and a random gate deferring to weighted_p at exactly the learned gate's measured fallback
rate in each (cell, bias) (agents/ai_harp.py random_gate_rate; rates from that seed's own
tau = 0.5 rows). Each variant scored alone against the 55 baseline settings with Algorithm 1:
cells matched, rural d=20 cost / regret / margin.
"""

from __future__ import annotations

import json

import numpy as np

from analysis.comparator import compare_policy
from experiments.revision1.provenance import out_dir, stamp, write_json
from experiments.revision1.runner import cells_from_rows, load_rows
from experiments.revision2.common import AGENT, agent_rows, baseline_rows, fin

R20 = "rural_highway/d=20/clear/fog_bank"


def score(base, rows):
    res = compare_policy(cells_from_rows(base + rows), "ai_harp")
    get = lambda m: fin(next(v for k, v in m.items() if str(k) == R20))  # noqa: E731
    return {"cells_matched": 4 - res.cells_failed,
            "matched": [str(k).replace("/clear/fog_bank", "") for k, v in res.agent_cost.items() if np.isfinite(v)],
            "rural_d20_cost": get(res.agent_cost), "rural_d20_regret": get(res.regret_per_cell),
            "rural_d20_margin": get(res.margin_per_cell)}


def main() -> int:
    base = baseline_rows("headline")
    out = {"seeds": {}}
    for s in range(5):
        rg = [r for r in load_rows(AGENT / f"ref_s{s}_randomgate.csv") if "random_gate_rate" in json.loads(r["params_json"])]
        if len(rg) < 240:
            out["seeds"][s] = {"incomplete": len(rg)}
            continue
        out["seeds"][s] = {"learned_gate": score(base, agent_rows([f"ref_s{s}"], 0.5)),
                           "network_only": score(base, agent_rows([f"ref_s{s}"], 0.0)),
                           "random_gate": score(base, rg)}
    done = [s for s, v in out["seeds"].items() if "incomplete" not in v]
    for arm in ("learned_gate", "network_only", "random_gate"):
        cm = [out["seeds"][s][arm]["cells_matched"] for s in done]
        rg = [out["seeds"][s][arm]["rural_d20_regret"] for s in done if out["seeds"][s][arm]["rural_d20_regret"] is not None]
        out[f"{arm}_summary"] = {"cells_matched_mean": float(np.mean(cm)) if cm else None,
                                 "cells_matched_sd": float(np.std(cm, ddof=1)) if len(cm) > 1 else None,
                                 "rural_d20_regret_mean": float(np.mean(rg)) if rg else None,
                                 "rural_d20_regret_sd": float(np.std(rg, ddof=1)) if len(rg) > 1 else None,
                                 "n_seeds": len(cm)}
    out["provenance"] = stamp("R12", {"rows": "results/revision1/agent_runs/ref_s{0..4}{,_randomgate}.csv"},
                              seeds={"eval": list(range(10)), "train": done})
    write_json(out_dir("R12", 3) / "r12_random_gate.json", out)
    f = lambda v, p=False: "–" if v is None else (f"{100 * v:+.1f}%" if p else f"{v:.3f}")  # noqa: E731
    for s, v in out["seeds"].items():
        if "incomplete" in v:
            print(s, "incomplete", v["incomplete"]); continue
        print(f"seed {s}: " + " | ".join(f"{a}: {v[a]['cells_matched']} cells {v[a]['matched']} r20 {f(v[a]['rural_d20_cost'])} "
                                       f"({f(v[a]['rural_d20_regret'], True)})" for a in ("learned_gate", "network_only", "random_gate")))
    for arm in ("learned_gate", "network_only", "random_gate"):
        print(arm, out[f"{arm}_summary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
