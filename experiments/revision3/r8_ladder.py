"""R8: the diagnostic ladder -- what does the policy gain from a privileged reward / input?

::

    python -m experiments.revision3.r8_ladder

Rungs (seeds 0, 1 each; same config, curriculum, budget as the reference):
  A  the reference agent: causal reward, causal inputs (checkpoints/campaign/ref = seed 0,
     checkpoints/revision1/ref_s1 = seed 1)
  B  ORACLE-relevance reward and oracle training targets; inputs causal (deployable inputs)
  C  rung B + oracle relevance as an extra node input (NON-DEPLOYABLE upper bound)
Evaluation: 4 headline cells x 10 eval seeds, sampled, tau 0.5 and 0, Algorithm 1 as in the
paper. Training side: pooled shortfall per (scenario, density) group over the last 200 finetune
updates, from each run's history.jsonl -- measured against that rung's OWN targets (causal for A,
oracle for B/C), so the two are not on the same scale.
"""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

from analysis.comparator import compare_policy
from common.config import PROJECT_ROOT
from experiments.revision1.provenance import out_dir, stamp, write_json
from experiments.revision1.runner import cells_from_rows
from experiments.revision2.common import agent_rows, baseline_rows, fin

CK = PROJECT_ROOT / "checkpoints"
RUNS = {("A", 0): ("ref_s0", CK / "campaign" / "ref"), ("A", 1): ("ref_s1", CK / "revision1" / "ref_s1"),
        ("B", 0): ("rungB_s0", CK / "revision1" / "rungB_s0"), ("B", 1): ("rungB_s1", CK / "revision1" / "rungB_s1"),
        ("C", 0): ("rungC_s0", CK / "revision1" / "rungC_s0"), ("C", 1): ("rungC_s1", CK / "revision1" / "rungC_s1")}


def training_side(d):
    h = [json.loads(l) for l in (d / "history.jsonl").read_text().splitlines() if l.strip()]
    ft = [r for r in h if r.get("stage") == "finetune"][-200:]
    acc = defaultdict(list)
    for r in ft:
        for g, v in (r.get("shortfall_by_group") or {}).items():
            acc[g].append(v)
    summ = json.loads((d / "run_summary.json").read_text()) if (d / "run_summary.json").exists() else {}
    return {"mean_shortfall_last200": {g: float(np.mean(v)) for g, v in sorted(acc.items())},
            "share_updates_met_last200": {g: float(np.mean(np.array(v) <= 0)) for g, v in sorted(acc.items())},
            "pooled_shortfall_last200": float(np.mean([r.get("shortfall_pooled", np.nan) for r in ft])),
            "stopped": summ.get("stopped"), "bit_reproducible": summ.get("bit_reproducible"),
            "resumed_from": summ.get("resumed_from")}


def main() -> int:
    base = baseline_rows("headline")
    out = {"runs": {}}
    for (rung, seed), (name, d) in RUNS.items():
        e = {"rung": rung, "seed": seed, "checkpoint_dir": str(d.relative_to(PROJECT_ROOT)).replace("\\", "/")}
        for tau in (0.5, 0.0):
            rows = agent_rows([name], tau)
            if len({(r["scenario"], r["density"]) for r in rows}) < 4:
                e[f"tau={tau:g}"] = {"incomplete": len(rows)}
                continue
            res = compare_policy(cells_from_rows(base + rows), "ai_harp")
            e[f"tau={tau:g}"] = {"cells_matched": 4 - res.cells_failed,
                                 "per_cell": {str(k).replace("/clear/fog_bank", ""): {
                                     "cost": fin(res.agent_cost[k]), "regret": fin(res.regret_per_cell[k]),
                                     "margin": fin(res.margin_per_cell[k]),
                                     "best_rwcr": max(p.quality for p in next(c for c in cells_from_rows(rows)
                                                                              if c.key == k).curves["ai_harp"].points)}
                                     for k in res.agent_cost}}
        e["training"] = training_side(d)
        out["runs"][f"{rung}_s{seed}"] = e
    out["provenance"] = stamp("R8", {"rungs": {"A": "causal reward + inputs", "B": "oracle reward, causal inputs",
                                               "C": "oracle reward + oracle input (NON-DEPLOYABLE)"},
                                     "targets_B_C": "results/revision3/R8/oracle_coverage_targets.json"},
                              seeds={"eval": list(range(10)), "train": [0, 1]})
    write_json(out_dir("R8", 3) / "r8_ladder.json", out)
    f = lambda v, p=False: "–" if v is None else (f"{100 * v:+.1f}%" if p else f"{v:.3f}")  # noqa: E731
    for k, e in out["runs"].items():
        for tau in ("tau=0.5", "tau=0"):
            x = e[tau]
            if "incomplete" in x:
                print(k, tau, "incomplete", x["incomplete"]); continue
            cells = " | ".join(f"{c}: {f(v['cost'])} ({f(v['regret'], True)}) bestRWCR {v['best_rwcr']:.3f}" for c, v in x["per_cell"].items())
            print(f"{k} {tau}: matched {x['cells_matched']} | {cells}")
        t = e["training"]
        print(f"   training pooled shortfall {t['pooled_shortfall_last200']:+.3f}; sparse: " +
              ", ".join(f"{g} {v:+.3f}" for g, v in t["mean_shortfall_last200"].items() if g.endswith(("|2", "|3", "|1"))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
