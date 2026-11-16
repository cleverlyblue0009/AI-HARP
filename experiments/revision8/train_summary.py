"""Revision 8: training summary per arm and seed, read from results/revision8/train/*/ (history.jsonl, run_meta.json,
theta.json). Training-pool quantities only (causal coverage targets, fitness); no evaluation number.

    python -m experiments.revision8.train_summary      -> results/revision8/train_summary.json
"""

from __future__ import annotations

import json

from common.config import PROJECT_ROOT

T = PROJECT_ROOT / "results" / "revision8" / "train"


def main() -> int:
    out = {}
    for arm in ("plain", "rand"):
        for k in range(10):
            d = T / f"{arm}_s{k}"
            h = [json.loads(l) for l in (d / "history.jsonl").open()]
            meta = json.loads((d / "run_meta.json").read_text())
            th = json.loads((d / "theta.json").read_text())
            last10 = h[-10:]
            out[f"{arm}_s{k}"] = {
                "generations": th["generations"], "machine": meta.get("machine", "cloud-session"),
                "cpu": meta.get("cpu"), "numpy": meta["numpy"], "cma": meta["cma_version"],
                "feasibility_rate_last10_mean": sum(x["feasibility_rate"] for x in last10) / len(last10),
                "best_cost_last_gen": h[-1]["best_cost"], "best_f_last_gen": h[-1]["best_f"],
                "median_feasible_cost_last_gen": h[-1]["median_feasible_cost"],
                "sigma_last": h[-1]["sigma"], "wall_s_total": round(sum(x["wall_s"] for x in h), 1),
                "theta": th["theta"]}
    from experiments.revision1.provenance import stamp, write_json
    p = PROJECT_ROOT / "results" / "revision8" / "train_summary.json"
    write_json(p, {"provenance": stamp("revision8/train_summary", {}), "runs": out})
    for r, v in out.items():
        print(f"{r:9s} {v['machine']:14s} {str(v['cpu'])[:28]:28s} feas10 {v['feasibility_rate_last10_mean']:.2f} "
              f"bestcost {v['best_cost_last_gen']:.3f} sigma {v['sigma_last']:.3f} wall {v['wall_s_total']/60:.0f}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
