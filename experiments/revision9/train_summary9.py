"""Revision 9: training summary per STRUCT9 run (results/revision9/train/*/), training-pool quantities only.

    python -m experiments.revision9.train_summary9      -> results/revision9/train_summary.json
"""

from __future__ import annotations

import json

from common.config import PROJECT_ROOT

T = PROJECT_ROOT / "results" / "revision9" / "train"


def main() -> int:
    out = {}
    for arm in ("plain", "rand"):
        for k in range(5):
            d = T / f"{arm}_s{k}"
            h = [json.loads(l) for l in (d / "history.jsonl").open()]
            meta = json.loads((d / "run_meta.json").read_text())
            th = json.loads((d / "theta.json").read_text())
            last10 = h[-10:]
            out[f"{arm}_s{k}"] = {
                "generations": th["generations"], "machine": meta.get("machine"), "cpu": meta.get("cpu"),
                "python": meta["python"], "numpy": meta["numpy"], "cma": meta["cma_version"],
                "miss_bounds": meta.get("miss_bounds"),
                "feasibility_rate_last10_mean": sum(x["feasibility_rate"] for x in last10) / len(last10),
                "best_cost_last_gen": h[-1]["best_cost"], "best_f_last_gen": h[-1]["best_f"],
                "best_miss_by_group_last_gen": h[-1].get("best_miss_by_group"),
                "wall_s_total": round(sum(x["wall_s"] for x in h), 1), "theta": th["theta"]}
    from experiments.revision1.provenance import stamp, write_json
    p = PROJECT_ROOT / "results" / "revision9" / "train_summary.json"
    write_json(p, {"provenance": stamp("revision9/train_summary", {}), "runs": out})
    for r, v in out.items():
        print(f"{r:9s} {str(v['cpu'])[:30]:30s} feas10 {v['feasibility_rate_last10_mean']:.2f} "
              f"bestf {v['best_f_last_gen']:.3f} wall {v['wall_s_total'] / 60:.0f}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
