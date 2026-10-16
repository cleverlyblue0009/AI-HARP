"""B2: does cost keep falling as slot-based schemes wait longer?

::

    python -m experiments.revision1.b2_slots

Joins the committed curves (n_slots <= 50, fallback_n_slots <= 32, from
results/pareto_cells.json) with the extended grid (75-200 / 64-128, from
results/revision1/B_sweeps/slots_runs.csv) in the four headline cells, and
marks for each point whether it meets Algorithm 1's target and deadline guard
(q* and miss bound of the committed, unextended cell). Writes
results/revision1/B2/b2_slots.{csv,json} and fig_b2_slots.{pdf,png}.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from analysis.comparator import load_cells
from common.config import PROJECT_ROOT
from experiments.revision1.provenance import out_dir, stamp, write_json
from experiments.revision1.runner import cells_from_rows, load_rows

R = PROJECT_ROOT / "results"
POLS = {"slotted_1p": "n_slots", "dvcast": "n_slots", "greedy_farthest": "fallback_n_slots"}


def main() -> int:
    committed = {(c.key.scenario, c.key.density): c for c in load_cells(R / "pareto_cells.json")}
    ext = {(c.key.scenario, c.key.density): c
           for c in cells_from_rows(load_rows(R / "revision1" / "B_sweeps" / "slots_runs.csv"))}
    rows = []
    for k, c in committed.items():
        tq = c.resolve_target(0.95, "relative", ("ai_harp",))
        mb = c.resolve_miss_bound(0.05, ("ai_harp",))
        for pol in POLS:
            pts = list(c.curves[pol].points) + (list(ext[k].curves[pol].points) if k in ext and pol in ext[k].curves else [])
            for p in pts:
                miss = p.extras.get("actionable_deadline_miss_rate", np.nan)
                rows.append({"scenario": k[0], "density": k[1], "policy": pol, "slots": p.value,
                             "extended": bool(float(p.value) > (32 if pol == "greedy_farthest" else 50)),
                             "rwcr": p.quality, "rwcr_std": p.quality_std, "cost": p.cost,
                             "tir_median_s": p.extras.get("tir_median_s"),
                             "tir_p95_s": p.extras.get("tir_p95_s"), "miss": miss,
                             "target": tq, "miss_bound": mb, "meets_target": p.quality >= tq,
                             "meets_guard": bool(miss <= mb), "qualifies": bool(p.quality >= tq and miss <= mb)})
    df = pd.DataFrame(rows).sort_values(["scenario", "density", "policy", "slots"])
    d = out_dir("B2")
    df.to_csv(d / "b2_slots.csv", index=False)
    summary = {}
    for (sc, den, pol), g in df.groupby(["scenario", "density", "policy"]):
        g = g.sort_values("slots")
        q_all = g[g.meets_target]
        q_guard = g[g.qualifies]
        summary[f"{sc}|{den:g}|{pol}"] = {
            "cheapest_meeting_target": None if q_all.empty else q_all.loc[q_all.cost.idxmin(), ["slots", "cost", "tir_median_s", "miss"]].to_dict(),
            "cheapest_qualifying_with_guard": None if q_guard.empty else q_guard.loc[q_guard.cost.idxmin(), ["slots", "cost", "tir_median_s", "miss"]].to_dict(),
            "cost_at_50_or_32": g[g.slots == (32 if pol == "greedy_farthest" else 50)].cost.tolist(),
            "cost_at_max": g.cost.iloc[-1], "max_slots": g.slots.iloc[-1],
            "tir_at_max": g.tir_median_s.iloc[-1]}
    write_json(d / "b2_slots.json", {"summary": summary, "provenance": stamp(
        "B2", {"committed": "results/pareto_cells.json",
               "extended": "results/revision1/B_sweeps/slots_runs.csv"}, seeds=list(range(10)))})
    _figure(df, d)
    print(df[["scenario", "density", "policy", "slots", "rwcr", "cost", "tir_median_s", "miss",
              "qualifies"]].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    return 0


def _figure(df: pd.DataFrame, d) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from analysis import figures as F

    F.apply_style() if hasattr(F, "apply_style") else None
    import matplotlib.pyplot as plt

    cells = list(df.groupby(["scenario", "density"]).groups)
    fig, axes = plt.subplots(2, len(cells), figsize=(7.16, 3.6), sharex=False)
    marks = {"slotted_1p": ("o", "-"), "dvcast": ("s", "--"), "greedy_farthest": ("^", ":")}
    colors = {"slotted_1p": "#0072B2", "dvcast": "#D55E00", "greedy_farthest": "#009E73"}
    for j, (sc, den) in enumerate(cells):
        g = df[(df.scenario == sc) & (df.density == den)]
        for pol, gg in g.groupby("policy"):
            gg = gg.sort_values("slots")
            m, ls = marks[pol]
            for i, y in enumerate(("cost", "tir_median_s")):
                ax = axes[i, j]
                ax.plot(gg.slots, gg[y], marker=m, linestyle=ls, color=colors[pol], ms=3, lw=0.9,
                        label=pol if (i == 0 and j == 0) else None)
                q = gg[gg.qualifies]
                ax.plot(q.slots, q[y], linestyle="none", marker=m, ms=5, mfc="none", mec="k", mew=0.6)
        axes[0, j].set_title(f"{sc.replace('_', ' ')} d={den:g}", fontsize=7)
        for i in range(2):
            axes[i, j].set_xscale("log")
            axes[i, j].axvline(50, color="0.6", lw=0.6, ls="--")
            axes[i, j].tick_params(labelsize=6)
        axes[1, j].set_xlabel("slots (100 ms each)", fontsize=7)
    axes[0, 0].set_ylabel("tx / at-risk informed", fontsize=7)
    axes[1, 0].set_ylabel("median TIR (s)", fontsize=7)
    fig.legend(loc="upper center", ncol=3, fontsize=6, frameon=False)
    fig.text(0.5, 0.005, "black ring = qualifies (RWCR >= q*, miss <= best + 0.05); dashed line = committed grid edge (50)",
             ha="center", fontsize=6)
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    for ext in ("pdf", "png"):
        fig.savefig(d / f"fig_b2_slots.{ext}", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
