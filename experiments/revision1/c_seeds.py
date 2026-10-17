"""C1/C2: training-seed variance of the full model and the ablations.

::

    python -m experiments.revision1.c_seeds

Per variant (ref, gcn, mlp, star) and training seed (0 = checkpoints/campaign/<v>,
1-2 = checkpoints/revision1/<v>_s<seed>), from the per-seed evaluation rows
results/revision1/agent_runs/<v>_s<seed>.csv (seed 0 of ref: ref_s0.csv), each
final checkpoint scored on the 4 headline cells x 10 paired eval seeds, tau = 0
and 0.5, sampled. Comparison as Algorithm 1, unbounded and under the B1 median-TIR
bounds. Reported as mean +/- SD across training seeds:

* cells matched (of 4);
* regret / margin averaged over the SAME cells for every variant: the cells that
  every (variant, seed) matched -- and, separately, per cell where all matched;
* RWCR, cost, median TIR at the agent's default setting (bias 0) averaged over
  all 4 cells (defined for everyone, so directly comparable);
* matched cost per cell.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from common.config import PROJECT_ROOT
from experiments.revision1.b1_latency import FIXED_D, bounded, derived_deadlines, evaluate
from experiments.revision1.provenance import out_dir, stamp, write_json
from experiments.revision1.runner import cells_from_rows, load_rows

R = PROJECT_ROOT / "results"
AR = R / "revision1" / "agent_runs"
VARIANTS = ("ref", "gcn", "mlp", "star")


def rows_for(variant: str, seed: int) -> Path | None:
    p = AR / (f"{variant}_s{seed}.csv")
    return p if p.exists() else None


def _ms(v):
    v = [x for x in v if x is not None and np.isfinite(x)]
    if not v:
        return {"mean": None, "sd": None, "n": 0}
    return {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)) if len(v) > 1 else None,
            "n": len(v), "values": [float(x) for x in v]}


def main() -> int:
    base = [r for r in load_rows(R / "revision1" / "B_sweeps" / "headline_runs.csv")
            if r["param"] or r["policy"] == "flooding"]
    der = {sc: v["D_s"] for sc, v in derived_deadlines().items()}
    bounds = {"unbounded": float("inf"), **{f"D={d:g}s": d for d in FIXED_D}, "D=derived": der}
    per: dict = defaultdict(dict)        # (variant, tau, bound) -> seed -> evaluate()
    default_pt: dict = defaultdict(dict)  # (variant, tau) -> seed -> {rwcr,cost,tir}
    for v in VARIANTS:
        for s in (0, 1, 2):
            p = rows_for(v, s)
            if p is None:
                continue
            rows = load_rows(p)
            for tau in (0.0, 0.5):
                ar = [r for r in rows if float(json.loads(r["params_json"]).get("tau", -1)) == tau
                      and not json.loads(r["params_json"]).get("deterministic")]
                if len({(r["scenario"], r["density"]) for r in ar}) < 4:
                    continue
                cells = cells_from_rows(base + ar)
                for bn, D in bounds.items():
                    per[(v, tau, bn)][s] = evaluate(bounded(cells, D, False), "ai_harp")
                b0 = [r for r in ar if float(r["value"]) == 0.0]
                f = lambda k: float(np.mean([float(r[k]) for r in b0]))  # noqa: E731
                default_pt[(v, tau)][s] = {"rwcr": f("rwcr"), "cost": f("tx_per_at_risk_informed"),
                                           "tir_median_s": f("tir_median_s"),
                                           "miss": f("actionable_deadline_miss_rate")}
    out: dict = {"by_variant": {}}
    for (v, tau, bn), seeds in sorted(per.items()):
        cellnames = list(next(iter(seeds.values()))["cells"])
        matched = {s: [c for c in cellnames if r["cells"][c]["agent_reaches"]] for s, r in seeds.items()}
        key = f"{v}|tau={tau:g}|{bn}"
        out["by_variant"][key] = {
            "seeds": sorted(seeds), "cells_matched": _ms([len(m) for m in matched.values()]),
            "matched_cells_by_seed": matched,
            "per_cell_cost": {c: _ms([seeds[s]["cells"][c]["agent_cost"] for s in seeds]) for c in cellnames},
            "per_cell_regret": {c: _ms([seeds[s]["cells"][c]["agent_regret"] for s in seeds]) for c in cellnames},
            "per_cell_margin": {c: _ms([seeds[s]["cells"][c]["agent_margin"] for s in seeds]) for c in cellnames},
            "fixed_best": next(iter(seeds.values()))["fixed_best"]}
    # common cells: matched by every (variant, seed) for this tau and bound
    for tau in (0.0, 0.5):
        for bn in bounds:
            keys = [k for k in out["by_variant"] if k.endswith(f"|tau={tau:g}|{bn}")]
            if not keys:
                continue
            common = set.intersection(*[set(c) for k in keys
                                        for c in out["by_variant"][k]["matched_cells_by_seed"].values()])
            for k in keys:
                v = k.split("|")[0]
                seeds = per[(v, tau, bn)]
                out["by_variant"][k]["common_cells"] = sorted(common)
                out["by_variant"][k]["regret_common"] = _ms(
                    [np.mean([seeds[s]["cells"][c]["agent_regret"] for c in common]) if common else None for s in seeds])
                out["by_variant"][k]["margin_common"] = _ms(
                    [np.mean([seeds[s]["cells"][c]["agent_margin"] for c in common]) if common else None for s in seeds])
    out["default_setting_all_4_cells"] = {
        f"{v}|tau={tau:g}": {m: _ms([d[m] for d in seeds.values()]) for m in ("rwcr", "cost", "tir_median_s", "miss")}
        for (v, tau), seeds in sorted(default_pt.items())}
    out["provenance"] = stamp("C1C2", {"rows": "results/revision1/agent_runs/<variant>_s<seed>.csv",
                                       "baseline_rows": "results/revision1/B_sweeps/headline_runs.csv",
                                       "bounds": list(bounds)}, seeds={"eval": list(range(10)),
                                                                       "train": [0, 1, 2]})
    write_json(out_dir("C1C2") / "c_seeds.json", out)
    _tex(out, out_dir("C1C2") / "table_ablation_seeds.tex")
    _print(out)
    return 0


def _tex(out, path) -> None:
    """Ablations over training seeds, same cells for every variant (tau = 0.5)."""
    names = {"ref": "full (GATv2)", "gcn": "GCN encoder", "mlp": "MLP encoder", "star": "star graph"}
    pm = lambda d, p=True: "--" if d["mean"] is None else (  # noqa: E731
        f"${100 * d['mean']:+.1f}" + (f" \\pm {100 * d['sd']:.1f}" if d["sd"] is not None else "") + r"\%$"
        if p else f"${d['mean']:.2f}" + (f" \\pm {d['sd']:.2f}" if d["sd"] is not None else "") + "$")
    lines = ["% Ablations over training seeds (revision-1 C2). tau=0.5, sampled, 4 headline cells,",
             "% 10 paired eval seeds per training seed; mean +/- SD across training seeds.",
             "% Regret/margin over the cells EVERY variant and seed matched (rural d=20).",
             "% Generated by experiments/revision1/c_seeds.py -- do not edit by hand.",
             r"\begin{tabular}{lrrrrr}", r"\toprule",
             r"Variant & seeds & Cells matched & Regret (common) & Margin (common) & Cost @ bias 0 \\",
             r"\midrule"]
    for v in VARIANTS:
        e = out["by_variant"].get(f"{v}|tau=0.5|unbounded")
        b = out["default_setting_all_4_cells"].get(f"{v}|tau=0.5")
        if not e or not b:
            continue
        lines.append(f"{names[v]} & {len(e['seeds'])} & {pm(e['cells_matched'], False)} & "
                     f"{pm(e['regret_common'])} & {pm(e['margin_common'])} & {pm(b['cost'], False)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _print(out):
    f = lambda d, p=False: "n/a" if d["mean"] is None else (  # noqa: E731
        (f"{100 * d['mean']:+.1f}%" if p else f"{d['mean']:.3f}") +
        ("" if d["sd"] is None else (f" ± {100 * d['sd']:.1f}" if p else f" ± {d['sd']:.3f}")) + f" (n={d['n']})")
    for k, e in out["by_variant"].items():
        if not k.endswith("unbounded") and not k.endswith("D=0.5s"):
            continue
        print(f"{k:32s} matched {f(e['cells_matched'])}  common {e.get('common_cells')}  "
              f"regret {f(e.get('regret_common', {'mean': None}), True)}  margin {f(e.get('margin_common', {'mean': None}), True)}")
    for k, e in out["default_setting_all_4_cells"].items():
        print(f"{k:16s} bias0 over 4 cells: RWCR {f(e['rwcr'])} cost {f(e['cost'])} TIR {f(e['tir_median_s'])}")


if __name__ == "__main__":
    raise SystemExit(main())
