"""R13: claims ledger -- does each headline claim survive the stress tests?

::

    python -m experiments.revision3.r13_claims_ledger

Columns: survives tolerance (R2: all 18 q* x tolerance x interpolation configurations,
results/revision2/R2), survives calibrated channel (R9, results/revision3/R9/r9_analysis.json),
survives SUMO (results/pareto_cells_sumo.json -- the 4 headline cells' baselines on SUMO -- and
the rural d=20 backend comparison in results/backend_validation_rural_d20.txt; the agent was never
evaluated on SUMO, so agent claims read "not tested"), n training seeds, verdict:
``robust`` only if it survives every applicable test and rests on >= 3 training seeds where it
involves the agent; otherwise ``exploratory`` (``refuted`` if it fails in the paper's own setting).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from analysis.comparator import best_fixed_baseline, load_cells, per_cell_oracle_best
from common.config import PROJECT_ROOT
from experiments.revision1.provenance import out_dir, stamp, write_json

R = PROJECT_ROOT / "results"


def sumo_facts():
    cells = load_cells(R / "pareto_cells_sumo.json")
    oracle = per_cell_oracle_best(cells)
    fixed = best_fixed_baseline(cells)
    out = {"fixed_best": fixed.label, "fixed_fails": sum(1 for v in fixed.per_cell_cost.values() if v == float("inf")),
           "oracle": {str(c.key).split("/clear")[0]: oracle.label(c.key) for c in cells},
           "penalty": {str(c.key).split("/clear")[0]: (fixed.cost(c.key) / oracle.cost(c.key))
                       if oracle.cost(c.key) not in (0, float("inf")) else None for c in cells}}
    r20 = next(c for c in cells if c.key.scenario == "rural_highway" and c.key.density == 20)
    fam = {p: max(pt.quality for pt in cu.points) for p, cu in r20.curves.items()}
    out["rural_d20_family_spread_excl_p03"] = max(v for p, v in fam.items() if p != "p_persistence_03") - \
        min(v for p, v in fam.items() if p != "p_persistence_03")
    return out


def main() -> int:
    r2 = json.loads((R / "revision2" / "R2" / "r2_target_sensitivity.json").read_text())
    r9 = json.loads((R / "revision3" / "R9" / "r9_analysis.json").read_text())
    cs = json.loads((R / "revision1" / "C1C2" / "c_seeds.json").read_text())
    sumo = sumo_facts()
    cfgs = r2["configs"]
    R20 = "rural_highway/d=20/clear/fog_bank"

    def r2_all(fn):
        ok = [fn(c) for c in cfgs.values()]
        return f"{sum(ok)}/{len(ok)} configs"

    seeds_ref = len(cs["by_variant"]["ref|tau=0.5|unbounded"]["seeds"])
    oracle_distinct = lambda c: len({e["oracle_policy"] for e in c["cells"].values()}) >= 3  # noqa: E731
    pen = lambda c: max((e["fixed_cost"] or 1e9) / e["oracle_cost"] for e in c["cells"].values() if e["oracle_cost"]) <= 1.13  # noqa: E731
    a20 = lambda c, s: c["cells"][R20]["agent"][str(s)]  # noqa: E731
    cal, dfl = r9["calibrated"], r9["default"]
    rows = [
        {"claim": "RWCR saturates at rural d=20: the non-default baselines' best RWCR lie within 0.01",
         "survives_tolerance": "n/a (not a matched-coverage claim)",
         "survives_calibrated_channel": f"yes (spread {cal['cells'][R20]['family_spread_excl_p_persistence']:.4f}; cell unaffected); "
                                        f"but saturation BREAKS in rural d=80 ({cal['cells']['rural_highway/d=80/clear/fog_bank']['family_spread_excl_p_persistence']:.3f}) "
                                        f"and urban d=20 ({cal['cells']['urban_nlos/d=20/clear/fog_bank']['family_spread_excl_p_persistence']:.3f})",
         "survives_sumo": (f"{'yes' if sumo['rural_d20_family_spread_excl_p03'] < 0.01 else 'no'} at rural d=20 "
                           f"(spread {sumo['rural_d20_family_spread_excl_p03']:.4f} vs the 0.01 threshold)"),
         "n_training_seeds": "n/a (baselines)",
         "verdict": ("robust at rural d=20 only" if sumo["rural_d20_family_spread_excl_p03"] < 0.01
                     else "exploratory: holds at rural d=20 in the fallback simulator (spread 0.0025) and under R9, "
                          "but not on SUMO (0.013) or in the dense calibrated cells")},
        {"claim": "The per-cell best baseline differs from cell to cell",
         "survives_tolerance": r2_all(oracle_distinct),
         "survives_calibrated_channel": "yes (" + ", ".join(e["oracle_policy"] for e in cal["cells"].values()) + ")",
         "survives_sumo": "yes (" + ", ".join(sumo["oracle"].values()) + ")",
         "n_training_seeds": "n/a", "verdict": "robust"},
        {"claim": "One fixed setting (dvcast(n_slots=30)) is within 13% of per-cell tuning in every headline cell",
         "survives_tolerance": r2_all(pen),
         "survives_calibrated_channel": f"no (best fixed becomes {cal['fixed_best']}; rural d=80 fixed {cal['cells']['rural_highway/d=80/clear/fog_bank']['fixed_cost']:.2f} vs oracle {cal['cells']['rural_highway/d=80/clear/fog_bank']['oracle_cost']:.2f})",
         "survives_sumo": f"no (best fixed {sumo['fixed_best']}; fails q* in {sumo['fixed_fails']} cell(s))",
         "n_training_seeds": "n/a", "verdict": "exploratory (backend- and setting-dependent; R3: best-fixed identity unstable)"},
        {"claim": "Agent matches coverage at rural d=20 with regret +1.8% / margin +1.9% (seed 0)",
         "survives_tolerance": f"regret sign kept in {sum(1 for c in cfgs.values() if a20(c,0)['regret'] is not None and a20(c,0)['regret'] > 0)}/{len(cfgs)}; margin flips without interpolation (+1.9% -> -0.8%)",
         "survives_calibrated_channel": "yes (cell unaffected; margin +25.3% vs the new fixed best)",
         "survives_sumo": "not tested (agent not run on SUMO)",
         "n_training_seeds": f"1 (seeds 0-4 give {', '.join(f'{100*a20(cfgs[list(cfgs)[8]],s)['regret']:+.1f}%' for s in range(5))})",
         "verdict": "exploratory (seed-specific; 5-seed regret CI [-19%, +10%], R3)"},
        {"claim": "Agent qualifies at rural d=20 on every training seed",
         "survives_tolerance": r2_all(lambda c: all(a20(c, s)["cost"] is not None for s in range(5))),
         "survives_calibrated_channel": "yes (all 5 seeds; cell unaffected)", "survives_sumo": "not tested",
         "n_training_seeds": str(seeds_ref), "verdict": "robust in the fallback simulator at q* <= 0.95 (fails at q* 0.98 for seed 0)"},
        {"claim": "Agent never reaches the sparse cell (rural d=2)",
         "survives_tolerance": r2_all(lambda c: all(v["cost"] is None for v in c["cells"]["rural_highway/d=2/clear/fog_bank"]["agent"].values())),
         "survives_calibrated_channel": "yes", "survives_sumo": "not tested (SUMO's rural d=2 ceiling is 0.221)",
         "n_training_seeds": str(seeds_ref), "verdict": "robust"},
        {"claim": "Agent reaches 3 of 4 cells on most seeds (seeds 2-4)",
         "survives_tolerance": r2_all(lambda c: sum(1 for k in c["cells"] if c["cells"][k]["agent"]["2"]["cost"] is not None) >= 3),
         "survives_calibrated_channel": "no (1/1/1/1/1: seeds 2-4 lose rural d=80 and urban d=20)",
         "survives_sumo": "not tested", "n_training_seeds": str(seeds_ref), "verdict": "exploratory"},
        {"claim": "Agent is faster (lower median TIR) than the reference baselines",
         "survives_tolerance": "n/a (latency not part of the target)",
         "survives_calibrated_channel": "not re-tested beyond rural d=20 (identical there)",
         "survives_sumo": "not tested", "n_training_seeds": "1 (B4, seed 0; significant in 3 of 4 cells)",
         "verdict": "exploratory"},
        {"claim": "The learned confidence gate adds value",
         "survives_tolerance": "n/a", "survives_calibrated_channel": "not tested",
         "survives_sumo": "not tested", "n_training_seeds": "5 (R12)",
         "verdict": "refuted as stated: a rate-matched random gate qualifies 3/4 cells on all seeds vs 2.2 +/- 1.1; the learned gate is cheaper only where all qualify"},
        {"claim": "GATv2 attention is not what earns the result (GCN / MLP / star ablations)",
         "survives_tolerance": "not tested", "survives_calibrated_channel": "not tested", "survives_sumo": "not tested",
         "n_training_seeds": "2 per ablation, 5 for the full model",
         "verdict": "exploratory (MLP clearly worse on the common cell; GCN/star within seed spread)"},
        {"claim": "The causal relevance feature is needed (R7 removal ablation)",
         "survives_tolerance": "not tested", "survives_calibrated_channel": "not tested", "survives_sumo": "not tested",
         "n_training_seeds": "2", "verdict": "not supported (removal stays within the full model's spread)"},
        {"claim": "The simulator is optimistic in dense traffic (validation gap)",
         "survives_tolerance": "n/a", "survives_calibrated_channel": "this IS the calibration (R9)",
         "survives_sumo": "consistent (SUMO lowers sparse and urban ceilings; README backend comparison)",
         "n_training_seeds": "n/a", "verdict": "robust"},
    ]
    d = out_dir("R13", 3)
    with (d / "r13_claims_ledger.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    write_json(d / "r13_claims_ledger.json", {"rows": rows, "sumo_facts": sumo, "provenance": stamp(
        "R13", {"sources": ["results/revision2/R2/r2_target_sensitivity.json", "results/revision3/R9/r9_analysis.json",
                            "results/pareto_cells_sumo.json", "results/revision1/C1C2/c_seeds.json"]})})
    for r in rows:
        print(f"- {r['claim'][:70]:70s} | tol: {r['survives_tolerance'][:22]:22s} | R9: {r['survives_calibrated_channel'][:28]:28s} | SUMO: {r['survives_sumo'][:22]:22s} | {r['verdict'][:40]}")
    print("SUMO facts:", sumo)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
