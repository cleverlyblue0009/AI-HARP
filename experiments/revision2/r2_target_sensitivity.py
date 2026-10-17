"""R2: sensitivity of Algorithm 1 to its target, miss tolerance and interpolation.

::

    python -m experiments.revision2.r2_target_sensitivity

Grid: q* = {0.90, 0.95, 0.98} x cell ceiling; miss tolerance = {0.02, 0.05, 0.10}
(actionable miss <= cell best + tolerance); interpolation on (Algorithm 1: linear
crossing between the cheapest qualifying point and the next-cheaper short point) or off
(only realised operating points: the cost of the cheapest qualifying point). 4 headline
cells; agent = full model, tau = 0.5, sampled, training seeds 0-2 (more when R6 lands).

Flags, against the paper's configuration (0.95 / 0.05 / interpolated): a change in a
cell's oracle scheme, in the best fixed setting, in which cells each agent seed matches,
or in the sign of the agent's regret / margin at rural d=20.
"""

from __future__ import annotations

import itertools

import numpy as np

from analysis.comparator import Cell, best_fixed_baseline, compare_policy, per_cell_oracle_best
from experiments.revision1.provenance import out_dir, stamp, write_json
from experiments.revision2.common import available_seeds, baseline_rows, cells_for, fin

TARGETS = (0.90, 0.95, 0.98)
MARGINS = (0.02, 0.05, 0.10)
BASE = (0.95, 0.05, True)
R20 = "rural_highway/d=20/clear/fog_bank"


class NoInterpCell(Cell):
    """Algorithm 1 restricted to realised points: no interpolated crossing."""

    def cost_at_matched(self, policy, target_quality, axis="cost", miss_bound=float("inf")):
        curve = self.curves.get(policy)
        if curve is None:
            return float("inf")
        pts = [p for p in curve.points if p.quality >= target_quality
               and self.within_miss_bound(p, miss_bound)]
        if not pts:
            return float("inf")
        best = min(pts, key=lambda p: p.cost)
        return best.cost if axis == "cost" else self.point_cost(best, axis)


def wrap(cells, interp):
    return cells if interp else [NoInterpCell(key=c.key, curves=c.curves) for c in cells]


def main() -> int:
    base = baseline_rows("headline")
    seeds = available_seeds("headline")
    agent_cells = {s: cells_for("headline", s, base=base) for s in seeds}
    ref_cells = cells_for("headline", None, base=base)
    out = {"seeds": seeds, "configs": {}}
    for t, m, interp in itertools.product(TARGETS, MARGINS, (True, False)):
        rc = wrap(ref_cells, interp)
        oracle = per_cell_oracle_best(rc, t, miss_margin=m)
        fixed = best_fixed_baseline(rc, t, miss_margin=m)
        cfg = {"target": t, "miss_margin": m, "interp": interp, "fixed_best": fixed.label, "cells": {}}
        agents = {s: compare_policy(wrap(agent_cells[s], interp), "ai_harp", t, miss_margin=m) for s in seeds}
        for c in rc:
            k = str(c.key)
            cfg["cells"][k] = {
                "target_rwcr": c.resolve_target(t), "oracle_policy": oracle.label(c.key),
                "oracle_cost": fin(oracle.cost(c.key)), "fixed_cost": fin(fixed.cost(c.key)),
                "agent": {s: {"cost": fin(r.agent_cost[ck]), "regret": fin(r.regret_per_cell[ck]),
                              "margin": fin(r.margin_per_cell[ck])}
                          for s, r in agents.items() for ck in r.agent_cost if str(ck) == k}}
        cfg["agent_cells_matched"] = {s: 4 - r.cells_failed for s, r in agents.items()}
        out["configs"][f"q={t:.2f}|miss={m:.2f}|{'interp' if interp else 'nointerp'}"] = cfg
    # flips vs the paper's configuration
    bkey = f"q={BASE[0]:.2f}|miss={BASE[1]:.2f}|interp"
    b = out["configs"][bkey]
    flips = {}
    for key, cfg in out["configs"].items():
        f = []
        if cfg["fixed_best"] != b["fixed_best"]:
            f.append(f"best fixed {b['fixed_best']} -> {cfg['fixed_best']}")
        for k, e in cfg["cells"].items():
            be = b["cells"][k]
            short = k.replace("/clear/fog_bank", "")
            if e["oracle_policy"] != be["oracle_policy"]:
                f.append(f"{short}: oracle {be['oracle_policy']} -> {e['oracle_policy']}")
            for s in seeds:
                a, ba = e["agent"][s], be["agent"][s]
                if (a["cost"] is None) != (ba["cost"] is None):
                    f.append(f"{short}: seed {s} {'now matches' if a['cost'] is not None else 'no longer matches'}")
                if k == R20 and a["cost"] is not None and ba["cost"] is not None:
                    for q in ("regret", "margin"):
                        if np.sign(a[q]) != np.sign(ba[q]):
                            f.append(f"rural d=20: seed {s} {q} sign {ba[q]:+.3f} -> {a[q]:+.3f}")
        flips[key] = f
    out["flips_vs_paper_config"] = flips
    out["provenance"] = stamp("R2", {"targets": TARGETS, "miss_margins": MARGINS, "interp": [True, False],
                                     "paper_config": BASE}, seeds={"eval": list(range(10)), "train": seeds})
    write_json(out_dir("R2", 2) / "r2_target_sensitivity.json", out)
    _table(out)
    return 0


def _table(out) -> None:
    import csv

    path = out_dir("R2", 2) / "r2_target_sensitivity.csv"
    seeds = out["seeds"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["target", "miss_margin", "interp", "cell", "q_star_rwcr", "oracle_policy", "oracle_cost",
                    "fixed_best", "fixed_cost"] + [f"agent_s{s}_{q}" for s in seeds for q in ("cost", "regret", "margin")])
        for cfg in out["configs"].values():
            for k, e in cfg["cells"].items():
                w.writerow([cfg["target"], cfg["miss_margin"], cfg["interp"], k, e["target_rwcr"], e["oracle_policy"],
                            e["oracle_cost"], cfg["fixed_best"], e["fixed_cost"]]
                           + [e["agent"][s][q] for s in seeds for q in ("cost", "regret", "margin")])
    for key, cfg in out["configs"].items():
        print(f"{key:28s} fixed {cfg['fixed_best']:30s} matched {cfg['agent_cells_matched']}  "
              f"flips: {len(out['flips_vs_paper_config'][key])}")


if __name__ == "__main__":
    raise SystemExit(main())
