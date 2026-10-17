"""B1: latency-bounded matched coverage -- Algorithm 1 plus median TIR <= D.

::

    python -m experiments.revision1.b1_latency                       # 4 headline cells
    python -m experiments.revision1.b1_latency --rows results/revision1/B1/grid24_runs.csv \
        --agent-rows results/revision1/B1/grid24_agent_runs.csv --tag grid24

A setting qualifies at bound D only if, in addition to Algorithm 1's rules
(RWCR >= q*, actionable miss <= cell best + 0.05), its seed-mean of per-run
median TIR is <= D. The bound enters the comparator's eligibility test, so the
interpolation's lower anchor must satisfy it too (no crossing is interpolated
toward a setting that breaks the deadline).

Two target variants:

* ``fixed``   (primary) q* and the miss bound stay those of the unbounded
  comparison: the coverage requirement does not change, latency is an extra
  constraint.
* ``rebased`` q* = 0.95 x the best RWCR among settings meeting the bound, and
  the miss bound is rebased the same way.

Derived deadline (per scenario): a vehicle one radio hop from a holder,
approaching the hazard at the scenario's maximum desired car speed v, can still
react and stop if the next hop reaches it within

    D_r = (R - v * t_r - v^2 / (2 a)) / v

with t_r = 2.5 s (configs/hazard.yaml deadlines.reaction_time_s, AASHTO),
a = 3.4 m/s^2 (AASHTO comfortable deceleration; ASSUMED), R the nominal
median-link range (rural 610.4 m, urban LOS 479.6 m) or, for urban_nlos, the
blocked-link (90-degree corner) range, since a grid hop round a corner is the
binding case there.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from analysis.comparator import (Cell, best_fixed_baseline, compare_policy, load_cells,
                                 per_cell_oracle_best)
from common.config import PROJECT_ROOT, load_yaml
from experiments.revision1.provenance import out_dir, rel, stamp, write_json

R = PROJECT_ROOT / "results"
TIR = "tir_median_s"
FIXED_D = (0.5, 1.0, 1.5, 2.0)
REACTION_S = 2.5
DECEL = 3.4


@dataclass
class LatencyCell(Cell):
    D: float = float("inf")
    rebased: bool = False

    def _fast(self, p) -> bool:
        if not np.isfinite(self.D):
            return True
        t = float(p.extras.get(TIR, np.nan))
        return bool(np.isfinite(t) and t <= self.D + 1e-9)

    def within_miss_bound(self, point, bound) -> bool:  # type: ignore[override]
        return Cell.within_miss_bound(point, bound) and self._fast(point)

    def achievable_quality(self, exclude=()) -> float:
        if not self.rebased:
            return super().achievable_quality(exclude)
        ex = set(exclude)
        q = [p.quality for n, c in self.curves.items() if n not in ex for p in c.points
             if np.isfinite(p.quality) and self._fast(p)]
        return float(max(q)) if q else float("nan")

    def best_miss_rate(self, exclude=()) -> float:
        if not self.rebased:
            return super().best_miss_rate(exclude)
        ex = set(exclude)
        v = [float(p.extras.get("actionable_deadline_miss_rate", np.nan))
             for n, c in self.curves.items() if n not in ex for p in c.points if self._fast(p)]
        v = [x for x in v if np.isfinite(x)]
        return float(min(v)) if v else float("nan")


def derived_deadlines() -> dict[str, dict[str, float]]:
    """D_r per scenario, with every input recorded."""
    from mobility.generate import get_trace, load_scenario
    from sim.phy import build_phy

    phy_cfg = load_yaml("phy.yaml")
    out = {}
    for sc in ("rural_highway", "urban_grid", "urban_nlos"):
        scn = load_scenario(sc)
        vmax = float(scn["vehicles"]["classes"]["car"]["speed_kmh_max"]) / 3.6
        trace = get_trace(scn, 20.0, 0, weather="clear", phy_cfg=phy_cfg)
        phy = build_phy(phy_cfg, scenario=sc, weather="clear", seed=0, trace_meta=trace.meta)
        s = phy.summary()
        rng = float(s["nlos_range_m"]) if sc == "urban_nlos" and s.get("nlos_range_m") else \
            float(s["nominal_range_m"])
        stop = vmax * REACTION_S + vmax ** 2 / (2 * DECEL)
        out[sc] = {"v_mps": vmax, "range_m": rng, "range_kind":
                   "nlos_corner" if sc == "urban_nlos" else "nominal_los",
                   "nominal_range_m": float(s["nominal_range_m"]),
                   "stopping_distance_m": stop, "t_r_s": REACTION_S, "a_mps2": DECEL,
                   "D_s": max(0.0, (rng - stop) / vmax)}
    return out


def bounded(cells: Sequence[Cell], D: dict[str, float] | float, rebased: bool) -> list[LatencyCell]:
    get = (lambda sc: D[sc]) if isinstance(D, dict) else (lambda sc: D)
    return [LatencyCell(key=c.key, curves=c.curves, D=float(get(c.key.scenario)), rebased=rebased)
            for c in cells]


def best_setting(cell: Cell, policy: str, tq: float, mb: float):
    """The cheapest qualifying setting of ``policy`` (the knob value k*)."""
    pts = [p for p in cell.curves.get(policy, []).points
           if p.quality >= tq and cell.within_miss_bound(p, mb)] if policy in cell.curves else []
    return min(pts, key=lambda p: p.cost) if pts else None


def _fin(x: float) -> float | None:
    return float(x) if x is not None and np.isfinite(x) else None


def evaluate(cells: Sequence[Cell], agent: str | None) -> dict[str, Any]:
    excl = (agent,) if agent else ()
    oracle = per_cell_oracle_best(cells, 0.95, "cost", exclude=excl)
    fixed = best_fixed_baseline(cells, 0.95, "cost", exclude=excl)
    res = compare_policy(cells, agent, 0.95, "cost") if agent else None
    out: dict[str, Any] = {"fixed_best": fixed.label,
                           "fixed_best_cells_failed": sum(1 for v in fixed.per_cell_cost.values()
                                                          if not np.isfinite(v)),
                           "fixed_best_score": fixed.score, "cells": {}}
    for c in cells:
        tq = c.resolve_target(0.95, "relative", excl)
        mb = oracle.miss_bound[c.key]
        schemes = {}
        for pol in c.curves:
            if pol in excl:
                continue
            bs = best_setting(c, pol, tq, mb)
            schemes[pol] = {"k_star": None if bs is None else bs.value,
                            "matched_cost": _fin(c.cost_at_matched(pol, tq, "cost", mb)),
                            "k_star_rwcr": None if bs is None else bs.quality,
                            "k_star_tir_median_s": None if bs is None else bs.extras.get(TIR),
                            "max_rwcr_within_bound": _fin(max(
                                [p.quality for p in c.curves[pol].points if c._fast(p)]
                                or [float("nan")])) if isinstance(c, LatencyCell) else None}
        e = {"D_s": getattr(c, "D", float("inf")), "ceiling": c.achievable_quality(excl),
             "target": tq, "miss_bound": mb, "oracle_policy": oracle.label(c.key),
             "oracle_cost": _fin(oracle.cost(c.key)), "fixed_cost": _fin(fixed.cost(c.key)),
             "schemes": schemes}
        if res is not None:
            a = res.agent_cost[c.key]
            bs = best_setting(c, agent, tq, mb)
            e.update(agent_cost=_fin(a), agent_regret=_fin(res.regret_per_cell[c.key]),
                     agent_margin=_fin(res.margin_per_cell[c.key]),
                     agent_k_star_bias=None if bs is None else bs.value,
                     agent_reaches=bool(np.isfinite(a)))
        out["cells"][str(c.key)] = e
    if res is not None:
        out.update(agent_mean_regret=_fin(res.mean_regret), agent_mean_margin=_fin(res.mean_margin),
                   agent_cells_failed=res.cells_failed)
    return out


def markdown(results: dict[str, Any], derived: dict[str, Any]) -> str:
    L = ["# B1 -- latency-bounded matched coverage", ""]
    L.append("Derived deadline D_r = (R - v t_r - v^2/2a)/v; t_r = 2.5 s, a = 3.4 m/s^2:")
    for sc, d in derived.items():
        L.append(f"* {sc}: v = {d['v_mps']:.2f} m/s, R = {d['range_m']:.1f} m ({d['range_kind']}), "
                 f"stopping distance {d['stopping_distance_m']:.1f} m -> **D_r = {d['D_s']:.2f} s**")
    for variant, per_d in results.items():
        L += ["", f"## Target variant: {variant}", ""]
        L.append("| D | cell | target | oracle (cost) | best fixed: cost | agent cost | regret | margin |")
        L.append("|---|---|---|---|---|---|---|---|")
        for dname, r in per_d.items():
            for ck, e in r["cells"].items():
                f = lambda v, pct=False: "n/a" if v is None else (f"{100 * v:+.1f}%" if pct else f"{v:.3f}")  # noqa: E731
                L.append(f"| {dname} | {ck} | {e['target']:.4f} | {e['oracle_policy']} ({f(e['oracle_cost'])}) "
                         f"| {r['fixed_best']}: {f(e['fixed_cost'])} | {f(e.get('agent_cost'))} "
                         f"| {f(e.get('agent_regret'), True)} | {f(e.get('agent_margin'), True)} |")
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cells", default=str(R / "agent_ref" / "sampled" / "pareto_cells_agent.json"),
                    help="comparator cells file with baseline (and optionally ai_harp) curves")
    ap.add_argument("--rows", default=None, help="per-seed baseline rows (runner CSV) instead")
    ap.add_argument("--agent-rows", nargs="*", default=[], help="per-seed agent rows (runner CSV)")
    ap.add_argument("--agent", default="ai_harp")
    ap.add_argument("--tag", default="headline")
    ap.add_argument("--extra-rows", nargs="*", default=[],
                    help="more per-seed baseline rows (e.g. gen_runs.csv for urban_grid cells)")
    ap.add_argument("--weather", default="clear")
    ap.add_argument("--hazard", default="fog_bank")
    ap.add_argument("--tau", type=float, default=0.5, help="agent gate arm to keep")
    args = ap.parse_args(argv)

    if args.rows:
        import json as _json

        from experiments.revision1.runner import cells_from_rows, load_rows

        def keep(r):
            return r["weather"] == args.weather and r["hazard_type"] == args.hazard

        rows = [r for p in [args.rows] + args.extra_rows for r in load_rows(Path(p))
                if keep(r) and (r["param"] or r["policy"] == "flooding")]
        seen = set()
        for p in args.agent_rows:
            for r in load_rows(Path(p)):
                q = _json.loads(r["params_json"])
                if (keep(r) and float(q.get("tau", -1)) == args.tau and not q.get("deterministic")
                        and "random_gate_rate" not in q and r["task_hash"] not in seen):
                    seen.add(r["task_hash"])
                    rows.append(r)
        args.agent_rows = list(args.agent_rows) + list(args.extra_rows)
        cells = cells_from_rows(rows)
        src = [rel(args.rows)] + [rel(p) for p in args.agent_rows]
    else:
        cells = load_cells(Path(args.cells))
        src = [rel(args.cells)]
    agent = args.agent if any(args.agent in c.curves for c in cells) else None
    derived = derived_deadlines()
    bounds: dict[str, Any] = {"unbounded": float("inf")}
    bounds.update({f"D={d:g}s": d for d in FIXED_D})
    bounds["D=derived"] = {sc: v["D_s"] for sc, v in derived.items()}

    results: dict[str, dict[str, Any]] = {}
    for variant in ("fixed", "rebased"):
        results[variant] = {}
        for name, D in bounds.items():
            # fixed: q* and miss bound from the UNBOUNDED cell, latency only
            # filters; rebased: both recomputed among settings meeting D.
            cs = bounded(cells, D, rebased=(variant == "rebased"))
            results[variant][name] = evaluate(cs, agent)
    d = out_dir("B1")
    payload = {"results": results, "derived_deadlines": derived, "inputs": src,
               "provenance": stamp("B1", {"bounds": {k: v for k, v in bounds.items()},
                                          "target": 0.95, "miss_margin": 0.05, "tag": args.tag,
                                          "inputs": src})}
    write_json(d / f"b1_{args.tag}.json", payload)
    (d / f"b1_{args.tag}.md").write_text(markdown(results, derived), encoding="utf-8")
    print(markdown({"fixed": results["fixed"]}, derived))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
