"""Phase 0.1: overhead at matched coverage -- the paper's headline result.

Why this replaces "RWCR vs density" as the primary claim
--------------------------------------------------------
RWCR saturates. At every density from 5 to 120 veh/km/lane on the rural
corridor, all nine baselines land between 0.92 and 1.00, because the network is
connected and flooding brute-forces coverage. A result that reads "AI-HARP
achieves 0.98 where flooding achieves 0.976" is not a contribution, and a
contribution that only exists in a 2-3 veh/km/lane window is not defensible
either.

The quantity that does not saturate is **cost at matched coverage**. Every
suppression scheme has a tunable knob that trades reachability against
transmissions; sweeping it traces an operating curve in

    (RWCR, transmissions per at-risk vehicle informed)

and the interesting question is not "who scores highest" but "who reaches a
given RWCR most cheaply". :func:`overhead_at_matched_rwcr` answers exactly
that, by interpolating along each policy's own curve. A policy that never
reaches the target at any setting scores ``inf``, which is the honest answer.

This framing also makes the comparison fair: it stops a policy from looking
good merely by transmitting more, and it is invariant to the saturation that
makes the raw metric useless.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

from agents.registry import BASELINE_POLICIES
from common.config import load_yaml
from common.logging_utils import get_logger
from experiments.run_sim import RunSpec, run_single

logger = get_logger("analysis.pareto")

#: The suppression knob for each policy, and the values to sweep.
#:
#: Each is the parameter that trades reachability against transmissions. For
#: flooding there is none -- it is a single point by construction, which is
#: exactly why it sits at the expensive end of every front.
POLICY_SWEEPS: dict[str, tuple[str, tuple[Any, ...]]] = {
    "flooding": ("", ()),
    "p_persistence_03": ("p", (0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0)),
    "slotted_1p": ("n_slots", (1, 2, 3, 5, 8, 12, 20)),
    "weighted_p": ("max_p", (0.1, 0.2, 0.35, 0.5, 0.7, 0.85, 1.0)),
    "counter_based": ("counter_threshold", (1, 2, 3, 4, 6, 10)),
    "greedy_farthest": ("fallback_n_slots", (1, 2, 4, 8, 16)),
    "dvcast": ("n_slots", (1, 2, 3, 5, 8, 12, 20)),
}

#: Policies whose curve is a single point (no suppression knob).
SINGLE_POINT = {"flooding"}

DEFAULT_COST = "tx_per_at_risk_informed"
DEFAULT_QUALITY = "rwcr"


@dataclass
class OperatingPoint:
    """One (policy, parameter) setting, aggregated over seeds."""

    policy: str
    param: str
    value: Any
    quality: float          # mean RWCR
    quality_std: float
    cost: float             # mean transmissions per at-risk vehicle informed
    cost_std: float
    n_seeds: int
    extras: dict[str, float] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return self.policy if not self.param else f"{self.policy}({self.param}={self.value})"


@dataclass
class PolicyCurve:
    """A policy's operating curve, sorted by increasing cost."""

    policy: str
    points: list[OperatingPoint]

    def sorted_by_cost(self) -> list[OperatingPoint]:
        return sorted(self.points, key=lambda p: p.cost)

    def overhead_at(self, target_quality: float) -> float:
        """Minimum cost at which this policy attains ``target_quality``.

        Interpolated linearly along the policy's own curve between the two
        bracketing operating points. Returns ``inf`` when no setting of the
        knob reaches the target -- which is a real and reportable outcome, not
        a missing value.
        """
        pts = [p for p in self.points if np.isfinite(p.cost) and np.isfinite(p.quality)]
        if not pts:
            return float("inf")
        reaching = [p for p in pts if p.quality >= target_quality]
        if not reaching:
            return float("inf")
        best = min(reaching, key=lambda p: p.cost)

        # Interpolate against the cheapest point that falls short, if one sits
        # below `best` in cost -- the true crossing lies between them.
        cheaper_short = [p for p in pts if p.quality < target_quality and p.cost < best.cost]
        if not cheaper_short:
            return best.cost
        lo = max(cheaper_short, key=lambda p: p.cost)
        if best.quality == lo.quality:
            return best.cost
        frac = (target_quality - lo.quality) / (best.quality - lo.quality)
        return float(lo.cost + frac * (best.cost - lo.cost))

    def best_quality(self) -> float:
        return max((p.quality for p in self.points), default=float("nan"))


def pareto_front(points: Sequence[OperatingPoint]) -> list[OperatingPoint]:
    """Points not dominated on (higher quality, lower cost).

    ``a`` dominates ``b`` when it is at least as good on both axes and strictly
    better on one.
    """
    usable = [p for p in points if np.isfinite(p.cost) and np.isfinite(p.quality)]
    front: list[OperatingPoint] = []
    for p in usable:
        if any(
            (q.quality >= p.quality and q.cost <= p.cost)
            and (q.quality > p.quality or q.cost < p.cost)
            for q in usable
        ):
            continue
        front.append(p)
    return sorted(front, key=lambda p: p.cost)


# ---------------------------------------------------------------------------
# Running the sweep
# ---------------------------------------------------------------------------
def _aggregate(rows: list[dict[str, Any]], key: str) -> tuple[float, float]:
    vals = np.array([r.get(key, np.nan) for r in rows], dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan"), float("nan")
    return float(vals.mean()), float(vals.std())


def sweep_policy(
    policy: str,
    seeds: Iterable[int],
    *,
    scenario: str,
    density: float,
    weather: str = "clear",
    hazard_type: str | None = "fog_bank",
    cost_metric: str = DEFAULT_COST,
    quality_metric: str = DEFAULT_QUALITY,
    cfgs: dict[str, Any] | None = None,
    extra_metrics: Sequence[str] = ("collisions_per_delivered", "airtime_ms",
                                    "dissemination_cbr", "tir_median_s",
                                    "tir_p95_s", "tir_uninformed_frac",
                                    "actionable_deadline_miss_rate",
                                    "risk_est_precision", "risk_peak_corr"),
) -> PolicyCurve:
    """Trace one policy's operating curve by sweeping its suppression knob."""
    cfgs = cfgs or {}
    phy_cfg = cfgs.get("phy") or load_yaml("phy.yaml")
    hz_cfg = cfgs.get("hazard") or load_yaml("hazard.yaml")
    exp_cfg = cfgs.get("experiment") or load_yaml("experiment.yaml")
    seeds = list(seeds)

    param, values = POLICY_SWEEPS.get(policy, ("", ()))
    settings: list[tuple[str, Any, dict[str, Any]]] = (
        [("", None, {})] if not values else [(param, v, {param: v}) for v in values]
    )

    points: list[OperatingPoint] = []
    for pname, value, params in settings:
        rows = []
        for seed in seeds:
            spec = RunSpec(
                scenario=scenario, density_veh_km_lane=density, weather=weather,
                policy=policy, policy_params=params, seed=seed, hazard_type=hazard_type,
            )
            m, _ = run_single(spec, phy_cfg=phy_cfg, hz_cfg=hz_cfg, exp_cfg=exp_cfg)
            rows.append(m)
        q_mu, q_sd = _aggregate(rows, quality_metric)
        c_mu, c_sd = _aggregate(rows, cost_metric)
        points.append(OperatingPoint(
            policy=policy, param=pname, value=value,
            quality=q_mu, quality_std=q_sd, cost=c_mu, cost_std=c_sd,
            n_seeds=len(seeds),
            extras={k: _aggregate(rows, k)[0] for k in extra_metrics},
        ))
    return PolicyCurve(policy=policy, points=points)


def sweep_all(
    seeds: Iterable[int],
    *,
    scenario: str,
    density: float,
    policies: Sequence[str] = tuple(POLICY_SWEEPS),
    **kw: Any,
) -> dict[str, PolicyCurve]:
    curves: dict[str, PolicyCurve] = {}
    for pol in policies:
        logger.info("Pareto sweep: %s on %s @ %g veh/km/lane", pol, scenario, density)
        curves[pol] = sweep_policy(pol, seeds, scenario=scenario, density=density, **kw)
    return curves


def build_cells(
    cell_specs: Sequence[dict[str, Any]],
    seeds: Iterable[int],
    *,
    policies: Sequence[str] = tuple(POLICY_SWEEPS),
    cfgs: dict[str, Any] | None = None,
):
    """Run the operating-curve sweep over several factorial cells.

    Each spec is ``{scenario, density, weather, hazard_type}``. Returns
    ``analysis.comparator.Cell`` objects, which is what the headline comparator
    consumes.
    """
    from analysis.comparator import Cell, CellKey

    seeds = list(seeds)
    cells = []
    for spec in cell_specs:
        key = CellKey(
            scenario=spec["scenario"], density=float(spec["density"]),
            weather=spec.get("weather", "clear"),
            hazard_type=spec.get("hazard_type", "fog_bank"),
        )
        logger.info("=== cell %s (%d seeds) ===", key, len(seeds))
        curves = sweep_all(
            seeds, scenario=key.scenario, density=key.density, policies=policies,
            weather=key.weather, hazard_type=key.hazard_type, cfgs=cfgs,
        )
        cells.append(Cell(key=key, curves=curves))
    return cells


def overhead_at_matched_rwcr(
    curves: dict[str, PolicyCurve], target: float = 0.95
) -> dict[str, float]:
    """**The headline number.** Cheapest cost at which each policy hits ``target``.

    ``inf`` means the policy never reaches the target RWCR at any setting of
    its suppression knob, which is exactly what should be reported for a scheme
    that cannot achieve the required coverage at all.
    """
    return {name: curve.overhead_at(target) for name, curve in curves.items()}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def format_curves(curves: dict[str, PolicyCurve], title: str, target: float = 0.95) -> str:
    """Console rendering of the operating curves, the front and the headline."""
    lines: list[str] = []
    head = f"{'policy':<18}{'knob':>22}{'RWCR':>16}{'tx / informed':>18}{'coll/del':>10}"
    lines += ["=" * len(head), f" OPERATING CURVES -- {title}", "=" * len(head), head,
              "-" * len(head)]

    all_points: list[OperatingPoint] = []
    for name in curves:
        curve = curves[name]
        all_points.extend(curve.points)
        for p in curve.sorted_by_cost():
            knob = "-" if not p.param else f"{p.param}={p.value}"
            coll = p.extras.get("collisions_per_delivered", float("nan"))
            lines.append(
                f"{p.policy:<18}{knob:>22}"
                f"{p.quality:>10.3f}+-{p.quality_std:<5.3f}"
                f"{p.cost:>12.2f}+-{p.cost_std:<5.2f}{coll:>10.2f}"
            )
        lines.append("")

    front = pareto_front(all_points)
    lines += ["=" * len(head), " PARETO FRONT (non-dominated: higher RWCR, lower cost)",
              "=" * len(head)]
    for p in front:
        lines.append(f"  {p.label:<40} RWCR={p.quality:.3f}  cost={p.cost:.2f}")

    lines += ["", "=" * len(head),
              f" HEADLINE: transmissions per at-risk vehicle informed, at RWCR >= {target}",
              "=" * len(head)]
    oh = overhead_at_matched_rwcr(curves, target)
    finite = {k: v for k, v in oh.items() if np.isfinite(v)}
    best = min(finite, key=finite.get) if finite else None
    for name, cost in sorted(oh.items(), key=lambda kv: (not np.isfinite(kv[1]), kv[1])):
        if not np.isfinite(cost):
            lines.append(f"  {name:<22} never reaches RWCR {target} "
                         f"(best = {curves[name].best_quality():.3f})")
        else:
            mark = "  <-- cheapest" if name == best else ""
            rel = f"  ({cost / finite[best]:.2f}x)" if best else ""
            lines.append(f"  {name:<22} {cost:8.2f}{rel}{mark}")
    lines.append("=" * len(head))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="AI-HARP overhead-at-matched-coverage analysis")
    ap.add_argument("--scenario", default="rural_highway",
                    choices=["rural_highway", "urban_grid", "urban_nlos"])
    ap.add_argument("--density", type=float, default=20.0)
    ap.add_argument("--weather", default="clear")
    ap.add_argument("--hazard", default="fog_bank")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--target", type=float, default=0.95)
    ap.add_argument("--policies", nargs="*", default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    if args.quiet:
        logging.getLogger("aiharp").setLevel(logging.WARNING)

    cfgs = {"phy": load_yaml("phy.yaml"), "hazard": load_yaml("hazard.yaml"),
            "experiment": load_yaml("experiment.yaml")}
    policies = args.policies or list(POLICY_SWEEPS)
    curves = sweep_all(
        range(args.seeds), scenario=args.scenario, density=args.density,
        policies=policies, weather=args.weather, hazard_type=args.hazard, cfgs=cfgs,
    )
    title = (f"{args.scenario} | {args.density:g} veh/km/lane | {args.weather} "
             f"| {args.seeds} seeds")
    print(format_curves(curves, title, args.target))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
