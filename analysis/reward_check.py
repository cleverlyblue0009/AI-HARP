"""Does a training objective rank whole policies sanely? Check BEFORE training.

The weighted-sum reward was declared "calibrated" after balancing a single
transmission against the relevance it directly informs. It then ranked
near-total silence (+1.5) above slotted_1p (-829) and flooding (-9301), and the
agent learned silence. The per-transmission check was not wrong; it was the
wrong check. This module runs the right one: score fixed reference policies on
real episodes under the objective, and look at the ranking.

For the constrained objective the per-episode Lagrangian, per at-risk vehicle,
is ``-cost + lambda * C`` (the ``-lambda * C*`` term is common to every policy
and cannot change a ranking). Two things matter:

* the break-even ``lambda`` above which silence stops being the best policy --
  it must be finite and below ``lambda_max``, or dual ascent can never escape
  silence;
* at a ``lambda`` where silence is no longer best, a coverage-meeting efficient
  scheme must beat flooding -- otherwise the objective has swapped one
  degenerate optimum for the other.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from agents.constrained_reward import (
    ConstrainedObjective,
    causal_coverage,
    tree_credit,
    warned_at_risk,
)
from agents.registry import build_policy
from common.config import load_yaml
from common.logging_utils import get_logger
from common.seeding import SeedBundle
from hazard.model import hazard_from_config
from hazard.risk_field import build_risk_field
from mobility.generate import get_trace, load_scenario
from sim.engine import DisseminationEngine, SimSettings
from sim.mac import build_mac
from sim.phy import build_phy

logger = get_logger("analysis.reward_check")

#: (label, registered policy, params). always-suppress is p-persistence at p=0:
#: the originator still broadcasts, nobody relays.
REFERENCES: tuple[tuple[str, str, dict[str, Any]], ...] = (
    ("always-suppress", "p_persistence_03", {"p": 0.0}),
    ("slotted_1p", "slotted_1p", {}),
    ("dvcast", "dvcast", {}),
    ("weighted_p", "weighted_p", {}),
    ("flooding", "flooding", {}),
)
SILENCE = "always-suppress"


@dataclass
class PolicyScore:
    label: str
    coverage: float          # causal
    cost: float              # (tx + w_coll * coll) / n_at_risk
    credit_per_at_risk: float
    transmissions: int

    def lagrangian(self, lam: float) -> float:
        """Per at-risk vehicle; the -lambda*C* constant is omitted."""
        return -self.cost + lam * self.coverage


def score_references(
    scenario: str, density: float, weather: str, hazard_type: str, seed: int,
    objective: ConstrainedObjective, cfgs: dict[str, Any],
    duration_s: float | None = None, corridor_length_m: float | None = None,
    references: Sequence[tuple[str, str, dict[str, Any]]] = REFERENCES,
) -> list[PolicyScore]:
    scenario_cfg = load_scenario(scenario)
    if duration_s is not None:
        scenario_cfg["simulation"]["duration_s"] = float(duration_s)
    if corridor_length_m is not None:
        scenario_cfg["geometry"]["length_m"] = float(corridor_length_m)
    trace = get_trace(scenario_cfg, density, seed, weather=weather, phy_cfg=cfgs["phy"])
    hazard = hazard_from_config(cfgs["hazard"], trace.meta, overrides={"type": hazard_type})
    phy = build_phy(cfgs["phy"], scenario_cfg["name"], weather, seed, trace_meta=trace.meta)
    mac = build_mac(cfgs["phy"], phy)
    risk = build_risk_field(cfgs["hazard"], trace, hazard)
    rel = risk.relevance_matrix(trace, hazard)

    out = []
    for label, name, params in references:
        res = DisseminationEngine(trace, phy, mac, risk, hazard, build_policy(name, **params),
                                  SeedBundle(master_seed=seed),
                                  SimSettings.from_config(cfgs["experiment"])).run()
        peak, at_risk, warned = warned_at_risk(rel, res.informed_step, risk.at_risk_threshold)
        n_ar = max(int(at_risk.sum()), 1)
        coll = (np.asarray(res.collisions_caused, dtype=float)
                if res.collisions_caused is not None else np.zeros(trace.n_vehicles))
        cost = float((res.tx_count.sum() + objective.collision_weight * coll.sum()) / n_ar)
        credit = tree_credit(res.informed_by, peak, at_risk, warned)
        out.append(PolicyScore(label=label, coverage=causal_coverage(peak, at_risk, warned),
                               cost=cost, credit_per_at_risk=float(credit.sum() / n_ar),
                               transmissions=int(res.n_transmissions)))
    return out


def silence_break_even(scores: Sequence[PolicyScore]) -> float:
    """Smallest lambda at which some policy beats always-suppress.

    For each policy p with more coverage than silence, p overtakes silence at
    ``lambda_p = (cost_p - cost_s) / (C_p - C_s)``. The minimum over p is where
    dual ascent first escapes silence. ``inf`` if no policy covers more.
    """
    s = next(x for x in scores if x.label == SILENCE)
    candidates = [(p.cost - s.cost) / (p.coverage - s.coverage)
                  for p in scores
                  if p.label != SILENCE and np.isfinite(p.coverage)
                  and p.coverage > s.coverage + 1e-9]
    return float(min(candidates)) if candidates else float("inf")


def ranking(scores: Sequence[PolicyScore], lam: float) -> list[str]:
    return [p.label for p in sorted(scores, key=lambda p: -p.lagrangian(lam))]


@dataclass
class CheckVerdict:
    break_even: float
    lambda_max: float
    silence_escapable: bool
    efficient_beats_flooding: bool
    lambda_checked: float

    @property
    def sane(self) -> bool:
        return self.silence_escapable and self.efficient_beats_flooding


def check(scores: Sequence[PolicyScore], objective: ConstrainedObjective,
          efficient: Sequence[str] = ("slotted_1p", "dvcast")) -> CheckVerdict:
    """The two conditions a usable constrained objective must satisfy."""
    be = silence_break_even(scores)
    escapable = bool(np.isfinite(be) and be < objective.lambda_max)
    # Just past the break-even, a cheap coverage-meeting scheme should lead, not
    # flooding. If flooding only wins at far higher lambda that is fine too.
    lam = min(be * 1.25, objective.lambda_max) if np.isfinite(be) else objective.lambda_max
    by = {p.label: p.lagrangian(lam) for p in scores}
    beats = any(by.get(e, -np.inf) > by.get("flooding", np.inf) for e in efficient)
    return CheckVerdict(break_even=be, lambda_max=objective.lambda_max,
                        silence_escapable=escapable, efficient_beats_flooding=beats,
                        lambda_checked=lam)


def format_report(scores: Sequence[PolicyScore], verdict: CheckVerdict, title: str,
                  lambdas: Sequence[float]) -> str:
    lines = ["=" * 78, f" OBJECTIVE RANKING CHECK -- {title}", "=" * 78,
             f"{'policy':<17}{'coverage':>9}{'cost':>8}{'credit/n':>10}{'tx':>6}"
             + "".join(f"{'L@' + format(l, 'g'):>9}" for l in lambdas),
             "-" * 78]
    for p in scores:
        lines.append(f"{p.label:<17}{p.coverage:>9.3f}{p.cost:>8.3f}"
                     f"{p.credit_per_at_risk:>10.3f}{p.transmissions:>6d}"
                     + "".join(f"{p.lagrangian(l):>9.3f}" for l in lambdas))
    lines.append("-" * 78)
    for l in lambdas:
        lines.append(f" lambda={l:<6g} ranking: {' > '.join(ranking(scores, l))}")
    lines += ["-" * 78,
              f" silence break-even lambda : {verdict.break_even:.3f} "
              f"(lambda_max {verdict.lambda_max:g}) -> "
              f"{'escapable' if verdict.silence_escapable else 'NOT ESCAPABLE'}",
              f" at lambda={verdict.lambda_checked:.3f}, efficient scheme beats flooding: "
              f"{verdict.efficient_beats_flooding}",
              f" VERDICT: {'SANE' if verdict.sane else 'NOT SANE -- do not train on this'}",
              "=" * 78]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Check an objective ranks policies sanely")
    ap.add_argument("--scenario", default="rural_highway")
    ap.add_argument("--density", type=float, default=40.0)
    ap.add_argument("--weather", default="clear")
    ap.add_argument("--hazard", default="fog_bank")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lambdas", type=float, nargs="*", default=[0.0, 1.0, 2.0, 5.0, 10.0])
    args = ap.parse_args(argv)
    logging.getLogger("aiharp").setLevel(logging.WARNING)

    cfg = load_yaml("agent.yaml")
    objective = ConstrainedObjective.from_config(cfg)
    cfgs = {"phy": load_yaml("phy.yaml"), "hazard": load_yaml("hazard.yaml"),
            "experiment": load_yaml("experiment.yaml")}
    scores = score_references(args.scenario, args.density, args.weather, args.hazard,
                              args.seed, objective, cfgs)
    verdict = check(scores, objective)
    print(format_report(scores, verdict,
                        f"{args.scenario} d={args.density:g} {args.weather} "
                        f"{args.hazard} seed={args.seed}", args.lambdas))
    return 0 if verdict.sane else 2


if __name__ == "__main__":
    raise SystemExit(main())
