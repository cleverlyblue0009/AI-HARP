"""Calibrate the reward's w1/w2 ratio against the measured baseline front.

The problem
-----------
The reward trades coverage against transmissions:

    r = w1 * d(relevance-weighted informed) - w2 * (transmission cost) - ...

Hand-picking w1 and w2 decides the answer before training starts. Set w2 too
low and the optimal policy is to flood; set it too high and the optimal policy
is silence. Neither is discovered by the agent -- both are chosen by whoever
wrote the config, and the result would be an artefact of that choice.

The calibration
---------------
The measured baseline front already tells us what a transmission is worth at
any given operating point. If a policy costs :math:`c` transmissions per
at-risk vehicle informed, then one transmission buys :math:`1/c` at-risk
vehicles, each carrying mean relevance :math:`\\bar\\rho`. A transmission is
therefore reward-neutral when

.. math::

    w_1 \\cdot \\frac{\\bar\\rho}{c} \\;=\\; w_2
    \\qquad\\Longrightarrow\\qquad
    \\frac{w_2}{w_1} \\;=\\; \\frac{\\bar\\rho}{c}

Above that ratio the agent prefers to suppress, below it to transmit. Setting
the ratio from the *target* operating cost makes the intended behaviour an
explicit, checkable modelling decision instead of a tuning accident.

Measured on rural at 20 veh/km/lane: the baseline cluster sits at c = 0.41 and
flooding at c = 2.11. With a mean at-risk relevance around 0.75 that gives

    target 0.41-class :  w2/w1 = 0.75 / 0.41 = 1.83
    flooding-class    :  w2/w1 = 0.75 / 2.11 = 0.36

so any ratio below ~0.36 makes flooding reward-optimal, and the configured
value must sit near 1.8 for the agent to aim at the baseline cluster rather
than at the scheme nobody deploys. The gap between those two numbers is narrow,
which is exactly why this is calibrated rather than guessed.

``sensitivity()`` sweeps the ratio so the paper can report how the learned
operating point moves with it, instead of claiming one ratio was correct.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from common.logging_utils import get_logger

logger = get_logger("analysis.reward_calibration")


@dataclass
class RewardCalibration:
    """A w1/w2 ratio derived from a target operating cost."""

    target_cost: float           # transmissions per at-risk vehicle informed
    mean_relevance: float        # mean peak relevance over the at-risk set
    w2_over_w1: float
    flooding_cost: float | None = None
    flooding_w2_over_w1: float | None = None
    w3_collision: float = 0.0
    collisions_per_tx_at_target: float = 0.0
    effective_cost: float = 0.0

    @property
    def margin_over_flooding(self) -> float:
        """How far the configured ratio sits above the flood-optimal ratio.

        A value near 1.0 means the reward barely distinguishes the target
        behaviour from flooding, and the learned policy will be sensitive to
        noise in the advantage estimate.
        """
        if not self.flooding_w2_over_w1:
            return float("nan")
        return self.w2_over_w1 / self.flooding_w2_over_w1

    def describe(self) -> str:
        lines = [
            f"target operating cost      : {self.target_cost:.3f} tx per at-risk informed",
            f"mean at-risk relevance     : {self.mean_relevance:.3f}",
            f"total price of a tx        : {self.effective_cost:.3f}",
            f"  of which collision term  : {self.w3_collision * self.collisions_per_tx_at_target:.3f}"
            f"  (w3={self.w3_collision} x {self.collisions_per_tx_at_target:.2f} coll/tx)",
            f"=> calibrated w2/w1        : {self.w2_over_w1:.3f}",
        ]
        if self.flooding_w2_over_w1:
            lines += [
                f"flood-optimal w2/w1        : {self.flooding_w2_over_w1:.3f} "
                f"(cost {self.flooding_cost:.2f})",
                f"margin over flood-optimal  : {self.margin_over_flooding:.2f}x",
            ]
        return "\n".join(lines)


def calibrate(
    target_cost: float,
    mean_relevance: float,
    flooding_cost: float | None = None,
    w3_collision: float = 0.0,
    collisions_per_tx_at_target: float = 0.0,
) -> RewardCalibration:
    """Ratio at which a transmission is reward-neutral at ``target_cost``.

    The collision term is part of the price of transmitting, so it must enter
    the calibration. A transmission costs ``w2 + w3 * E[collisions it causes]``,
    not ``w2`` -- and the difference is not small: measured at rural d=20, a
    transmission is blamed for ~2.0 lost receptions at the target operating
    point and ~16.3 under flooding. Calibrating w2 alone and then adding a
    collision term silently multiplies the effective cost (by 5.5x at the
    originally configured w3 = 0.5), pushing the break-even to 0.075 tx per
    at-risk vehicle informed against a target of 0.41. The agent would learn
    near-silence, and that would look like a finding.

    So ``w2`` is solved for, given ``w3`` and the collision rate expected at
    the target:

        w2 = mean_relevance / target_cost - w3 * collisions_per_tx_at_target
    """
    if target_cost <= 0:
        raise ValueError("target_cost must be positive")
    total = mean_relevance / target_cost
    collision_share = w3_collision * collisions_per_tx_at_target
    ratio = total - collision_share
    if ratio <= 0:
        raise ValueError(
            f"w3 * collisions ({collision_share:.2f}) already exceeds the total "
            f"transmission price ({total:.2f}); lower w3."
        )
    flood_ratio = (mean_relevance / flooding_cost) if flooding_cost else None
    return RewardCalibration(
        target_cost=target_cost, mean_relevance=mean_relevance, w2_over_w1=ratio,
        flooding_cost=flooding_cost, flooding_w2_over_w1=flood_ratio,
        w3_collision=w3_collision,
        collisions_per_tx_at_target=collisions_per_tx_at_target,
        effective_cost=total,
    )


def calibrate_from_cells(cells, target: float = 0.95, quantile: float = 0.0):
    """Derive the ratio from measured Pareto cells.

    ``quantile`` selects how ambitious the target is: 0.0 aims at the very best
    baseline cost observed (the 0.41 cluster), 0.5 at the median. Aiming below
    the best observed cost would ask the agent to beat every baseline in every
    cell simply by construction of the reward, which is not a calibration but a
    thumb on the scale.
    """
    from analysis.comparator import per_cell_oracle_best

    oracle = per_cell_oracle_best(cells, target=target)
    costs = [c for _, c in oracle.per_cell.values() if np.isfinite(c)]
    if not costs:
        raise ValueError("no cell reaches the target; cannot calibrate")
    target_cost = float(np.quantile(costs, quantile)) if quantile > 0 else float(min(costs))

    rels, floods = [], []
    for cell in cells:
        for curve in cell.curves.values():
            for p in curve.points:
                r = p.extras.get("mean_at_risk_relevance")
                if r is not None and np.isfinite(r):
                    rels.append(r)
        f = cell.cost_at_matched("flooding", target)
        if np.isfinite(f):
            floods.append(f)
    mean_rel = float(np.mean(rels)) if rels else 0.75
    return calibrate(target_cost, mean_rel, float(np.mean(floods)) if floods else None)


def sensitivity(
    cal: RewardCalibration, ratios: Sequence[float] = (0.25, 0.5, 1.0, 1.8, 3.0, 6.0)
) -> list[dict[str, float]]:
    """Implied break-even operating cost for a range of w2/w1 ratios.

    This is the table the paper reports instead of asserting that one ratio was
    right: each row says what operating cost that ratio makes reward-neutral,
    and therefore roughly where the learned policy should settle.
    """
    out = []
    for r in ratios:
        implied = cal.mean_relevance / r if r > 0 else float("inf")
        out.append({
            "w2_over_w1": float(r),
            "implied_break_even_cost": float(implied),
            "floods": bool(cal.flooding_w2_over_w1 and r < cal.flooding_w2_over_w1),
        })
    return out


def format_sensitivity(cal: RewardCalibration, rows: list[dict[str, float]]) -> str:
    lines = ["=" * 78, " REWARD CALIBRATION (w1/w2)", "=" * 78, cal.describe(), "",
             f"{'w2/w1':>10}{'implied break-even cost':>28}{'regime':>20}", "-" * 78]
    for r in rows:
        regime = "FLOODS" if r["floods"] else "suppresses"
        mark = "  <-- configured" if abs(r["w2_over_w1"] - cal.w2_over_w1) < 1e-6 else ""
        lines.append(
            f"{r['w2_over_w1']:>10.2f}{r['implied_break_even_cost']:>28.3f}"
            f"{regime:>20}{mark}"
        )
    lines.append("=" * 78)
    lines.append(" 'implied break-even cost' is the transmissions-per-at-risk-informed at")
    lines.append(" which a transmission is reward-neutral. The learned policy should settle")
    lines.append(" near it. Ratios marked FLOODS make blind flooding reward-optimal.")
    lines.append("=" * 78)
    return "\n".join(lines)
