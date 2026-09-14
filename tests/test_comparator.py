"""The headline comparator's matched-quality rule, on synthetic cells."""

from __future__ import annotations

import numpy as np
import pytest

from analysis.comparator import (
    MISS_METRIC, Cell, CellKey, best_fixed_baseline, compare_policy, per_cell_oracle_best,
)
from analysis.pareto import OperatingPoint, PolicyCurve

KEY = CellKey("rural_highway", 20.0, "clear", "fog_bank")


def _pt(policy, value, quality, cost, miss):
    return OperatingPoint(policy=policy, param="k", value=value, quality=quality,
                          quality_std=0.0, cost=cost, cost_std=0.0, n_seeds=10,
                          extras={MISS_METRIC: miss})


def _cell(key=KEY):
    """`slow` gets cheaper by waiting: its cheapest point matches RWCR but misses
    many more deadlines than the best scheme in the cell."""
    return Cell(key=key, curves={
        "fast": PolicyCurve("fast", [_pt("fast", 1, 0.95, 1.00, 0.10),
                                     _pt("fast", 2, 0.85, 0.60, 0.20)]),
        "slow": PolicyCurve("slow", [_pt("slow", 5, 0.95, 0.90, 0.12),
                                     _pt("slow", 50, 0.95, 0.40, 0.40)]),
    })


def test_a_cheap_point_that_waits_past_deadlines_is_disqualified():
    cell = _cell()
    bound = cell.resolve_miss_bound(0.05)
    assert bound == pytest.approx(0.15)
    assert cell.cost_at_matched("slow", 0.90, miss_bound=bound) == pytest.approx(0.90)


def test_without_the_guard_the_slow_point_wins():
    cell = _cell()
    assert cell.cost_at_matched("slow", 0.90) == pytest.approx(0.40)
    oracle = per_cell_oracle_best([cell], target=0.90, mode="absolute", miss_margin=None)
    assert oracle.label(KEY) == "slow" and oracle.cost(KEY) == pytest.approx(0.40)


def test_oracle_best_respects_the_guard():
    oracle = per_cell_oracle_best([_cell()], target=0.90, mode="absolute", miss_margin=0.05)
    assert oracle.label(KEY) == "slow"
    assert oracle.cost(KEY) == pytest.approx(0.90)


def test_interpolation_never_uses_a_disqualified_setting():
    """The only point cheaper than `fast`'s qualifier misses the guard, so the
    crossing cannot be interpolated towards it."""
    cell = Cell(key=KEY, curves={
        "fast": PolicyCurve("fast", [_pt("fast", 1, 0.95, 1.00, 0.10),
                                     _pt("fast", 2, 0.85, 0.20, 0.90)]),
    })
    assert cell.cost_at_matched("fast", 0.90) == pytest.approx(0.60)       # unguarded
    assert cell.cost_at_matched("fast", 0.90,
                                miss_bound=cell.resolve_miss_bound(0.05)) == pytest.approx(1.00)


def _noisy_d80_cell():
    """The rural d=80 p_persistence_03 case: p=0.08 reaches higher RWCR than
    the dearer p=0.1 (seed noise), and p=0.1 narrowly fails the guard."""
    return Cell(key=KEY, curves={
        "p": PolicyCurve("p", [_pt("p", 0.08, 0.8638, 0.769, 0.3607),
                               _pt("p", 0.10, 0.8470, 1.089, 0.3744),
                               _pt("p", 0.15, 0.9044, 1.726, 0.3266)]),
        "ref": PolicyCurve("ref", [_pt("ref", 1, 0.9150, 11.75, 0.3180)]),
    })


def test_no_interpolation_across_an_anchor_that_fails_the_guard():
    cell = _noisy_d80_cell()
    tq = 0.8695
    tight = cell.cost_at_matched("p", tq, miss_bound=cell.resolve_miss_bound(0.05))
    loose = cell.cost_at_matched("p", tq, miss_bound=cell.resolve_miss_bound(0.10))
    assert tight == pytest.approx(1.726)          # p=0.1 anchor fails: no jump to p=0.08
    assert loose == pytest.approx(1.089 + (tq - 0.8470) / (0.9044 - 0.8470) * (1.726 - 1.089))


@pytest.mark.parametrize("make", [_cell, _noisy_d80_cell])
def test_loosening_the_margin_never_raises_matched_cost(make):
    cell = make()
    for policy in cell.curves:
        costs = [cell.cost_at_matched(policy, 0.8695, miss_bound=cell.resolve_miss_bound(m))
                 for m in (0.0, 0.02, 0.05, 0.10, 0.5)]
        costs.append(cell.cost_at_matched(policy, 0.8695))          # guard off
        assert all(a >= b - 1e-12 for a, b in zip(costs, costs[1:])), (policy, costs)


def test_fixed_best_is_charged_where_its_setting_misses_the_guard():
    fixed = best_fixed_baseline([_cell()], target=0.90, mode="absolute", miss_margin=0.05)
    assert fixed.label != "slow(k=50)"
    unguarded = best_fixed_baseline([_cell()], target=0.90, mode="absolute", miss_margin=None)
    assert unguarded.label == "slow(k=50)"


def test_scored_policy_cannot_set_the_best_miss_rate():
    """Like the RWCR ceiling, the best miss rate excludes the policy scored."""
    cell = _cell()
    cell.curves["agent"] = PolicyCurve("agent", [_pt("agent", 0, 0.95, 0.5, 0.0)])
    res = compare_policy([cell], "agent", target=0.90, mode="absolute", miss_margin=0.05)
    assert res.oracle.miss_bound[KEY] == pytest.approx(0.15)
    assert res.agent_cost[KEY] == pytest.approx(0.5)
    assert res.miss_margin == 0.05


def test_an_agent_that_waits_is_disqualified_too():
    cell = _cell()
    cell.curves["agent"] = PolicyCurve("agent", [_pt("agent", 0, 0.95, 0.3, 0.50)])
    res = compare_policy([cell], "agent", target=0.90, mode="absolute", miss_margin=0.05)
    assert not np.isfinite(res.agent_cost[KEY])


def test_points_without_a_recorded_miss_rate_cannot_meet_a_finite_bound():
    cell = _cell()
    p = _pt("fast", 3, 0.99, 0.1, np.nan)
    assert not cell.within_miss_bound(p, 0.2)
    assert cell.within_miss_bound(p, float("inf"))


def test_guard_is_skipped_with_a_warning_when_no_miss_rates_exist():
    cell = Cell(key=CellKey("urban_nlos", 5.0, "clear", "crash"), curves={
        "a": PolicyCurve("a", [_pt("a", 1, 0.9, 1.0, np.nan)]),
    })
    assert cell.resolve_miss_bound(0.05) == float("inf")
