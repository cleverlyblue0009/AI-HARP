"""Revision-5 baselines (C4/C5): the pre-registered RAR and SCF-slotted rules."""

from __future__ import annotations

import numpy as np

from agents.base import ActionType, Trigger
from agents.registry import build_policy
from agents.rule_baselines import RelevanceAwareRule, StoreCarryForwardSlotted
from tests.test_policies import make_ctx


def _empty(**kw):
    e = np.zeros(0)
    return make_ctx(neighbours=np.zeros(0, dtype=int), neighbour_distances=e, neighbour_dx=e,
                    neighbour_dy=e, neighbour_vx=e, neighbour_vy=e, neighbour_relevance=e,
                    neighbour_informed=np.zeros(0, dtype=bool), **kw)


def test_rar_probability_formula_and_clipping():
    p = RelevanceAwareRule(abc="[1.0, 0.5, -0.25]")
    ctx = make_ctx(relevance=0.4)                      # 3 neighbours
    assert np.isclose(p.rebroadcast_probability(ctx), 0.4 + 0.5 * (1 - 3 / 30) - 0.25)
    assert RelevanceAwareRule(abc=[2, 1, 0.25]).rebroadcast_probability(ctx) == 1.0
    assert RelevanceAwareRule(abc=[0.5, 0, -0.25]).rebroadcast_probability(make_ctx(relevance=0.1)) == 0.0


def test_rar_defers_by_distance_and_never_repeats():
    p = RelevanceAwareRule(abc=[2, 1, 0.25])           # p = 1
    a = p.decide(make_ctx())
    assert a.kind is ActionType.DEFER and a.cancel_on_duplicates == 1 and a.delay_steps >= 1
    assert p.decide(make_ctx(own_tx_count=1)).kind is ActionType.SUPPRESS
    assert p.decide(make_ctx(trigger=Trigger.ORIGINATE)).kind is ActionType.BROADCAST
    assert RelevanceAwareRule(abc=[0.5, 0, -0.25]).decide(make_ctx(relevance=0.1)).kind is ActionType.SUPPRESS


def test_scf_carries_until_neighbour_or_timeout():
    p = StoreCarryForwardSlotted(slots_timeout="[8, 5]")
    p.reset(10, np.random.default_rng(0))
    assert p.decide(_empty(time_s=1.0)).kind is ActionType.CARRY
    assert p.decide(_empty(time_s=3.0, trigger=Trigger.TIMER)).kind is ActionType.CARRY
    assert p.decide(make_ctx(time_s=4.0, trigger=Trigger.TIMER)).kind is ActionType.BROADCAST
    # a new carry episode times out after 5 s
    assert p.decide(_empty(time_s=10.0, index=4)).kind is ActionType.CARRY
    assert p.decide(_empty(time_s=15.0, index=4, trigger=Trigger.TIMER)).kind is ActionType.SUPPRESS
    # a duplicate while carrying stops it; with neighbours it is slotted 1-persistence
    assert p.decide(_empty(index=5, trigger=Trigger.RECEIVE, duplicate_count=1)).kind is ActionType.SUPPRESS
    assert p.decide(make_ctx(index=6)).kind is ActionType.DEFER


def test_registered():
    assert build_policy("rar", abc="[1, 0, 0]").name == "rar"
    assert build_policy("scf_slotted", slots_timeout="[3, 2]").name == "scf_slotted"
