"""ETSI CBF validation policy: timer, epoch rounding, cancel, destination area."""

from __future__ import annotations

import pytest

from agents.base import ActionType, Trigger
from agents.cbf import EtsiCbf, cbf_timeout_s
from agents.registry import BASELINE_POLICIES, build_policy
from tests.test_policies import make_ctx


def test_timeout_endpoints_follow_the_etsi_formula():
    assert cbf_timeout_s(0.0, 0.001, 0.1, 1000.0) == pytest.approx(0.1)
    assert cbf_timeout_s(1000.0, 0.001, 0.1, 1000.0) == pytest.approx(0.001)
    assert cbf_timeout_s(5000.0, 0.001, 0.1, 1000.0) == pytest.approx(0.001)   # clamped
    assert cbf_timeout_s(500.0, 0.001, 0.1, 1000.0) == pytest.approx(0.0505)


def test_ten_millisecond_epochs_resolve_the_ordering():
    pol = EtsiCbf()
    near = pol.decide(make_ctx(sender_distance_m=50.0, dt=0.01))
    far = pol.decide(make_ctx(sender_distance_m=700.0, dt=0.01))
    assert near.kind is far.kind is ActionType.DEFER
    assert near.delay_steps == 10          # 95.05 ms -> 10 epochs
    assert far.delay_steps == 4            # 30.7 ms -> 4 epochs
    assert near.cancel_on_duplicates == far.cancel_on_duplicates == 1


def test_hundred_millisecond_epochs_collapse_every_timer_to_one_epoch():
    pol = EtsiCbf()
    for d in (0.0, 300.0, 900.0):
        assert pol.decide(make_ctx(sender_distance_m=d, dt=0.1)).delay_steps == 1


def test_outside_the_destination_area_suppresses_but_originates():
    pol = EtsiCbf(area_x_min_m=1000.0, area_x_max_m=2000.0)
    assert pol.decide(make_ctx(x=500.0, sender_distance_m=100.0, dt=0.01)).kind is ActionType.SUPPRESS
    assert pol.decide(make_ctx(x=1500.0, sender_distance_m=100.0, dt=0.01)).kind is ActionType.DEFER
    assert pol.decide(make_ctx(x=500.0, trigger=Trigger.ORIGINATE, dt=0.01)).kind is ActionType.BROADCAST


def test_registered_for_validation_but_not_a_baseline():
    assert build_policy("etsi_cbf").name == "etsi_cbf"
    assert "etsi_cbf" not in BASELINE_POLICIES


def test_rejects_bad_parameters():
    with pytest.raises(ValueError):
        EtsiCbf(to_min_ms=0.0)
    with pytest.raises(ValueError):
        EtsiCbf(to_min_ms=50.0, to_max_ms=10.0)
