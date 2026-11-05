"""Revision 7: the engine's max_decisions_per_epoch option (results/revision7/ADDENDUM.md)."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from common.config import load_yaml
from experiments.run_sim import RunSpec, run_single
from sim.engine import DisseminationEngine, SimSettings

SPEC = dict(scenario="rural_highway", density_veh_km_lane=10.0, weather="clear", policy="dvcast",
            policy_params={"n_slots": 100}, seed=4, duration_s=20.0, corridor_length_m=2000.0)
KEYS = ("rwcr", "pdr", "transmissions", "tir_median_s", "spatial_reach_m")


def exp(cap=None):
    e = copy.deepcopy(load_yaml("experiment.yaml"))
    if cap is not None:
        e["simulation"]["max_decisions_per_epoch"] = cap
    return e


def test_off_by_default():
    assert SimSettings().max_decisions_per_epoch == 0
    assert SimSettings.from_config(exp()).max_decisions_per_epoch == 0


def test_default_run_has_no_counter_and_is_unchanged_by_an_explicit_zero():
    a, ra = run_single(RunSpec(**SPEC), exp_cfg=exp(), return_result=True)
    b, rb = run_single(RunSpec(**SPEC), exp_cfg=exp(0), return_result=True)
    assert "capped_callbacks" not in ra.meta
    for k in KEYS:
        assert a[k] == pytest.approx(b[k], nan_ok=True)
    assert np.array_equal(ra.informed_step, rb.informed_step)


def test_cap_never_fires_in_this_engine_and_changes_nothing():
    """One reception per vehicle per epoch, timers after receptions: no earlier decision can exist in the epoch."""
    a, ra = run_single(RunSpec(**SPEC), exp_cfg=exp(), return_result=True)
    b, rb = run_single(RunSpec(**SPEC), exp_cfg=exp(1), return_result=True)
    assert rb.meta["capped_callbacks"] == 0
    for k in KEYS:
        assert a[k] == pytest.approx(b[k], nan_ok=True)
    assert np.array_equal(ra.informed_step, rb.informed_step)
    assert ra.action_counts == rb.action_counts


def test_when_a_decision_exists_in_the_epoch_the_policy_is_not_called(monkeypatch):
    """Forces the condition this engine cannot reach by itself, so the branch is exercised."""
    orig = DisseminationEngine._transmit

    def marked(self, tx, step, *a, **k):
        self._last_decision[:] = step
        return orig(self, tx, step, *a, **k)
    base, rb = run_single(RunSpec(**SPEC), exp_cfg=exp(), return_result=True)
    monkeypatch.setattr(DisseminationEngine, "_transmit", marked)
    _, rc = run_single(RunSpec(**SPEC), exp_cfg=exp(1), return_result=True)
    assert rc.meta["capped_callbacks"] > 0
    assert sum(rc.action_counts.values()) < sum(rb.action_counts.values())
