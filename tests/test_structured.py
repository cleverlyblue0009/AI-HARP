"""Revision 8: STRUCT contains slotted_1p and RAR exactly (results/revision8/ADDENDUM.md, sections 1-2).

Gate before any STRUCT training run:
* >= 1,000 decision contexts recorded from real engine runs on TRAINING-POOL seeds: STRUCT at the equivalence
  parameters makes identical decisions (action, delay, cancel threshold) to slotted_1p (n_slots 5 and 50) and RAR
  (default abc), with 0 mismatches; against RAR the policy stream is left in the identical state (same draws).
* Whole runs: slotted_1p n_slots 5 and 50 vs STRUCT at the matching parameters give byte-identical engine output on
  3 seeds in each of the four headline cells.
* No engine component other than the policy reads the policy RNG stream.
"""

from __future__ import annotations

import dataclasses
import logging
import pickle

import numpy as np
import pytest

from agents.base import ActionType, Trigger
from agents.registry import build_policy
from agents.rule_baselines import RelevanceAwareRule
from agents.structured import N_PARAMS, StructuredPolicy, theta_rar, theta_slotted

HEADLINE = [("rural_highway", 2.0), ("rural_highway", 20.0), ("rural_highway", 80.0), ("urban_nlos", 20.0)]
TRAIN_SEEDS = (100, 101, 102)          # training pool 100-131; never CAL/SEL/EVAL


def _run(policy: str, params: dict, scenario: str, density: float, seed: int):
    from experiments.run_sim import RunSpec, run_single
    logging.disable(logging.WARNING)
    return run_single(RunSpec(scenario=scenario, density_veh_km_lane=density, weather="clear", policy=policy,
                              policy_params=params, seed=seed, hazard_type="fog_bank", backend="fallback"),
                      return_result=True)


# --------------------------------------------------------------------------------------------- contexts --
@pytest.fixture(scope="module")
def contexts():
    """Every first-reception context RAR(1, 0.5, 0) saw on training-pool seeds, with the policy-stream state at the
    moment of the decision (contexts are copied, so the recording changes nothing)."""
    from agents import registry
    rec = []

    class _Rec(RelevanceAwareRule):
        def decide(self, ctx):
            if ctx.trigger is Trigger.RECEIVE:
                rec.append((dataclasses.replace(ctx, rng=None), ctx.rng.bit_generator.state))
            return super().decide(ctx)

    registry._REGISTRY["_rar_rec_test"] = _Rec
    try:
        for seed in TRAIN_SEEDS:
            for sc, d in HEADLINE:
                _run("_rar_rec_test", {"abc": [1.0, 0.5, 0.0]}, sc, d, seed)
    finally:
        registry._REGISTRY.pop("_rar_rec_test", None)
    return rec


def _with_rng(ctx, state):
    g = np.random.default_rng()
    g.bit_generator.state = state
    return dataclasses.replace(ctx, rng=g)


def _key(a):
    return (a.kind, a.delay_steps if a.kind is ActionType.DEFER else None,
            a.cancel_on_duplicates if a.kind is ActionType.DEFER else None)


def test_enough_recorded_contexts(contexts):
    assert len(contexts) >= 1000


def test_parameter_count():
    assert N_PARAMS == 21 <= 40
    for t in (theta_slotted(5), theta_slotted(50), theta_rar(1, 0.5, 0)):
        assert t.shape == (N_PARAMS,)


@pytest.mark.parametrize("n_slots", [5, 50])
def test_struct_equals_slotted_1p_on_recorded_contexts(contexts, n_slots):
    ref = build_policy("slotted_1p", n_slots=n_slots)
    st = StructuredPolicy(theta=theta_slotted(n_slots))
    mism = 0
    for ctx, state in contexts:
        a, b = ref.decide(_with_rng(ctx, state)), st.decide(_with_rng(ctx, state))
        mism += _key(a) != _key(b)
    assert mism == 0, f"{mism} mismatches of {len(contexts)}"


@pytest.mark.parametrize("abc", [(1.0, 0.5, 0.0)])
def test_struct_equals_rar_decisions_and_draws(contexts, abc):
    ref = build_policy("rar", abc=list(abc))
    st = StructuredPolicy(theta=theta_rar(*abc))
    mism = draws = 0
    for ctx, state in contexts:
        ca, cb = _with_rng(ctx, state), _with_rng(ctx, state)
        a, b = ref.decide(ca), st.decide(cb)
        mism += _key(a) != _key(b)
        draws += ca.rng.bit_generator.state != cb.rng.bit_generator.state
    assert mism == 0, f"{mism} decision mismatches of {len(contexts)}"
    assert draws == 0, f"{draws} contexts with different policy-stream consumption"


@pytest.mark.parametrize("abc", [(a, b, c) for a in (0.5, 1.0, 2.0) for b in (0.0, 0.5, 1.0)
                                 for c in (-0.25, 0.0, 0.25)])
def test_every_rar_grid_setting_is_contained(contexts, abc):
    ref = build_policy("rar", abc=list(abc))
    st = StructuredPolicy(theta=theta_rar(*abc))
    sub = contexts[::7]
    assert sum(_key(ref.decide(_with_rng(c, s))) != _key(st.decide(_with_rng(c, s))) for c, s in sub) == 0


@pytest.mark.parametrize("n_slots", [1, 2])
def test_other_representable_slot_counts(contexts, n_slots):
    ref = build_policy("slotted_1p", n_slots=n_slots)
    st = StructuredPolicy(theta=theta_slotted(n_slots))
    assert sum(_key(ref.decide(_with_rng(c, s))) != _key(st.decide(_with_rng(c, s))) for c, s in contexts) == 0


def test_unrepresentable_slot_count_refused():
    with pytest.raises(ValueError):
        theta_slotted(8)


def test_delta_knob_shifts_p_by_015():
    st0 = StructuredPolicy(theta=theta_rar(1.0, 0.5, 0.0))
    st2 = StructuredPolicy(theta=theta_rar(1.0, 0.5, 0.0), delta=2.0)
    x = np.array([1.0, 0.3, 0.2, 0.5])
    assert st0.probability(x) - st2.probability(x) == pytest.approx(0.30)


def test_struct_always_draws_exactly_one_uniform(contexts):
    st = StructuredPolicy(theta=theta_slotted(5))
    for ctx, state in contexts[:200]:
        c = _with_rng(ctx, state)
        g = np.random.default_rng()
        g.bit_generator.state = state
        g.random()
        st.decide(c)
        assert c.rng.bit_generator.state == g.bit_generator.state


# ------------------------------------------------------------------------------------------- whole runs --
def _fingerprint(result, metrics):
    arrays = {k: getattr(result, k) for k in ("informed_step", "hops", "informed_by", "tx_count",
                                               "collisions_caused")}
    counters = {k: getattr(result, k) for k in ("origin_index", "origin_step", "n_transmissions", "n_rx_attempts",
                                                 "n_rx_success", "n_fail_sensitivity", "n_fail_sinr", "n_fail_beacon",
                                                 "n_backoff_ties", "n_hidden_overlaps", "n_busy_deferrals",
                                                 "n_dropped_ttl")}
    meta = {k: v for k, v in result.meta.items() if k != "policy"}
    m = {k: v for k, v in metrics.items() if k not in ("policy", "policy_params", "params_json", "wall_s",
                                                        "cpu_s", "config_hash")}
    return pickle.dumps(({k: np.asarray(v).tobytes() for k, v in arrays.items()}, counters, result.transmissions,
                         result.action_counts, meta, sorted((k, repr(v)) for k, v in m.items())))


@pytest.mark.parametrize("cell", HEADLINE)
@pytest.mark.parametrize("n_slots", [5, 50])
def test_whole_runs_byte_identical_to_slotted_1p(cell, n_slots):
    sc, d = cell
    for seed in TRAIN_SEEDS:
        ma, ra = _run("slotted_1p", {"n_slots": n_slots}, sc, d, seed)
        mb, rb = _run("struct", {"theta": theta_slotted(n_slots).tolist()}, sc, d, seed)
        assert ra.n_transmissions > 0
        assert _fingerprint(ra, ma) == _fingerprint(rb, mb), f"{sc} d={d} seed={seed} n_slots={n_slots}"


# ------------------------------------------------------------------------------ policy-stream isolation --
class _Guard:
    """Stands in for the policy stream: any use at all (draw, state read) raises."""

    def __getattr__(self, name):
        raise AssertionError(f"policy RNG stream accessed: .{name}")


@pytest.mark.parametrize("cell", HEADLINE)
def test_no_engine_component_reads_the_policy_stream(cell, monkeypatch):
    """The engine with the policy stream replaced by a guard that raises on any access runs to completion with a policy
    that itself never draws (slotted_1p, p = 1), and gives the same output as with the real stream."""
    from common.seeding import SeedBundle
    real = SeedBundle.rng
    sc, d = cell
    ma, ra = _run("slotted_1p", {"n_slots": 5}, sc, d, TRAIN_SEEDS[0])

    def guarded(self, stream):
        return _Guard() if stream == "policy" else real(self, stream)

    monkeypatch.setattr(SeedBundle, "rng", guarded)
    mb, rb = _run("slotted_1p", {"n_slots": 5}, sc, d, TRAIN_SEEDS[0])
    assert ra.n_transmissions > 0
    assert _fingerprint(ra, ma) == _fingerprint(rb, mb)


def test_engine_source_never_draws_from_the_policy_stream():
    """Static check: in sim/ the policy stream is only created and passed on, never called."""
    import re
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    for p in (root / "sim").rglob("*.py"):
        src = p.read_text(encoding="utf-8")
        assert not re.search(r"rng_pol\s*\.", src), f"{p.name} calls a method on the policy stream"
        assert 'rng("policy")' not in src or p.name == "engine.py"
