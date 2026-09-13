"""Phase 2/6 integration tests: engine invariants and metric bounds."""

from __future__ import annotations

import numpy as np
import pytest

from analysis.metrics import compute_metrics
from experiments.run_sim import RunSpec, run_single
from sim.engine import NO_STEP


@pytest.fixture(scope="module")
def run(hz_cfg):
    spec = RunSpec(
        scenario="rural_highway", density_veh_km_lane=10.0, weather="clear",
        policy="flooding", seed=0, duration_s=20.0, corridor_length_m=2000.0,
    )
    metrics, result = run_single(spec, return_result=True)
    return metrics, result


def test_message_is_originated(run):
    _, res = run
    assert res.origin_step >= 0
    assert res.origin_index >= 0


def test_nobody_is_informed_before_the_originator(run):
    _, res = run
    informed = res.informed_step[res.informed_step >= 0]
    assert informed.min() == res.origin_step


def test_originator_has_hop_count_zero(run):
    _, res = run
    assert res.hops[res.origin_index] == 0


def test_hop_counts_are_consistent_with_the_relay_chain(run):
    _, res = run
    for v in np.flatnonzero(res.informed_step >= 0):
        parent = res.informed_by[v]
        if parent == NO_STEP:
            assert v == res.origin_index
        else:
            assert res.hops[v] == res.hops[parent] + 1
            # A relay cannot inform anyone before it was itself informed.
            assert res.informed_step[parent] < res.informed_step[v]


def test_flooding_transmits_exactly_once_per_informed_vehicle(run):
    """The defining property of blind flooding, and the cleanest check that the
    engine is not double-firing scheduled transmissions."""
    _, res = run
    informed = res.informed_step >= 0
    assert set(np.unique(res.tx_count[informed])) <= {0, 1}
    assert res.tx_count[~informed].sum() == 0
    assert res.n_transmissions == int(res.tx_count.sum())


def test_rebroadcast_never_happens_in_the_reception_epoch(run):
    """There is always at least one 100 ms epoch of turnaround."""
    _, res = run
    by_vehicle: dict[int, list[int]] = {}
    for t in res.transmissions:
        by_vehicle.setdefault(t["vehicle"], []).append(t["step"])
    for v, steps in by_vehicle.items():
        assert min(steps) > res.informed_step[v] or v == res.origin_index


def test_metrics_are_in_range(run):
    m, _ = run
    for key in ("pdr", "rwcr", "coverage", "at_risk_coverage", "deadline_miss_rate",
                "tir_uninformed_frac"):
        v = m[key]
        if np.isfinite(v):
            assert 0.0 <= v <= 1.0, f"{key} = {v}"


def test_frame_accounting_balances(run):
    """Every in-range reception attempt is either a success or one named loss."""
    m, res = run
    assert (
        res.n_rx_success + res.n_fail_sensitivity + res.n_fail_sinr + res.n_fail_beacon
        == res.n_rx_attempts
    )


def test_latency_percentiles_are_ordered(run):
    m, _ = run
    if np.isfinite(m["tir_median_s"]) and np.isfinite(m["tir_p95_s"]):
        assert m["tir_median_s"] <= m["tir_p95_s"]
    assert m["latency_mean_s"] <= m["latency_p95_s"] + 1e-9


def test_spatial_reach_does_not_exceed_the_corridor(run):
    m, _ = run
    assert m["spatial_reach_m"] <= 2000.0 + 50.0


def test_run_is_reproducible_under_a_fixed_seed(hz_cfg):
    spec = RunSpec(
        scenario="rural_highway", density_veh_km_lane=10.0, weather="clear",
        policy="flooding", seed=4, duration_s=15.0, corridor_length_m=2000.0,
    )
    a, _ = run_single(spec)
    b, _ = run_single(spec)
    for key in ("rwcr", "pdr", "transmissions", "tir_median_s", "spatial_reach_m"):
        assert a[key] == pytest.approx(b[key], nan_ok=True)


def test_different_seeds_produce_different_outcomes(hz_cfg):
    out = []
    for seed in (0, 1, 2):
        spec = RunSpec(
            scenario="rural_highway", density_veh_km_lane=10.0, weather="clear",
            policy="flooding", seed=seed, duration_s=15.0, corridor_length_m=2000.0,
        )
        m, _ = run_single(spec)
        out.append(m["rwcr"])
    assert len(set(out)) > 1, "seeds are not perturbing the run"


def test_weather_degrades_delivery(hz_cfg):
    """Sanity check on the weather pipeline end to end."""
    base = dict(scenario="rural_highway", density_veh_km_lane=10.0, policy="flooding",
                seed=0, duration_s=20.0, corridor_length_m=2000.0)
    clear, _ = run_single(RunSpec(weather="clear", **base))
    heavy, _ = run_single(RunSpec(weather="heavy_rain", **base))
    assert heavy["comm_range_m"] < clear["comm_range_m"]


@pytest.fixture(scope="module")
def saturated_run(hz_cfg):
    """Flooding with CCA forced busy 99% of the time."""
    from sim.mac import MacModel

    mp = pytest.MonkeyPatch()
    mp.setattr(MacModel, "channel_busy_probability",
               lambda self, n: np.full(np.shape(n), 0.99))
    try:
        spec = RunSpec(
            scenario="rural_highway", density_veh_km_lane=10.0, weather="clear",
            policy="flooding", seed=0, duration_s=20.0, corridor_length_m=2000.0,
        )
        yield run_single(spec, return_result=True)
    finally:
        mp.undo()


def test_a_busy_medium_never_discards_the_originators_frame(saturated_run):
    """The engine used to drop a frame after five busy draws; at an urban d=80
    originator (busy probability 0.81) 28% of episodes never transmitted."""
    _, res = saturated_run
    assert res.origin_index >= 0
    assert res.tx_count[res.origin_index] == 1
    assert res.n_busy_deferrals > 0
    assert res.n_transmissions > 1


def test_a_busy_frame_goes_out_in_its_scheduled_epoch(saturated_run, run):
    """Deferral is sub-epoch: no whole-epoch delay, so first transmission timing
    matches the unloaded run."""
    _, busy = saturated_run
    _, calm = run
    first = lambda r: min(t["step"] for t in r.transmissions)  # noqa: E731
    assert first(busy) == busy.origin_step + 1
    assert first(busy) == first(calm)


def test_flooding_relays_every_informed_vehicle_even_on_a_saturated_medium(saturated_run):
    """No silent relay loss: every informed vehicle still active one epoch after
    being informed transmits exactly once."""
    _, res = saturated_run
    tr = res.trace
    for v in np.flatnonzero(res.informed_step >= 0):
        nxt = int(res.informed_step[v]) + 1
        if nxt < tr.n_steps and tr.active[nxt, v]:
            assert res.tx_count[v] == 1, f"vehicle {v} was informed but never relayed"


def test_removed_busy_deferral_cap_is_refused():
    from sim.engine import SimSettings

    with pytest.raises(ValueError, match="max_busy_deferrals"):
        SimSettings.from_config({"simulation": {"max_busy_deferrals": 5}})


def test_config_hash_changes_with_configuration(hz_cfg):
    a = RunSpec(policy="flooding", seed=0).key()
    b = RunSpec(policy="flooding", seed=1).key()
    c = RunSpec(policy="flooding", seed=0).key()
    assert a == c and a != b
