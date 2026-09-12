"""Phase 2 MAC tests: the collision model and channel access."""

from __future__ import annotations

import numpy as np
import pytest


def test_cw_slots_spans_zero_to_cwmin(mac):
    assert mac.cw_slots == mac.cw_min + 1


def test_tie_probability_is_zero_for_a_lone_transmitter(mac):
    assert mac.tie_probability(0) == 0.0
    assert mac.tie_probability(1) == 0.0


def test_tie_probability_matches_the_analytic_formula(mac):
    w = mac.cw_slots
    for n in (2, 3, 5, 10):
        assert mac.tie_probability(n) == pytest.approx(1 - (1 - 1 / w) ** (n - 1))


def test_tie_probability_is_monotonic_and_bounded(mac):
    p = [mac.tie_probability(n) for n in range(1, 60)]
    assert p == sorted(p)
    assert all(0.0 <= x <= 1.0 for x in p)
    assert p[-1] > 0.9  # the broadcast storm


def test_two_contenders_collide_with_probability_one_over_w(mac):
    assert mac.tie_probability(2) == pytest.approx(1.0 / mac.cw_slots)


def test_empirical_backoff_ties_match_tie_probability(mac):
    """The engine draws slots rather than using the closed form; they must agree."""
    rng = np.random.default_rng(0)
    n, trials = 6, 20_000
    collisions = 0
    for _ in range(trials):
        slots = mac.draw_backoff_slots(n, rng)
        # Did transmitter 0 share its slot with anyone?
        collisions += int((slots[1:] == slots[0]).any())
    assert collisions / trials == pytest.approx(mac.tie_probability(n), abs=0.01)


def test_backoff_slots_are_uniform_over_the_window(mac):
    rng = np.random.default_rng(1)
    slots = mac.draw_backoff_slots(200_000, rng)
    assert slots.min() == 0
    assert slots.max() == mac.cw_min
    counts = np.bincount(slots, minlength=mac.cw_slots)
    assert np.allclose(counts / counts.sum(), 1 / mac.cw_slots, atol=0.01)


# ------------------------------------------------------------ hidden nodes --
def test_hidden_overlap_probability_is_the_vulnerable_window_fraction(mac):
    epoch = 0.1
    assert mac.hidden_overlap_probability(epoch) == pytest.approx(
        mac.vulnerable_window_s / epoch
    )


def test_vulnerable_window_is_two_frame_durations(mac):
    assert mac.vulnerable_window_s == pytest.approx(2.0 * mac.frame_duration_s)


def test_hidden_overlap_probability_is_capped_at_one(mac):
    assert mac.hidden_overlap_probability(1e-9) == 1.0


# ------------------------------------------------------- background beacons --
def test_beacon_survival_falls_with_neighbour_count(mac):
    p = mac.beacon_survival_probability(np.array([0, 5, 20, 100]))
    assert p[0] == pytest.approx(1.0)
    assert np.all(np.diff(p) < 0)
    assert np.all((p >= 0) & (p <= 1))


def test_beacon_survival_matches_poisson_thinning(mac):
    n = 25
    lam = n * mac.background_beacon_hz
    assert float(mac.beacon_survival_probability(n)) == pytest.approx(
        np.exp(-lam * mac.vulnerable_window_s)
    )


def test_channel_busy_probability_rises_with_density(mac):
    p = mac.channel_busy_probability(np.array([0, 10, 50, 200]))
    assert p[0] == pytest.approx(0.0)
    assert np.all(np.diff(p) >= 0)
    assert np.all(p <= 0.99)


def test_disabling_beacons_removes_background_loss(phy_cfg, phy):
    from sim.mac import build_mac

    cfg = {**phy_cfg, "mac": {**phy_cfg["mac"], "background_beacon_enabled": False}}
    quiet = build_mac(cfg, phy)
    assert float(quiet.beacon_survival_probability(100)) == 1.0
    assert float(quiet.channel_busy_probability(100)) == 0.0


# -------------------------------------------------------------------- DCC --
def test_dcc_leaves_the_beacon_rate_alone_at_low_density(mac):
    assert float(mac.effective_beacon_hz(2)) == pytest.approx(mac.background_beacon_hz)


def test_dcc_reduces_the_beacon_rate_under_congestion(mac):
    rates = mac.effective_beacon_hz(np.array([2, 50, 200, 800]))
    assert np.all(np.diff(rates) <= 0)
    assert rates[-1] < mac.background_beacon_hz


def test_dcc_respects_the_standards_rate_floor(mac):
    assert float(mac.effective_beacon_hz(100_000)) == pytest.approx(mac.dcc_min_beacon_hz)


def test_dcc_holds_channel_load_near_the_target(mac):
    """The property that matters: the medium must not saturate.

    Without DCC the offered load grows without bound, the channel is sensed
    busy every epoch, and no hazard message is ever transmitted -- a modelling
    artefact that looked like a result until the smoke test caught it.
    """
    n = np.array([10, 100, 400, 1000])
    cbr = mac.beacon_arrival_rate(n) * mac.beacon_duration_s
    assert np.all(cbr <= mac.dcc_target_cbr + 1e-9)


def test_disabling_dcc_lets_the_channel_saturate(phy_cfg, phy):
    """Guard on the regression itself, so the ablation stays available."""
    from sim.mac import build_mac

    cfg = {**phy_cfg, "mac": {**phy_cfg["mac"], "dcc_enabled": False}}
    no_dcc = build_mac(cfg, phy)
    assert float(no_dcc.channel_busy_probability(800)) > 0.9
    assert float(mac_busy(phy_cfg, phy, 800)) < 0.9


def mac_busy(phy_cfg, phy, n):
    from sim.mac import build_mac

    return build_mac(phy_cfg, phy).channel_busy_probability(n)


def test_frame_and_beacon_durations_are_consistent(mac, phy):
    assert mac.frame_duration_s == pytest.approx(phy.frame_duration_s(mac.frame_bytes))
    assert mac.beacon_duration_s == pytest.approx(phy.frame_duration_s(mac.background_beacon_bytes))


# --------------------------------------------------- inert parameter guard --
def test_inert_mac_parameters_are_not_read_by_any_computation():
    """configs/phy.yaml marks slot/SIFS/DIFS/CWmax/AIFSN as [INERT]: recorded
    for the paper's parameter table but read by no computation, because at a
    100 ms decision epoch interframe spacing is four orders of magnitude below
    the timestep.

    They are also [STD-UNVERIFIED] -- the primary IEEE/ETSI documents are
    paywalled and were not accessed. That is acceptable only for as long as
    they stay inert. This test fails the moment one is wired into a
    computation, forcing verification before it can affect a result.
    """
    import inspect

    import sim.mac as mac_module

    src = inspect.getsource(mac_module)
    body = src.split("def build_mac", 1)[0]        # ignore the constructor
    for field in ("slot_time_s", "sifs_s", "difs_s", "cw_max", "aifsn"):
        uses = [ln.strip() for ln in body.splitlines()
                if f"self.{field}" in ln or f".{field}" in ln and "float" not in ln]
        assert not uses, (
            f"{field} is marked [INERT] in configs/phy.yaml but is now used: {uses}. "
            "Verify it against the primary standard and retag it before relying on it."
        )
