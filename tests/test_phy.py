"""Phase 2 PHY tests: link budget, dual-slope path loss, fading, weather."""

from __future__ import annotations

import math

import numpy as np
import pytest

from sim.phy import (
    SPEED_OF_LIGHT,
    build_phy,
    free_space_loss_db,
)


# --------------------------------------------------------------- free space --
def test_free_space_loss_matches_friis():
    f = 5.9e9
    d = 100.0
    expected = 20 * math.log10(4 * math.pi * d * f / SPEED_OF_LIGHT)
    assert free_space_loss_db(d, f) == pytest.approx(expected)


def test_free_space_loss_at_1m_5900mhz():
    # 20*log10(4*pi*1*5.9e9/c) = 47.86 dB; the reference intercept used by the
    # log-distance model when reference_loss_db is null.
    assert free_space_loss_db(1.0, 5.9e9) == pytest.approx(47.86, abs=0.01)


def test_free_space_doubling_distance_adds_6db():
    a = free_space_loss_db(100.0, 5.9e9)
    b = free_space_loss_db(200.0, 5.9e9)
    assert b - a == pytest.approx(6.02, abs=0.01)


# --------------------------------------------------------------- dual slope --
def test_path_loss_is_continuous_at_breakpoint(phy):
    d_bp = phy.breakpoint_distance_m
    lo = float(phy.path_loss_db(d_bp - 1e-6))
    hi = float(phy.path_loss_db(d_bp + 1e-6))
    assert lo == pytest.approx(hi, abs=1e-4)


def test_path_loss_slopes_match_configured_exponents(phy):
    d_bp = phy.breakpoint_distance_m
    near = float(phy.path_loss_db(d_bp / 2)) - float(phy.path_loss_db(d_bp / 4))
    far = float(phy.path_loss_db(d_bp * 4)) - float(phy.path_loss_db(d_bp * 2))
    assert near == pytest.approx(10 * phy.path_loss_exponent * math.log10(2), abs=1e-6)
    assert far == pytest.approx(10 * phy.path_loss_exponent_far * math.log10(2), abs=1e-6)


def test_path_loss_monotonic_increasing(phy):
    d = np.array([1.0, 10.0, 50.0, 150.0, 300.0, 1000.0, 5000.0])
    pl = phy.path_loss_db(d)
    assert np.all(np.diff(pl) > 0)


def test_path_loss_below_reference_distance_is_clamped(phy):
    assert float(phy.path_loss_db(0.1)) == pytest.approx(float(phy.path_loss_db(1.0)))


def test_single_slope_would_overestimate_range(phy):
    """Regression guard for the bug the dual-slope model fixed.

    With a single n = 1.9 slope the nominal DSRC range comes out near 2 km,
    which would make a 10 km corridor fully connected and delete the sparse
    regime the paper studies. The dual-slope model must land in the measured
    300-900 m band.
    """
    assert 300.0 < phy.nominal_range_m < 900.0


# -------------------------------------------------------------- link budget --
def test_noise_floor_is_minus_95_dbm_at_10mhz_nf9(phy):
    # -174 dBm/Hz + 10*log10(10e6) + 9 = -174 + 70 + 9 = -95 dBm
    assert phy.noise_dbm == pytest.approx(-95.0, abs=0.01)


def test_median_rx_power_decreases_with_distance(phy):
    p = phy.median_rx_power_dbm(np.array([10.0, 100.0, 500.0, 2000.0]))
    assert np.all(np.diff(p) < 0)


def test_range_for_power_inverts_the_link_budget(phy):
    for target in (-70.0, -85.0, -92.0):
        d = phy.range_for_power_m(target)
        assert float(phy.median_rx_power_dbm(d)) == pytest.approx(target, abs=0.05)


def test_carrier_sense_range_exceeds_nominal_range(phy):
    # CS threshold (-92 dBm) is below sensitivity (-85 dBm), so a node senses
    # further than it can decode. If this inverts, the hidden-terminal model
    # is meaningless.
    assert phy.range_for_power_m(-92.0) > phy.nominal_range_m


# ------------------------------------------------------------------ fading --
def test_nakagami_power_gain_has_unit_mean(phy):
    rng = np.random.default_rng(0)
    gain_db = phy.fading_db(200_000, rng)
    mean_linear = np.mean(10 ** (gain_db / 10))
    assert mean_linear == pytest.approx(1.0, rel=0.02)


def test_nakagami_variance_decreases_with_m(phy_cfg):
    rng = np.random.default_rng(1)
    weak = build_phy(phy_cfg, "urban_grid", "clear", seed=0)     # m = 1.5
    strong = build_phy(phy_cfg, "rural_highway", "clear", seed=0)  # m = 3.0
    assert strong.nakagami_m > weak.nakagami_m
    v_weak = np.var(10 ** (weak.fading_db(100_000, rng) / 10))
    v_strong = np.var(10 ** (strong.fading_db(100_000, rng) / 10))
    assert v_strong < v_weak


# --------------------------------------------------------------- shadowing --
def test_shadowing_is_deterministic_for_a_link(phy):
    tx = np.array([3, 3, 3])
    rx = np.array([9, 9, 9])
    d = np.array([200.0, 200.0, 200.0])
    s = phy.shadowing_db(tx, rx, d)
    assert np.allclose(s, s[0])


def test_shadowing_is_symmetric(phy):
    a = phy.shadowing_db(np.array([4]), np.array([11]), np.array([180.0]))
    b = phy.shadowing_db(np.array([11]), np.array([4]), np.array([180.0]))
    assert a == pytest.approx(b)


def test_shadowing_redraws_after_a_decorrelation_length(phy):
    d0 = 200.0
    d1 = d0 + 2 * phy.shadowing_decorrelation_m
    a = phy.shadowing_db(np.array([4]), np.array([11]), np.array([d0]))
    b = phy.shadowing_db(np.array([4]), np.array([11]), np.array([d1]))
    assert a != pytest.approx(b)


def test_shadowing_has_configured_standard_deviation(phy):
    n = 40_000
    tx = np.arange(n) % 500
    rx = 500 + (np.arange(n) // 500)
    d = 100.0 + (np.arange(n) % 97) * 30.0
    s = phy.shadowing_db(tx, rx, d)
    assert np.std(s) == pytest.approx(phy.shadowing_std_db, rel=0.05)
    assert np.mean(s) == pytest.approx(0.0, abs=0.1)


# ------------------------------------------------------------------ weather --
# NOTE: the ITU-R hydrometeor models themselves are validated against the
# Recommendations' own published tables in tests/test_itu.py. What is tested
# here is only how build_phy() wires them into the link budget.
@pytest.mark.parametrize("weather", ["clear", "moderate_rain", "heavy_rain", "dense_fog"])
def test_weather_never_improves_the_link(phy_cfg, weather):
    clear = build_phy(phy_cfg, "rural_highway", "clear", seed=0)
    wx = build_phy(phy_cfg, "rural_highway", weather, seed=0)
    assert wx.nominal_range_m <= clear.nominal_range_m + 1e-6
    assert wx.weather.total_db_per_km >= 0.0


def test_weather_has_no_measurable_channel_effect_at_5_9ghz(phy_cfg):
    """The consequence of deleting the uncitable empirical excess-loss term.

    With only genuine ITU-R hydrometeor attenuation in the model, the worst
    weather in the config changes the nominal range by well under a metre.
    Weather in this project degrades dissemination through TRAFFIC (speed and
    headway), not through the radio channel -- and the paper must say so
    rather than implying a channel effect it does not model.
    """
    clear = build_phy(phy_cfg, "rural_highway", "clear", seed=0)
    for weather in ("moderate_rain", "heavy_rain", "dense_fog"):
        wx = build_phy(phy_cfg, "rural_highway", weather, seed=0)
        delta = clear.nominal_range_m - wx.nominal_range_m
        frac = delta / clear.nominal_range_m
        # Worst case is heavy rain: ~1.4 m off a 562 m range, i.e. 0.24%.
        # Shadowing alone is 3 dB standard deviation, which moves the range by
        # tens of metres -- weather is far below the noise of the channel.
        assert 0.0 <= frac < 0.005, (
            f"{weather}: range changed by {delta:.3f} m ({frac:.3%})"
        )


def test_empirical_excess_loss_is_disabled_by_default(phy_cfg):
    """Guard on the deletion: those numbers were invented and must stay off
    until a citable measurement replaces them."""
    assert phy_cfg["weather"]["enable_empirical_excess_loss"] is False
    for weather in ("clear", "moderate_rain", "heavy_rain", "dense_fog"):
        wx = build_phy(phy_cfg, "rural_highway", weather, seed=0)
        assert wx.weather.excess_db_per_km == 0.0
        assert wx.weather.exponent_delta == 0.0


def test_excess_loss_can_be_re_enabled_for_a_sensitivity_study(phy_cfg):
    """The mechanism is retained so a cited value can be dropped in later."""
    cfg = {**phy_cfg, "weather": {**phy_cfg["weather"], "enable_empirical_excess_loss": True}}
    cfg["weather"]["conditions"] = {
        k: {**v} for k, v in phy_cfg["weather"]["conditions"].items()
    }
    cfg["weather"]["conditions"]["heavy_rain"]["excess_loss_db_per_km"] = 5.0
    wx = build_phy(cfg, "rural_highway", "heavy_rain", seed=0)
    clear = build_phy(cfg, "rural_highway", "clear", seed=0)
    assert wx.weather.excess_db_per_km == 5.0
    assert wx.nominal_range_m < clear.nominal_range_m


def test_weather_terms_stay_separable(phy_cfg):
    """ITU-R and empirical excess terms must never be silently merged."""
    heavy = build_phy(phy_cfg, "rural_highway", "heavy_rain", seed=0)
    w = heavy.weather
    assert w.itu_total_db_per_km == pytest.approx(w.itu_rain_db_per_km + w.itu_fog_db_per_km)
    assert w.total_db_per_km == pytest.approx(w.itu_total_db_per_km + w.excess_db_per_km)


def test_rain_attenuation_is_computed_at_the_actual_carrier(phy_cfg):
    """Not read from a tabulated row at a neighbouring frequency."""
    from sim.itu import rain_specific_attenuation_db_per_km

    heavy = build_phy(phy_cfg, "rural_highway", "heavy_rain", seed=0)
    expected = rain_specific_attenuation_db_per_km(
        25.0, float(phy_cfg["radio"]["carrier_frequency_hz"]) / 1e9,
        phy_cfg["weather"]["itu_rain_polarisation"],
    )
    assert heavy.weather.itu_rain_db_per_km == pytest.approx(expected)


def test_weather_behavioural_factors_survive(phy_cfg):
    """Deleting the channel term must not delete the traffic term, which is
    the weather effect this project actually models."""
    for weather, expect_slower in (("clear", False), ("heavy_rain", True), ("dense_fog", True)):
        wx = build_phy(phy_cfg, "rural_highway", weather, seed=0)
        assert (wx.weather.speed_factor < 1.0) is expect_slower
        assert (wx.weather.headway_factor > 1.0) is expect_slower


def test_frame_duration_matches_data_rate(phy):
    # 300 B at 6 Mb/s = 400 us of payload, plus a 40 us preamble/SIGNAL.
    assert phy.frame_duration_s(300) == pytest.approx(40e-6 + 300 * 8 / 6e6)


# ----------------------------------------------------- derived breakpoint --
def test_two_ray_breakpoint_matches_the_geometry():
    """d_bp = 4 h_t h_r / lambda. At 1.5 m and 5.9 GHz this is 177 m."""
    from sim.phy import two_ray_breakpoint_m

    assert two_ray_breakpoint_m(1.5, 1.5, 5.9e9) == pytest.approx(177.1, abs=0.5)


def test_breakpoint_scales_with_antenna_height():
    """Truck-height antennas push the breakpoint out, which is why the model
    notes its single fleet-average height as a limitation."""
    from sim.phy import two_ray_breakpoint_m

    car = two_ray_breakpoint_m(1.5, 1.5, 5.9e9)
    truck = two_ray_breakpoint_m(3.0, 3.0, 5.9e9)
    assert truck == pytest.approx(4 * car, rel=1e-9)


def test_breakpoint_is_derived_not_configured(phy_cfg, phy):
    """The config carries antenna height, not a recalled fitted breakpoint."""
    assert phy_cfg["path_loss"]["breakpoint_distance_m"] is None
    assert phy.breakpoint_distance_m == pytest.approx(177.1, abs=0.5)


def test_breakpoint_can_still_be_overridden(phy_cfg):
    cfg = {**phy_cfg, "path_loss": {**phy_cfg["path_loss"],
                                    "breakpoint_distance_m": {"rural_highway": 100.0}}}
    p = build_phy(cfg, "rural_highway", "clear", seed=0)
    assert p.breakpoint_distance_m == pytest.approx(100.0)
