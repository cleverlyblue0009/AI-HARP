"""Validation of the ITU-R models in :mod:`sim.itu` against the standards.

The P.838-3 test is the important one: it checks the reconstructed closed form
against the Recommendation's own published table, which is what establishes
that the coefficient signs are right. It is not a regression test against our
own output.
"""

from __future__ import annotations

import math

import pytest

from sim.itu import (
    FOG_LWC_THICK_G_M3,
    P838_TABLE5,
    fog_kl_db_per_km_per_g_m3,
    fog_specific_attenuation_db_per_km,
    rain_coefficients,
    rain_specific_attenuation_db_per_km,
    water_permittivity,
)

DSRC_GHZ = 5.9


# ------------------------------------------------- P.838-3 against Table 5 --
@pytest.mark.parametrize("row", P838_TABLE5, ids=[f"{r[0]}GHz" for r in P838_TABLE5])
def test_p838_closed_form_reproduces_published_table(row):
    """Equations (2)/(3) must reproduce Table 5 of the same Recommendation.

    This is what confirms the coefficient signs, which cannot be read out of
    the published PDF's text layer.
    """
    f, k_h, a_h, k_v, a_v = row
    got_kh, got_ah = rain_coefficients(f, "horizontal")
    got_kv, got_av = rain_coefficients(f, "vertical")
    assert got_kh == pytest.approx(k_h, rel=0.005)
    assert got_ah == pytest.approx(a_h, rel=0.005)
    assert got_kv == pytest.approx(k_v, rel=0.005)
    assert got_av == pytest.approx(a_v, rel=0.005)


def test_p838_worst_case_table_deviation_is_small():
    worst = 0.0
    for f, k_h, a_h, k_v, a_v in P838_TABLE5:
        kh, ah = rain_coefficients(f, "horizontal")
        kv, av = rain_coefficients(f, "vertical")
        for got, doc in ((kh, k_h), (ah, a_h), (kv, k_v), (av, a_v)):
            worst = max(worst, abs(got - doc) / doc)
    assert worst < 0.005, f"worst deviation {worst:.4%} -- coefficients suspect"


def test_rain_attenuation_follows_the_power_law():
    k, alpha = rain_coefficients(DSRC_GHZ)
    for r in (1.0, 5.0, 25.0, 100.0):
        assert rain_specific_attenuation_db_per_km(r, DSRC_GHZ) == pytest.approx(k * r**alpha)


def test_rain_attenuation_is_zero_without_rain():
    assert rain_specific_attenuation_db_per_km(0.0, DSRC_GHZ) == 0.0


def test_rain_attenuation_monotonic_in_rate():
    vals = [rain_specific_attenuation_db_per_km(r, DSRC_GHZ) for r in (0, 1, 5, 25, 50, 100)]
    assert vals == sorted(vals)


def test_horizontal_polarisation_attenuates_more_at_dsrc():
    """Sanity check on which polarisation is the conservative choice here."""
    kh, _ = rain_coefficients(DSRC_GHZ, "horizontal")
    kv, _ = rain_coefficients(DSRC_GHZ, "vertical")
    assert kh > kv


def test_p838_rejects_out_of_range_frequency():
    with pytest.raises(ValueError):
        rain_coefficients(0.5)


def test_p838_rejects_unknown_polarisation():
    with pytest.raises(ValueError):
        rain_coefficients(DSRC_GHZ, "circular")  # type: ignore[arg-type]


# --------------------------------------------------------------- P.840-8 --
def test_fog_permittivity_is_physical():
    """Water at 5.9 GHz: large real part, positive loss."""
    re, im = water_permittivity(DSRC_GHZ, 283.15)
    assert 60.0 < re < 90.0
    assert im > 0.0


def test_fog_kl_increases_with_frequency():
    """K_l rises steeply with frequency, which is why fog matters at mmWave
    and not at 5.9 GHz."""
    vals = [fog_kl_db_per_km_per_g_m3(f) for f in (1.0, 5.9, 30.0, 100.0)]
    assert vals == sorted(vals)
    assert vals[-1] > 100 * vals[1]


def test_fog_attenuation_is_linear_in_liquid_water():
    a = fog_specific_attenuation_db_per_km(0.25, DSRC_GHZ)
    b = fog_specific_attenuation_db_per_km(0.50, DSRC_GHZ)
    assert b == pytest.approx(2 * a)


def test_fog_attenuation_is_zero_without_fog():
    assert fog_specific_attenuation_db_per_km(0.0, DSRC_GHZ) == 0.0


# ------------------------------------------------- the headline physics ----
def test_heavy_rain_is_negligible_over_a_dsrc_hop():
    """The fact the paper must state.

    If this ever fails, either the model broke or someone changed the carrier
    frequency -- and the paper's weather section needs rewriting either way.
    """
    gamma = rain_specific_attenuation_db_per_km(25.0, DSRC_GHZ, "vertical")
    over_562m = gamma * 0.562
    assert gamma < 0.2, f"{gamma:.4f} dB/km is larger than P.838 predicts here"
    assert over_562m < 0.1, f"{over_562m:.4f} dB over a hop is not negligible"


def test_thick_fog_is_negligible_over_a_dsrc_hop():
    """Thick fog (0.5 g/m^3) gives ~0.012 dB/km at 5.9 GHz.

    Note this is ~14x the 0.0017 (dB/km)/(g/m^3) coefficient this project
    previously carried as a recalled value -- computing K_l from P.840-8's own
    equations rather than recalling it changed the number materially. It is
    still four orders of magnitude below anything that affects a link budget,
    so no conclusion moves; but it is why recalled constants were not shipped.
    """
    kl = fog_kl_db_per_km_per_g_m3(DSRC_GHZ)
    assert 0.02 < kl < 0.03, f"K_l = {kl:.5f} (dB/km)/(g/m^3)"
    gamma = fog_specific_attenuation_db_per_km(FOG_LWC_THICK_G_M3, DSRC_GHZ)
    assert gamma < 0.02
    assert gamma * 0.562 < 0.01   # over one DSRC hop


def test_even_violent_rain_cannot_close_a_link_budget():
    """50 mm/h over the full 10 km corridor, not just one hop."""
    gamma = rain_specific_attenuation_db_per_km(50.0, DSRC_GHZ, "horizontal")
    assert gamma * 10.0 < 5.0  # dB over 10 km


def test_mmwave_comparison_shows_the_frequency_dependence():
    """Context for the paper: the same rain at 60 GHz is a different world."""
    low = rain_specific_attenuation_db_per_km(25.0, DSRC_GHZ)
    high = rain_specific_attenuation_db_per_km(25.0, 60.0)
    assert high > 50 * low
