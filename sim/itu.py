"""ITU-R hydrometeor attenuation models, evaluated at the actual carrier.

Both recommendations are implemented from their own governing equations rather
than by reading the nearest row of a published table, so attenuation is
computed at exactly 5.9 GHz instead of being borrowed from the 6 GHz entry.

* :func:`rain_coefficients` -- ITU-R P.838-3 equations (2) and (3) with the
  coefficient Tables 1-4.
* :func:`fog_kl` -- ITU-R P.840-8 equations (2) to (11), the Rayleigh /
  double-Debye model for cloud and fog liquid water.

Provenance
----------
The P.838-3 coefficient signs cannot be read reliably out of the published PDF
(the text layer drops minus signs), so the reconstructed formula is validated
against Recommendation P.838-3's own Table 5, transcribed verbatim into
:data:`P838_TABLE5`. Agreement is within 0.12% over 1-10 GHz for all four
quantities, which confirms the sign pattern empirically rather than from
recollection. ``tests/test_itu.py`` runs that check on every test run.

What these models say about 5.9 GHz
-----------------------------------
Very little, and that is the point. Heavy rain at 25 mm/h gives roughly
0.07 dB/km with vertical polarisation, i.e. about 0.04 dB over a 562 m DSRC
hop; thick fog at 0.5 g/m^3 gives around 0.001 dB/km. Neither is capable of
degrading a link budget measurably. P.840-8 says as much in its own text:
fog attenuation becomes significant "at frequencies of the order of 100 GHz
and above".

Any V2X paper reporting multi-dB weather attenuation at 5.9 GHz from ITU-R
alone has made an arithmetic error. See ``configs/phy.yaml`` for how this
project handles that.
"""

from __future__ import annotations

import math
from typing import Literal

Polarisation = Literal["horizontal", "vertical"]

# ---------------------------------------------------------------------------
# ITU-R P.838-3 -- Specific attenuation model for rain
#
#   gamma_R = k * R^alpha                                        (1)
#   log10(k) = sum_j a_j exp(-((log10 f - b_j)/c_j)^2)
#              + m_k log10 f + c_k                               (2)
#   alpha    = sum_j a_j exp(-((log10 f - b_j)/c_j)^2)
#              + m_a log10 f + c_a                               (3)
# ---------------------------------------------------------------------------

#: Table 1 -- coefficients for k_H. ((a_j, b_j, c_j), ...), m_k, c_k
_P838_K_H = (
    ((-5.33980, -0.10008, 1.13098),
     (-0.35351, 1.26970, 0.45400),
     (-0.23789, 0.86036, 0.15354),
     (-0.94158, 0.64552, 0.16817)),
    -0.18961, 0.71147,
)
#: Table 2 -- coefficients for k_V.
_P838_K_V = (
    ((-3.80595, 0.56934, 0.81061),
     (-3.44965, -0.22911, 0.51059),
     (-0.39902, 0.73042, 0.11899),
     (0.50167, 1.07319, 0.27195)),
    -0.16398, 0.63297,
)
#: Table 3 -- coefficients for alpha_H.
_P838_A_H = (
    ((-0.14318, 1.82442, -0.55187),
     (0.29591, 0.77564, 0.19822),
     (0.32177, 0.63773, 0.13164),
     (-5.37610, -0.96230, 1.47828),
     (16.1721, -3.29980, 3.43990)),
    0.67849, -1.95537,
)
#: Table 4 -- coefficients for alpha_V.
_P838_A_V = (
    ((-0.07771, 2.33840, -0.76284),
     (0.56727, 0.95545, 0.54039),
     (-0.20238, 1.14520, 0.26809),
     (-48.2991, 0.791669, 0.116226),
     (48.5833, 0.791459, 0.116479)),
    -0.053739, 0.83433,
)

#: Table 5, transcribed verbatim: (f_GHz, k_H, alpha_H, k_V, alpha_V).
#: Used only to validate the closed form above -- never read at runtime.
P838_TABLE5: tuple[tuple[float, float, float, float, float], ...] = (
    (1.0, 0.0000259, 0.9691, 0.0000308, 0.8592),
    (1.5, 0.0000443, 1.0185, 0.0000574, 0.8957),
    (2.0, 0.0000847, 1.0664, 0.0000998, 0.9490),
    (2.5, 0.0001321, 1.1209, 0.0001464, 1.0085),
    (3.0, 0.0001390, 1.2322, 0.0001942, 1.0688),
    (3.5, 0.0001155, 1.4189, 0.0002346, 1.1387),
    (4.0, 0.0001071, 1.6009, 0.0002461, 1.2476),
    (4.5, 0.0001340, 1.6948, 0.0002347, 1.3987),
    (5.0, 0.0002162, 1.6969, 0.0002428, 1.5317),
    (5.5, 0.0003909, 1.6499, 0.0003115, 1.5882),
    (6.0, 0.0007056, 1.5900, 0.0004878, 1.5728),
    (7.0, 0.001915, 1.4810, 0.001425, 1.4745),
    (8.0, 0.004115, 1.3905, 0.003450, 1.3797),
    (9.0, 0.007535, 1.3155, 0.006691, 1.2895),
    (10.0, 0.01217, 1.2571, 0.01129, 1.2156),
)

P838_VALID_RANGE_GHZ = (1.0, 1000.0)


def _p838_sum(spec, f_ghz: float) -> float:
    terms, m, c = spec
    lf = math.log10(f_ghz)
    return sum(a * math.exp(-(((lf - b) / cj) ** 2)) for a, b, cj in terms) + m * lf + c


def rain_coefficients(f_ghz: float, polarisation: Polarisation = "vertical") -> tuple[float, float]:
    """``(k, alpha)`` for the P.838 power law at ``f_ghz``.

    Vertical polarisation is the default because vehicular DSRC antennas are
    roof-mounted vertical monopoles. Horizontal gives a ~1.4x larger ``k`` at
    5.9 GHz; both are negligible in absolute terms.
    """
    lo, hi = P838_VALID_RANGE_GHZ
    if not lo <= f_ghz <= hi:
        raise ValueError(f"P.838-3 is valid over {lo}-{hi} GHz; got {f_ghz}")
    if polarisation == "vertical":
        return 10.0 ** _p838_sum(_P838_K_V, f_ghz), _p838_sum(_P838_A_V, f_ghz)
    if polarisation == "horizontal":
        return 10.0 ** _p838_sum(_P838_K_H, f_ghz), _p838_sum(_P838_A_H, f_ghz)
    raise ValueError(f"polarisation must be 'vertical' or 'horizontal', got {polarisation!r}")


def rain_specific_attenuation_db_per_km(
    rain_rate_mm_h: float, f_ghz: float, polarisation: Polarisation = "vertical"
) -> float:
    """ITU-R P.838-3 equation (1): ``gamma_R = k * R^alpha`` in dB/km."""
    if rain_rate_mm_h <= 0.0:
        return 0.0
    k, alpha = rain_coefficients(f_ghz, polarisation)
    return k * rain_rate_mm_h**alpha


# ---------------------------------------------------------------------------
# ITU-R P.840-8 -- Attenuation due to clouds and fog
#
#   gamma_c = K_l(f, T) * M                                      (1)
#   K_l     = 0.819 f / (eps'' (1 + eta^2))                      (2)
#   eta     = (2 + eps') / eps''                                 (3)
# with the double-Debye permittivity of water, equations (4)-(11).
# ---------------------------------------------------------------------------
P840_VALID_RANGE_GHZ = (0.0, 200.0)


def water_permittivity(f_ghz: float, temperature_k: float) -> tuple[float, float]:
    """Double-Debye complex permittivity of water, P.840-8 eqs (4)-(11).

    Returns ``(eps_real, eps_imag)``.
    """
    theta = 300.0 / temperature_k                       # (9)
    eps0 = 77.66 + 103.3 * (theta - 1.0)                # (6)
    eps1 = 0.0671 * eps0                                # (7)
    eps2 = 3.52                                         # (8)
    fp = 20.20 - 146.0 * (theta - 1.0) + 316.0 * (theta - 1.0) ** 2   # (10) GHz
    fs = 39.8 * fp                                      # (11) GHz

    rp = f_ghz / fp
    rs = f_ghz / fs
    eps_im = (f_ghz * (eps0 - eps1)) / (fp * (1.0 + rp**2)) + \
             (f_ghz * (eps1 - eps2)) / (fs * (1.0 + rs**2))           # (5)
    eps_re = (eps0 - eps1) / (1.0 + rp**2) + (eps1 - eps2) / (1.0 + rs**2) + eps2  # (4)
    return eps_re, eps_im


def fog_kl_db_per_km_per_g_m3(f_ghz: float, temperature_k: float = 283.15) -> float:
    """Cloud/fog specific attenuation coefficient ``K_l``, P.840-8 eq (2).

    The default temperature is 10 C, a representative liquid-water temperature
    for radiation fog on a road surface.
    """
    lo, hi = P840_VALID_RANGE_GHZ
    if not lo < f_ghz <= hi:
        raise ValueError(f"P.840-8 Rayleigh model is valid to {hi} GHz; got {f_ghz}")
    eps_re, eps_im = water_permittivity(f_ghz, temperature_k)
    eta = (2.0 + eps_re) / eps_im                       # (3)
    return 0.819 * f_ghz / (eps_im * (1.0 + eta**2))    # (2)


def fog_specific_attenuation_db_per_km(
    liquid_water_g_m3: float, f_ghz: float, temperature_k: float = 283.15
) -> float:
    """ITU-R P.840-8 equation (1): ``gamma_c = K_l * M`` in dB/km."""
    if liquid_water_g_m3 <= 0.0:
        return 0.0
    return fog_kl_db_per_km_per_g_m3(f_ghz, temperature_k) * liquid_water_g_m3


#: P.840-8 Section 1: representative fog liquid water densities.
FOG_LWC_MEDIUM_G_M3 = 0.05   # visibility of the order of 300 m
FOG_LWC_THICK_G_M3 = 0.5     # visibility of the order of 50 m
