"""IEEE 802.11p / DSRC physical layer at 5.9 GHz.

Link budget
-----------
::

    P_rx(d) = P_tx + G_tx + G_rx - PL(d) - L_weather(d) - S(link) + F(packet)

* ``PL(d)``       log-distance path loss with a scenario-dependent exponent
* ``L_weather``   weather attenuation, split into a physical ITU-R hydrometeor
                  term and an empirical road-environment excess term (see the
                  long comment in ``configs/phy.yaml`` -- these must not be
                  conflated in the paper)
* ``S(link)``     log-normal shadowing, spatially correlated per link
* ``F(packet)``   Nakagami-m fast fading, redrawn per transmitted frame

Every constant comes from ``configs/phy.yaml``. Nothing here is hardcoded.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from common.logging_utils import get_logger
from sim.buildings import BuildingGrid, build_building_grid

logger = get_logger("sim.phy")

SPEED_OF_LIGHT = 299_792_458.0  # m/s, exact by SI definition


def free_space_loss_db(distance_m: float, frequency_hz: float) -> float:
    """Friis free-space path loss in dB: ``20*log10(4*pi*d*f/c)``."""
    if distance_m <= 0:
        raise ValueError("distance must be positive")
    return 20.0 * math.log10(4.0 * math.pi * distance_m * frequency_hz / SPEED_OF_LIGHT)


def two_ray_breakpoint_m(
    tx_height_m: float, rx_height_m: float, frequency_hz: float
) -> float:
    """Two-ray ground-reflection breakpoint, ``d_bp = 4 h_t h_r / lambda``.

    Beyond this distance the ground-reflected ray arrives in antiphase with the
    direct ray and the received power decays as roughly ``d^-4`` instead of
    ``d^-2``. Deriving the breakpoint from antenna geometry rather than reading
    a fitted constant ties it to a quantity the paper states anyway, and
    removes a recalled measurement value from the configuration.

    With 1.5 m antennas at 5.9 GHz this gives 177 m.
    """
    if min(tx_height_m, rx_height_m) <= 0:
        raise ValueError("antenna heights must be positive")
    wavelength = SPEED_OF_LIGHT / frequency_hz
    return 4.0 * tx_height_m * rx_height_m / wavelength


# The ITU-R hydrometeor models live in sim/itu.py and are evaluated at the
# actual carrier frequency from the Recommendations' own equations. They are
# re-exported here so existing callers keep working.
from sim.itu import (  # noqa: E402
    fog_specific_attenuation_db_per_km,
    rain_specific_attenuation_db_per_km,
)


@dataclass
class WeatherEffect:
    """Resolved per-condition weather terms, kept separable for reporting."""

    name: str
    rain_rate_mm_h: float
    fog_liquid_water_g_m3: float
    itu_rain_db_per_km: float
    itu_fog_db_per_km: float
    excess_db_per_km: float
    exponent_delta: float
    visibility_m: float
    speed_factor: float
    headway_factor: float

    @property
    def itu_total_db_per_km(self) -> float:
        """Physically-grounded hydrometeor attenuation only."""
        return self.itu_rain_db_per_km + self.itu_fog_db_per_km

    @property
    def total_db_per_km(self) -> float:
        return self.itu_total_db_per_km + self.excess_db_per_km

    def summary(self) -> str:
        return (
            f"{self.name}: ITU-R {self.itu_total_db_per_km:.4f} dB/km "
            f"(rain {self.itu_rain_db_per_km:.4f} + fog {self.itu_fog_db_per_km:.4f}) "
            f"| empirical excess {self.excess_db_per_km:.2f} dB/km "
            f"| path-loss exponent +{self.exponent_delta:.2f} "
            f"| driver speed x{self.speed_factor:.2f}"
        )


@dataclass
class PhyModel:
    """The resolved physical layer for one (scenario, weather) pair."""

    frequency_hz: float
    bandwidth_hz: float
    data_rate_bps: float
    tx_power_dbm: float
    tx_gain_dbi: float
    rx_gain_dbi: float
    noise_figure_db: float
    thermal_noise_density_dbm_hz: float
    sinr_threshold_db: float
    sensitivity_dbm: float
    capture_enabled: bool
    reference_distance_m: float
    reference_loss_db: float
    path_loss_exponent: float          # n1, below the breakpoint
    path_loss_exponent_far: float      # n2, above the breakpoint
    breakpoint_distance_m: float
    shadowing_std_db: float
    shadowing_decorrelation_m: float
    nakagami_m: float
    resample_fading_per_packet: bool
    weather: WeatherEffect
    scenario: str
    seed: int = 0
    #: Building geometry for NLOS classification. None for open-road scenarios.
    buildings: "BuildingGrid | None" = None
    nlos_corner_loss_db: float = 0.0
    _shadow_salt: int = field(default=0, repr=False)

    # ------------------------------------------------------------------ noise --
    @property
    def noise_dbm(self) -> float:
        """Thermal noise power in the channel bandwidth, including NF."""
        return (
            self.thermal_noise_density_dbm_hz
            + 10.0 * math.log10(self.bandwidth_hz)
            + self.noise_figure_db
        )

    @property
    def noise_mw(self) -> float:
        return 10.0 ** (self.noise_dbm / 10.0)

    @property
    def eirp_dbm(self) -> float:
        return self.tx_power_dbm + self.tx_gain_dbi

    # -------------------------------------------------------------- path loss --
    def path_loss_db(self, distance_m: np.ndarray | float) -> np.ndarray:
        """Dual-slope log-distance path loss, clamped at the reference distance.

        Continuous at the breakpoint by construction: the far branch starts
        from the near branch's value at ``d_bp``.
        """
        d = np.maximum(np.asarray(distance_m, dtype=float), self.reference_distance_m)
        d_bp = self.breakpoint_distance_m
        near = 10.0 * self.path_loss_exponent * np.log10(d / self.reference_distance_m)
        loss_at_bp = 10.0 * self.path_loss_exponent * np.log10(d_bp / self.reference_distance_m)
        far = loss_at_bp + 10.0 * self.path_loss_exponent_far * np.log10(np.maximum(d, d_bp) / d_bp)
        return self.reference_loss_db + np.where(d <= d_bp, near, far)

    def weather_loss_db(self, distance_m: np.ndarray | float) -> np.ndarray:
        """Total weather attenuation over the link, in dB."""
        d_km = np.asarray(distance_m, dtype=float) / 1000.0
        return d_km * self.weather.total_db_per_km

    # ------------------------------------------------------------- shadowing --
    def shadowing_db(self, tx: np.ndarray, rx: np.ndarray, distance_m: np.ndarray) -> np.ndarray:
        """Spatially correlated log-normal shadowing, deterministic per link.

        Rather than carrying per-pair state, the shadowing value is a pure
        function of ``(tx, rx, floor(d / decorrelation_m), seed)``. The link
        therefore keeps one shadowing realisation until the pair's separation
        has changed by a decorrelation length, at which point it redraws --
        which is the behaviour the Gudmundson correlation model prescribes,
        and it is exactly reproducible under a fixed seed.
        """
        if self.shadowing_std_db <= 0:
            return np.zeros_like(np.asarray(distance_m, dtype=float))
        a = np.minimum(tx, rx).astype(np.uint64)
        b = np.maximum(tx, rx).astype(np.uint64)
        bucket = (np.asarray(distance_m) / self.shadowing_decorrelation_m).astype(np.int64)
        u = _hash_uniform(a, b, bucket.astype(np.uint64), np.uint64(self._shadow_salt))
        return self.shadowing_std_db * _ndtri(u)

    # ----------------------------------------------------------------- fading --
    def fading_db(self, size: int | tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
        """Nakagami-m fading gain in dB.

        The received *power* gain of a Nakagami-m channel is Gamma distributed
        with shape ``m`` and unit mean, i.e. ``Gamma(m, 1/m)``.
        """
        gain = rng.gamma(shape=self.nakagami_m, scale=1.0 / self.nakagami_m, size=size)
        return 10.0 * np.log10(np.maximum(gain, 1e-12))

    # ------------------------------------------------------------ link budget --
    def median_rx_power_dbm(self, distance_m: np.ndarray | float) -> np.ndarray:
        """Received power with no fading and no shadowing (the median link)."""
        return (
            self.tx_power_dbm + self.tx_gain_dbi + self.rx_gain_dbi
            - self.path_loss_db(distance_m) - self.weather_loss_db(distance_m)
        )

    def rx_power_dbm(
        self, tx: np.ndarray, rx: np.ndarray, distance_m: np.ndarray,
        rng: np.random.Generator, with_fading: bool = True,
    ) -> np.ndarray:
        """Full stochastic received power for a set of links."""
        p = self.median_rx_power_dbm(distance_m) - self.shadowing_db(tx, rx, distance_m)
        if with_fading:
            p = p + self.fading_db(np.shape(distance_m), rng)
        return p

    # -------------------------------------------------------------- buildings --
    @property
    def has_buildings(self) -> bool:
        return self.buildings is not None and self.nlos_corner_loss_db > 0.0

    def nlos_excess_db(self, pos_tx: np.ndarray, pos_rx: np.ndarray) -> np.ndarray:
        """``[n_tx, n_rx]`` additional loss for links blocked by a building.

        Zero everywhere when the scenario has no buildings, so open-road
        scenarios are completely unaffected. For blocked links the signal must
        travel the Manhattan path around the corner and pay a knife-edge
        diffraction loss; charging it as an *excess over the LOS model* leaves
        path loss, shadowing and fading untouched.
        """
        shape = (pos_tx.shape[0], pos_rx.shape[0])
        if not self.has_buildings:
            return np.zeros(shape)
        b = self.buildings
        assert b is not None
        nlos = b.is_nlos(pos_tx, pos_rx)
        if not nlos.any():
            return np.zeros(shape)
        d_e = np.maximum(b.euclidean_distance(pos_tx, pos_rx), self.reference_distance_m)
        d_m = np.maximum(b.manhattan_distance(pos_tx, pos_rx), self.reference_distance_m)
        detour = self.path_loss_db(d_m) - self.path_loss_db(d_e)
        return np.where(nlos, detour + self.nlos_corner_loss_db, 0.0)

    def nlos_range_m(self) -> float:
        """Median-link range for a *blocked* link, worst case (90-degree corner).

        Reported alongside ``nominal_range_m`` because the gap between them is
        what creates urban fragmentation.
        """
        if not self.has_buildings:
            return self.nominal_range_m
        # A symmetric corner: Manhattan path is sqrt(2) times the Euclidean.
        lo, hi = self.reference_distance_m, 1e5

        def f(d: float) -> float:
            detour = float(self.path_loss_db(d * math.sqrt(2)) - self.path_loss_db(d))
            return float(self.median_rx_power_dbm(d)) - detour - self.nlos_corner_loss_db \
                - self.sensitivity_dbm

        if f(hi) > 0:
            return hi
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if f(mid) > 0:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    # ------------------------------------------------------------------ range --
    def range_for_power_m(self, target_dbm: float) -> float:
        """Distance at which the *median* link budget falls to ``target_dbm``.

        Inverts the log-distance model analytically. Used for the nominal
        communication range (at receiver sensitivity), for the carrier-sense
        neighbourhood, and for pruning the interferer search.
        """
        # P = EIRP + Grx - PL0 - 10 n log10(d/d0) - (w/1000) * d  -- the weather
        # term is linear in d, so solve by bisection over a generous bracket.
        lo, hi = self.reference_distance_m, 1e5
        f = lambda d: float(self.median_rx_power_dbm(d)) - target_dbm  # noqa: E731
        if f(hi) > 0:
            return hi
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if f(mid) > 0:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    @property
    def nominal_range_m(self) -> float:
        """Median-link range at receiver sensitivity."""
        return self.range_for_power_m(self.sensitivity_dbm)

    @property
    def sinr_limited_range_m(self) -> float:
        """Median-link range where SINR (noise only) hits the decode threshold."""
        return self.range_for_power_m(self.noise_dbm + self.sinr_threshold_db)

    def frame_duration_s(self, frame_bytes: int) -> float:
        """Airtime for a frame, including the 802.11p PHY preamble + signal field."""
        preamble_s = 40e-6  # [STD] 32 us preamble + 8 us SIGNAL at 10 MHz
        return preamble_s + (frame_bytes * 8.0) / self.data_rate_bps

    def summary(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "weather": self.weather.name,
            "eirp_dbm": round(self.eirp_dbm, 1),
            "noise_dbm": round(self.noise_dbm, 1),
            "path_loss_n1": round(self.path_loss_exponent, 2),
            "path_loss_n2": round(self.path_loss_exponent_far, 2),
            "breakpoint_m": self.breakpoint_distance_m,
            "nakagami_m": self.nakagami_m,
            "shadowing_std_db": self.shadowing_std_db,
            "weather_itu_db_per_km": round(self.weather.itu_total_db_per_km, 4),
            "weather_excess_db_per_km": round(self.weather.excess_db_per_km, 2),
            "nominal_range_m": round(self.nominal_range_m, 1),
            "sinr_limited_range_m": round(self.sinr_limited_range_m, 1),
            "nlos_range_m": round(self.nlos_range_m(), 1) if self.has_buildings else None,
            "nlos_corner_loss_db": self.nlos_corner_loss_db if self.has_buildings else None,
        }


# ---------------------------------------------------------------------------
# Deterministic hashing helpers for spatially correlated shadowing.
# ---------------------------------------------------------------------------
def _splitmix64(x: np.ndarray) -> np.ndarray:
    """SplitMix64 finaliser: a fast, well-mixed integer hash."""
    with np.errstate(over="ignore"):
        x = (x + np.uint64(0x9E3779B97F4A7C15)).astype(np.uint64)
        z = x
        z = ((z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)).astype(np.uint64)
        z = ((z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)).astype(np.uint64)
        return (z ^ (z >> np.uint64(31))).astype(np.uint64)


def _hash_uniform(*keys: np.ndarray) -> np.ndarray:
    """Deterministic uniform(0,1) draw from a tuple of integer keys."""
    with np.errstate(over="ignore"):
        h = np.zeros(np.broadcast(*keys).shape, dtype=np.uint64)
        for k in keys:
            h = _splitmix64(h ^ np.asarray(k, dtype=np.uint64) * np.uint64(0x27220A95))
    # Map to (0,1) exclusive; 2**53 keeps full float64 precision.
    u = (h >> np.uint64(11)).astype(np.float64) / float(1 << 53)
    return np.clip(u, 1e-12, 1.0 - 1e-12)


def _ndtri(u: np.ndarray) -> np.ndarray:
    """Inverse standard normal CDF (SciPy when present, Acklam's rational fit otherwise)."""
    try:
        from scipy.special import ndtri  # type: ignore

        return ndtri(u)
    except ImportError:  # pragma: no cover
        # Acklam's algorithm; |error| < 1.15e-9 over the whole range.
        a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
             1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
        b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
             6.680131188771972e01, -1.328068155288572e01]
        c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
             -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
        d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
             3.754408661907416e00]
        plow, phigh = 0.02425, 1 - 0.02425
        u = np.asarray(u, dtype=float)
        out = np.empty_like(u)
        lo, hi = u < plow, u > phigh
        mid = ~(lo | hi)
        q = np.sqrt(-2 * np.log(np.where(lo, u, plow)))
        out[lo] = (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5])[lo] / \
                  ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)[lo]
        q = np.sqrt(-2 * np.log(np.where(hi, 1 - u, plow)))
        out[hi] = -((((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5])[hi] /
                    ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)[hi])
        q = np.where(mid, u, 0.5) - 0.5
        r = q * q
        out[mid] = ((((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q /
                    (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1))[mid]
        return out


# ---------------------------------------------------------------------------
# Construction from YAML
# ---------------------------------------------------------------------------
def resolve_weather(
    phy_cfg: dict[str, Any], weather: str, frequency_hz: float | None = None
) -> WeatherEffect:
    """Resolve a weather condition into its separable attenuation terms.

    Hydrometeor attenuation is computed at ``frequency_hz`` from ITU-R
    P.838-3 / P.840-8 rather than read from config, so there are no k/alpha/K_l
    constants that can drift from the standard.
    """
    w = phy_cfg["weather"]
    try:
        c = w["conditions"][weather]
    except KeyError as exc:
        raise KeyError(f"Unknown weather {weather!r}; known: {list(w['conditions'])}") from exc

    f_ghz = (frequency_hz if frequency_hz is not None
             else float(phy_cfg["radio"]["carrier_frequency_hz"])) / 1e9
    polarisation = w.get("itu_rain_polarisation", "vertical")
    fog_temp_k = float(w.get("fog_temperature_k", 283.15))

    # The empirical excess-loss term is disabled by default: the values that
    # were there were not citable. See the block comment in configs/phy.yaml.
    excess_enabled = bool(w.get("enable_empirical_excess_loss", False))
    excess = float(c.get("excess_loss_db_per_km", 0.0)) if excess_enabled else 0.0
    dn = float(c.get("exponent_delta", 0.0)) if excess_enabled else 0.0

    return WeatherEffect(
        name=weather,
        rain_rate_mm_h=float(c["rain_rate_mm_h"]),
        fog_liquid_water_g_m3=float(c["fog_liquid_water_g_m3"]),
        itu_rain_db_per_km=rain_specific_attenuation_db_per_km(
            float(c["rain_rate_mm_h"]), f_ghz, polarisation
        ),
        itu_fog_db_per_km=fog_specific_attenuation_db_per_km(
            float(c["fog_liquid_water_g_m3"]), f_ghz, fog_temp_k
        ),
        excess_db_per_km=excess,
        exponent_delta=dn,
        visibility_m=float(c["visibility_m"]),
        speed_factor=float(c.get("speed_factor", 1.0)),
        headway_factor=float(c.get("headway_factor", 1.0)),
    )


def build_phy(
    phy_cfg: dict[str, Any],
    scenario: str,
    weather: str = "clear",
    seed: int = 0,
    trace_meta: dict[str, Any] | None = None,
) -> PhyModel:
    """Resolve ``configs/phy.yaml`` into a :class:`PhyModel`."""
    r, rx, pl, fa = phy_cfg["radio"], phy_cfg["receiver"], phy_cfg["path_loss"], phy_cfg["fading"]
    weff = resolve_weather(phy_cfg, weather, float(r["carrier_frequency_hz"]))

    d0 = float(pl["reference_distance_m"])
    ref_loss = pl.get("reference_loss_db")
    ref_loss = free_space_loss_db(d0, float(r["carrier_frequency_hz"])) if ref_loss is None \
        else float(ref_loss)

    if scenario not in pl["exponent"]:
        raise KeyError(f"No path-loss exponent for scenario {scenario!r} in configs/phy.yaml")

    # Breakpoint: derived from the two-ray geometry unless explicitly overridden.
    bp_cfg = pl.get("breakpoint_distance_m")
    if isinstance(bp_cfg, dict) and scenario in bp_cfg and bp_cfg[scenario] is not None:
        breakpoint_m = float(bp_cfg[scenario])
    else:
        h = float(pl["antenna_height_m"][scenario])
        breakpoint_m = two_ray_breakpoint_m(h, h, float(r["carrier_frequency_hz"]))

    # Buildings: only for scenarios that declare an NLOS model.
    nlos_cfg = phy_cfg.get("nlos", {})
    corner_loss = float(nlos_cfg.get("corner_loss_db", {}).get(scenario, 0.0))
    buildings = None
    if corner_loss > 0.0 and trace_meta is not None:
        buildings = build_building_grid(
            trace_meta, float(nlos_cfg.get("street_half_width_m", 6.0))
        )

    model = PhyModel(
        frequency_hz=float(r["carrier_frequency_hz"]),
        bandwidth_hz=float(r["bandwidth_hz"]),
        data_rate_bps=float(r["data_rate_mbps"]) * 1e6,
        tx_power_dbm=float(r["tx_power_dbm"]),
        tx_gain_dbi=float(r["tx_antenna_gain_dbi"]),
        rx_gain_dbi=float(r["rx_antenna_gain_dbi"]),
        noise_figure_db=float(r["noise_figure_db"]),
        thermal_noise_density_dbm_hz=float(r["thermal_noise_density_dbm_hz"]),
        sinr_threshold_db=float(rx["sinr_threshold_db"]),
        sensitivity_dbm=float(rx["sensitivity_dbm"]),
        capture_enabled=bool(rx["capture_enabled"]),
        reference_distance_m=d0,
        reference_loss_db=ref_loss,
        # The weather exponent delta applies to the far slope only: the
        # sub-breakpoint regime is near-field LOS, not ground-reflection
        # dominated, so a wet surface has no mechanism to steepen it.
        path_loss_exponent=float(pl["exponent"][scenario]),
        path_loss_exponent_far=float(pl["exponent_far"][scenario]) + weff.exponent_delta,
        breakpoint_distance_m=breakpoint_m,
        shadowing_std_db=float(pl["shadowing_std_db"][scenario]),
        shadowing_decorrelation_m=float(pl["shadowing_decorrelation_m"]),
        nakagami_m=float(fa["m"][scenario]),
        resample_fading_per_packet=bool(fa["resample_per_packet"]),
        weather=weff,
        scenario=scenario,
        seed=seed,
        buildings=buildings,
        nlos_corner_loss_db=corner_loss,
        _shadow_salt=int(seed) & 0xFFFFFFFF,
    )
    logger.debug("PHY: %s", model.summary())
    return model
