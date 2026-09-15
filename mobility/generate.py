"""Trace acquisition: pick a backend, generate once, cache forever.

``get_trace`` is the only mobility entry point the rest of the project uses.
It resolves the backend (SUMO if installed, pure-Python otherwise), logs that
choice as a banner so it can never be missed, and caches the result to
``cache/traces/*.npz`` keyed by a hash of everything that affects the trace.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

from common.config import PROJECT_ROOT, config_hash, ensure_dir, load_yaml
from common.logging_utils import get_logger, log_banner
from common.seeding import make_rng
from mobility.fallback import generate_fallback_trace
from mobility.trace import CorruptTraceError, Trace, load_trace, save_trace

logger = get_logger("mobility.generate")

TRACE_CACHE = PROJECT_ROOT / "cache" / "traces"

_SCENARIO_FILES = {
    "rural_highway": "scenario_rural.yaml",
    "urban_grid": "scenario_urban.yaml",
    "urban_nlos": "scenario_urban_nlos.yaml",
    # Real OpenStreetMap networks (SUMO backend only).
    "rural_highway_osm": "scenario_rural_osm.yaml",
    "urban_grid_osm": "scenario_urban_osm.yaml",
}


def _deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in over.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out

_BANNER_SHOWN: set[str] = set()


class MobilityBackend(str, Enum):
    SUMO = "sumo"
    FALLBACK = "fallback"
    AUTO = "auto"


def load_scenario(name: str) -> dict[str, Any]:
    """Load a scenario config by short name (``rural_highway``/``urban_grid``).

    A file may declare ``extends: <other scenario file or name>``; it is then
    deep-merged over that base, so a variant states only what differs.
    """
    cfg = dict(load_yaml(_SCENARIO_FILES.get(name, name)))
    base = cfg.pop("extends", None)
    return _deep_merge(load_scenario(base), cfg) if base else cfg


def weather_mobility_factors(weather: str, phy_cfg: dict[str, Any] | None = None) -> tuple[float, float]:
    """(speed_factor, headway_factor) for a weather condition.

    These are the *behavioural* effects of weather -- drivers slow down and
    leave bigger gaps. They are separate from the radio attenuation terms in
    :mod:`sim.phy`, and they matter: in heavy rain the network topology changes
    because the traffic stream changes, not only because the channel does.
    """
    cfg = phy_cfg if phy_cfg is not None else load_yaml("phy.yaml")
    try:
        w = cfg["weather"]["conditions"][weather]
    except KeyError as exc:
        known = list(cfg["weather"]["conditions"])
        raise KeyError(f"Unknown weather {weather!r}; known: {known}") from exc
    return float(w.get("speed_factor", 1.0)), float(w.get("headway_factor", 1.0))


def resolve_backend(requested: str | MobilityBackend = MobilityBackend.AUTO):
    """Resolve the mobility backend and announce it loudly.

    Returns ``(backend, tools)`` where ``tools`` is a ``SumoTools`` or None.
    """
    requested = MobilityBackend(requested)
    from mobility.sumo_runner import find_sumo  # local: avoids cost when unused

    tools = find_sumo()

    if requested is MobilityBackend.SUMO:
        if tools is None:
            raise RuntimeError(
                "backend='sumo' was requested but no SUMO installation was found. "
                "Install SUMO and set SUMO_HOME, or use backend='auto'/'fallback'."
            )
        backend = MobilityBackend.SUMO
    elif requested is MobilityBackend.FALLBACK:
        backend = MobilityBackend.FALLBACK
    else:
        backend = MobilityBackend.SUMO if tools is not None else MobilityBackend.FALLBACK

    key = backend.value
    if key not in _BANNER_SHOWN:
        _BANNER_SHOWN.add(key)
        if backend is MobilityBackend.SUMO:
            log_banner(logger, "MOBILITY BACKEND: SUMO", [
                f"binary      : {tools.sumo}",
                f"SUMO_HOME   : {tools.sumo_home or '(not set; found on PATH)'}",
                "car-following: Krauss (SUMO default)",
            ])
        else:
            log_banner(logger, "MOBILITY BACKEND: PURE-PYTHON FALLBACK", [
                "No SUMO installation detected (no SUMO_HOME, nothing on PATH).",
                "Mobility is the built-in Krauss microscopic model:",
                "  - real car-following + dawdling + mixed car/truck fleet",
                "  - NO lane changing, NO OSM geometry, NO junction gap acceptance",
                "Results MUST be reported as backend=fallback, not as SUMO results.",
            ])
    return backend, tools


#: Bumped whenever SUMO trace construction changes, so a SUMO trace built by an
#: older pipeline is never served from the cache. The key covers configuration,
#: not code: after grids gained edges and randomTrips demand, the cached broken
#: urban_nlos trace (0.25 veh/km/lane) was still returned for the new code.
#: 2: grid edges + real dimensions, randomTrips grid demand, OSM projection,
#:    FCD recorded from the warm-up.
#: Fallback keys deliberately omit it, so every committed fallback trace (and
#: results/runs.csv) keeps its key.
SUMO_PIPELINE_VERSION = 2


def trace_cache_key(
    scenario: dict[str, Any], density: float, seed: int, backend: str,
    speed_factor: float, headway_factor: float, duration_s: float | None,
) -> str:
    payload = {
        "scenario": scenario,
        "density": density,
        "seed": seed,
        "backend": backend,
        "speed_factor": speed_factor,
        "headway_factor": headway_factor,
        "duration_s": duration_s,
    }
    if backend == MobilityBackend.SUMO.value:
        payload["sumo_pipeline"] = SUMO_PIPELINE_VERSION
    return config_hash(payload)


def get_trace(
    scenario: str | dict[str, Any],
    density_veh_km_lane: float,
    seed: int,
    *,
    weather: str = "clear",
    backend: str | MobilityBackend = MobilityBackend.AUTO,
    duration_s: float | None = None,
    use_cache: bool = True,
    phy_cfg: dict[str, Any] | None = None,
) -> Trace:
    """Return a mobility trace, generating and caching it if necessary."""
    scfg = load_scenario(scenario) if isinstance(scenario, str) else scenario
    speed_factor, headway_factor = weather_mobility_factors(weather, phy_cfg)
    resolved, tools = resolve_backend(backend)
    if scfg.get("sumo", {}).get("osm_extract") and resolved is not MobilityBackend.SUMO:
        raise RuntimeError(
            f"Scenario {scfg['name']!r} is a real OSM network and needs the SUMO backend "
            "(set SUMO_HOME); the pure-Python fallback can only build synthetic geometry."
        )

    key = trace_cache_key(scfg, density_veh_km_lane, seed, resolved.value,
                          speed_factor, headway_factor, duration_s)
    cache_path = TRACE_CACHE / f"{scfg['name']}_d{density_veh_km_lane:g}_s{seed}_{key}.npz"

    if use_cache and cache_path.exists():
        try:
            trace = load_trace(cache_path)
        except CorruptTraceError as exc:
            # Generation is deterministic in (config, density, seed), so
            # regenerating reproduces the file that should have been there.
            logger.warning("Trace cache CORRUPT %s (%s); deleting and regenerating",
                           cache_path.name, exc)
            cache_path.unlink(missing_ok=True)
        else:
            logger.info("Trace cache HIT  %s | %s", cache_path.name, _fmt(trace.summary()))
            return trace

    logger.info(
        "Trace cache MISS -> generating (%s, density=%g veh/km/lane, weather=%s, seed=%d)",
        scfg["name"], density_veh_km_lane, weather, seed,
    )
    if resolved is MobilityBackend.SUMO:
        from mobility.sumo_runner import generate_sumo_trace

        trace = generate_sumo_trace(
            scfg, density_veh_km_lane, seed, tools=tools, speed_factor=speed_factor
        )
    else:
        rng = make_rng(seed, "mobility")
        trace = generate_fallback_trace(
            scfg, density_veh_km_lane, rng,
            duration_s=duration_s, speed_factor=speed_factor, headway_factor=headway_factor,
        )
    trace.meta.setdefault("weather", weather)
    trace.meta.setdefault("seed", seed)

    if use_cache:
        ensure_dir(TRACE_CACHE)
        save_trace(trace, cache_path)
    logger.info("Trace ready: %s", _fmt(trace.summary()))
    return trace


def _fmt(summary: dict[str, Any]) -> str:
    return " ".join(f"{k}={v}" for k, v in summary.items() if k not in {"scenario", "backend"})
