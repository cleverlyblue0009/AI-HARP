"""The hazard object.

A hazard is **not a point**. Treating it as one is what makes conventional
dissemination schemes broadcast to the wrong vehicles: a point hazard has no
direction, no extent and no lifetime, so every vehicle within radius R looks
equally at risk. Here a hazard carries:

* ``htype``            - what it is
* ``severity``         - how bad, in [0, 1]
* ``[span_start_m, span_end_m]`` - the stretch of road it occupies
* ``onset_time_s`` + a decay law - when it starts and whether it clears
* ``affected_direction`` - which carriageway it actually threatens

Those five properties are what the risk field in :mod:`hazard.risk_field`
turns into a per-vehicle relevance score.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class HazardType(str, Enum):
    FOG_BANK = "fog_bank"
    LANDSLIDE = "landslide"
    WATERLOGGING = "waterlogging"
    CRASH = "crash"
    BLACK_ICE = "black_ice"


class DecayKind(str, Enum):
    NONE = "none"                # persistent: a landslide does not dissipate
    EXPONENTIAL = "exponential"  # severity decays with time constant tau
    LINEAR_TTL = "linear_ttl"    # severity decays linearly to zero over ttl


class DirectionRelevance(str, Enum):
    BOTH = "both"                # occupies/threatens both carriageways
    DIRECTIONAL = "directional"  # threatens one carriageway only


@dataclass(frozen=True)
class Hazard:
    """One hazard instance.

    ``span_start_m`` / ``span_end_m`` are positions along the corridor's
    x-axis for a highway scenario, or along ``edge_id`` for a grid scenario.
    """

    hazard_id: str
    htype: HazardType
    severity0: float
    span_start_m: float
    span_end_m: float
    onset_time_s: float
    decay: DecayKind
    direction_relevance: DirectionRelevance
    affected_direction: int          # +1 / -1, or 0 when it threatens both
    blocks_road: bool
    detection_range_m: float
    safety_deadline_s: float
    tau_s: float | None = None
    ttl_s: float | None = None
    edge_id: int | None = None       # grid scenarios: the edge it sits on
    heading_rad: float | None = None # grid scenarios: that edge's heading, which
                                     # is what "affected direction" means when
                                     # there is no single corridor axis
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.severity0 <= 1.0:
            raise ValueError(f"severity must be in [0,1], got {self.severity0}")
        if self.span_end_m < self.span_start_m:
            raise ValueError("span_end_m must be >= span_start_m")
        if self.decay is DecayKind.EXPONENTIAL and not self.tau_s:
            raise ValueError("exponential decay requires tau_s")
        if self.decay is DecayKind.LINEAR_TTL and not self.ttl_s:
            raise ValueError("linear_ttl decay requires ttl_s")
        if self.affected_direction not in (-1, 0, 1):
            raise ValueError("affected_direction must be -1, 0 or +1")

    # ------------------------------------------------------------------ time --
    @property
    def span_length_m(self) -> float:
        return self.span_end_m - self.span_start_m

    @property
    def span_mid_m(self) -> float:
        return 0.5 * (self.span_start_m + self.span_end_m)

    def severity_at(self, t: float | np.ndarray) -> float | np.ndarray:
        """Severity at absolute simulation time ``t``.

        Zero before onset. After onset the decay law applies:

        * ``none``        -> constant ``severity0``
        * ``exponential`` -> ``severity0 * exp(-(t - onset) / tau)``
        * ``linear_ttl``  -> ``severity0 * max(0, 1 - (t - onset) / ttl)``
        """
        dt = np.asarray(t, dtype=float) - self.onset_time_s
        if self.decay is DecayKind.NONE:
            sev = np.full_like(dt, self.severity0)
        elif self.decay is DecayKind.EXPONENTIAL:
            sev = self.severity0 * np.exp(-np.maximum(dt, 0.0) / float(self.tau_s))
        else:
            sev = self.severity0 * np.maximum(0.0, 1.0 - np.maximum(dt, 0.0) / float(self.ttl_s))
        sev = np.where(dt < 0.0, 0.0, sev)
        return float(sev) if np.isscalar(t) or np.ndim(t) == 0 else sev

    def is_active(self, t: float) -> bool:
        return bool(t >= self.onset_time_s and self.severity_at(t) > 1e-6)

    def contains(self, s_m: float | np.ndarray) -> np.ndarray:
        """Whether a longitudinal position lies inside the hazard span."""
        s = np.asarray(s_m, dtype=float)
        return (s >= self.span_start_m) & (s <= self.span_end_m)

    def describe(self) -> str:
        d = {1: "+x carriageway", -1: "-x carriageway", 0: "both carriageways"}
        if self.decay is DecayKind.NONE:
            decay = "persistent"
        elif self.decay is DecayKind.EXPONENTIAL:
            decay = f"exp decay tau={self.tau_s:g}s"
        else:
            decay = f"linear TTL={self.ttl_s:g}s"
        return (
            f"{self.htype.value} sev={self.severity0:.2f} "
            f"span=[{self.span_start_m:.0f},{self.span_end_m:.0f}]m "
            f"({self.span_length_m:.0f}m) onset={self.onset_time_s:g}s "
            f"{decay} affects={d[self.affected_direction]} "
            f"deadline={self.safety_deadline_s:g}s"
        )


def _deadline_for(severity: float, hz_cfg: dict[str, Any], type_default: float) -> float:
    dl = hz_cfg.get("deadlines", {})
    if dl.get("mode", "severity") != "severity":
        return float(type_default)
    for band in dl.get("bands", []):
        if severity >= float(band["min_severity"]):
            return float(band["deadline_s"])
    return float(type_default)


def hazard_from_config(
    hz_cfg: dict[str, Any],
    trace_meta: dict[str, Any],
    *,
    overrides: dict[str, Any] | None = None,
    hazard_id: str = "h0",
) -> Hazard:
    """Build the deterministic default hazard instance from ``hazard.yaml``.

    ``overrides`` may set any of ``type``, ``severity``, ``onset_time_s``,
    ``position_frac``, ``extent_m``, ``affected_direction``.
    """
    inst = {**hz_cfg["default_instance"], **(overrides or {})}
    htype = HazardType(inst["type"])
    tcfg = hz_cfg["types"][htype.value]
    severity = float(inst["severity"])
    dir_rel = DirectionRelevance(tcfg["direction_relevance"])

    edge_id: int | None = inst.get("edge_id")
    heading_rad: float | None = None

    if trace_meta.get("kind") == "grid":
        # A grid has no single corridor axis, so a hazard occupies a stretch of
        # ONE directed edge. Placing it by a scalar "position along the
        # corridor" would put it at an arbitrary point in the plane and make
        # the risk field meaningless.
        edges = trace_meta.get("edges") or []
        if not edges:
            raise ValueError(
                "Grid trace metadata carries no edge list; cannot place a hazard. "
                "Regenerate the trace (the cached .npz predates edge metadata)."
            )
        if edge_id is None:
            # Default placement: the edge whose midpoint is closest to the
            # centre of the grid, so the at-risk set is not truncated by the
            # network boundary.
            cx = np.mean([0.5 * (e["p0"][0] + e["p1"][0]) for e in edges])
            cy = np.mean([0.5 * (e["p0"][1] + e["p1"][1]) for e in edges])
            edge_id = int(min(
                edges,
                key=lambda e: (0.5 * (e["p0"][0] + e["p1"][0]) - cx) ** 2
                + (0.5 * (e["p0"][1] + e["p1"][1]) - cy) ** 2,
            )["id"])
        edge = next(e for e in edges if e["id"] == edge_id)
        heading_rad = float(edge["heading"])
        edge_len = float(edge["length"])
        # The span cannot be longer than the edge it sits on.
        extent = min(float(inst["extent_m"]), edge_len)
        mid = float(np.clip(float(inst["position_frac"]) * edge_len,
                            extent / 2.0, edge_len - extent / 2.0))
        start, end = mid - extent / 2.0, mid + extent / 2.0
        corridor_m = edge_len
    else:
        corridor_m = float(trace_meta.get("length_m") or 0.0)
        if corridor_m <= 0:
            raise ValueError("Highway trace metadata carries no corridor length_m.")
        extent = float(inst["extent_m"])
        mid = float(inst["position_frac"]) * corridor_m
        start = max(0.0, mid - extent / 2.0)
        end = min(corridor_m, mid + extent / 2.0)

    affected = 0 if dir_rel is DirectionRelevance.BOTH else int(inst["affected_direction"])

    return Hazard(
        hazard_id=hazard_id,
        htype=htype,
        severity0=severity,
        span_start_m=start,
        span_end_m=end,
        onset_time_s=float(inst["onset_time_s"]),
        decay=DecayKind(tcfg["decay"]),
        direction_relevance=dir_rel,
        affected_direction=affected,
        blocks_road=bool(tcfg["blocks_road"]),
        detection_range_m=float(inst["detection_range_m"]),
        safety_deadline_s=_deadline_for(severity, hz_cfg, tcfg["safety_deadline_s"]),
        tau_s=tcfg.get("tau_s"),
        ttl_s=tcfg.get("ttl_s"),
        edge_id=edge_id,
        heading_rad=heading_rad,
        meta={"source": "default_instance", "corridor_m": corridor_m},
    )


def sample_hazard(
    hz_cfg: dict[str, Any],
    trace_meta: dict[str, Any],
    rng: np.random.Generator,
    *,
    htype: str | HazardType | None = None,
    hazard_id: str = "h0",
    position_frac_range: tuple[float, float] = (0.35, 0.75),
    onset_time_s: float | None = None,
) -> Hazard:
    """Randomly sample a hazard instance for the experiment sweep.

    Severity and extent come from the per-type ranges in ``hazard.yaml``;
    placement is restricted to the middle of the corridor so that there is a
    populated upstream at-risk set to warn.
    """
    types = list(hz_cfg["types"])
    chosen = HazardType(htype) if htype is not None else HazardType(rng.choice(types))
    tcfg = hz_cfg["types"][chosen.value]

    sev_lo, sev_hi = tcfg["severity_range"]
    ext_lo, ext_hi = tcfg["spatial_extent_m"]
    severity = float(rng.uniform(sev_lo, sev_hi))
    extent = float(rng.uniform(ext_lo, ext_hi))
    frac = float(rng.uniform(*position_frac_range))
    dir_rel = DirectionRelevance(tcfg["direction_relevance"])
    affected = 0 if dir_rel is DirectionRelevance.BOTH else int(rng.choice([-1, 1]))

    inst_defaults = hz_cfg["default_instance"]
    return hazard_from_config(
        {**hz_cfg, "default_instance": {
            **inst_defaults,
            "type": chosen.value,
            "severity": severity,
            "extent_m": extent,
            "position_frac": frac,
            "affected_direction": affected if affected != 0 else 1,
            "onset_time_s": (
                float(inst_defaults["onset_time_s"]) if onset_time_s is None else onset_time_s
            ),
        }},
        trace_meta,
        hazard_id=hazard_id,
    )
