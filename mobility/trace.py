"""The compact mobility trace: a dense ``[T, N]`` array bundle.

The build brief specifies a per-sample record layout::

    (timestep, vehicle_id, x, y, vx, vy, heading, lane)

We store the same information in dense ``[T, N]`` arrays (T timesteps, N vehicle
slots) plus an ``active`` mask, which is what every downstream consumer actually
wants: a whole-timestep slice with no grouping. :meth:`Trace.to_records` emits
the long form when it is needed (e.g. for exporting to other tools).

Coordinates are metres in a scenario-local Cartesian frame. Headings are
radians, 0 = +x, counter-clockwise. Arrays are float32; the positional
quantisation that costs (~1e-4 m at 10 km) is far below any modelled effect.
"""

from __future__ import annotations

import json
import os
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from common.config import ensure_dir
from common.logging_utils import get_logger

logger = get_logger("mobility.trace")

TRACE_FORMAT_VERSION = 1


@dataclass
class VehicleState:
    """One vehicle at one timestep. Materialised on demand by the simulator."""

    index: int
    vehicle_id: str
    vclass: str
    x: float
    y: float
    vx: float
    vy: float
    heading: float
    lane: int
    direction: int

    @property
    def speed(self) -> float:
        return float(np.hypot(self.vx, self.vy))


@dataclass
class Trace:
    """A cached mobility trace.

    Attributes
    ----------
    dt : float
        Sample period in seconds (100 ms throughout this project).
    x, y, vx, vy, heading, lane, active : np.ndarray
        ``[T, N]`` arrays. ``active[t, i]`` is False when slot ``i`` holds no
        vehicle at step ``t`` (it has not entered the corridor yet, or has
        already left it). All other arrays are undefined where inactive.
    vehicle_ids, vclass, length_m, direction : np.ndarray
        ``[N]`` per-vehicle static attributes. ``direction`` is +1/-1 for a
        highway carriageway and 0 for a grid scenario where it is meaningless.
    routes : list | None
        Per-vehicle planned route as a tuple of edge ids; grid scenarios only.
        Used by the route-aware risk field.
    """

    dt: float
    x: np.ndarray
    y: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    heading: np.ndarray
    lane: np.ndarray
    active: np.ndarray
    vehicle_ids: np.ndarray
    vclass: np.ndarray
    length_m: np.ndarray
    direction: np.ndarray
    scenario: str
    backend: str
    routes: list[tuple[str, ...]] | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    # ---------------------------------------------------------------- shape --
    @property
    def n_steps(self) -> int:
        return int(self.x.shape[0])

    @property
    def n_vehicles(self) -> int:
        return int(self.x.shape[1])

    @property
    def duration_s(self) -> float:
        return self.n_steps * self.dt

    def time_of(self, step: int) -> float:
        return step * self.dt

    def step_of(self, t: float) -> int:
        return int(round(t / self.dt))

    # ------------------------------------------------------------- accessors --
    def speed(self, step: int) -> np.ndarray:
        """Scalar speed [N] at ``step`` (0 where inactive)."""
        return np.hypot(self.vx[step], self.vy[step])

    def active_indices(self, step: int) -> np.ndarray:
        return np.flatnonzero(self.active[step])

    def positions(self, step: int) -> np.ndarray:
        """``[N, 2]`` position array at ``step``."""
        return np.stack([self.x[step], self.y[step]], axis=1)

    def state(self, step: int, index: int) -> VehicleState:
        return VehicleState(
            index=index,
            vehicle_id=str(self.vehicle_ids[index]),
            vclass=str(self.vclass[index]),
            x=float(self.x[step, index]),
            y=float(self.y[step, index]),
            vx=float(self.vx[step, index]),
            vy=float(self.vy[step, index]),
            heading=float(self.heading[step, index]),
            lane=int(self.lane[step, index]),
            direction=int(self.direction[index]),
        )

    def iter_active(self, step: int) -> Iterator[VehicleState]:
        for i in self.active_indices(step):
            yield self.state(step, int(i))

    # ------------------------------------------------------------ diagnostics --
    def summary(self) -> dict[str, Any]:
        counts = self.active.sum(axis=1)
        spd = np.hypot(self.vx, self.vy)
        moving = spd[self.active & (spd > 0.1)]
        return {
            "scenario": self.scenario,
            "backend": self.backend,
            "dt_s": self.dt,
            "steps": self.n_steps,
            "duration_s": round(self.duration_s, 2),
            "vehicle_slots": self.n_vehicles,
            "concurrent_mean": round(float(counts.mean()), 1),
            "concurrent_min": int(counts.min()),
            "concurrent_max": int(counts.max()),
            "speed_mean_ms": round(float(moving.mean()), 2) if moving.size else 0.0,
            "speed_min_ms": round(float(moving.min()), 2) if moving.size else 0.0,
            "speed_max_ms": round(float(moving.max()), 2) if moving.size else 0.0,
            "truck_share": round(float((self.vclass == "truck").mean()), 3),
        }

    def to_records(self) -> np.ndarray:
        """Long form: one structured record per active (timestep, vehicle)."""
        dtype = np.dtype(
            [
                ("timestep", np.int32),
                ("vehicle_id", "U16"),
                ("x", np.float32),
                ("y", np.float32),
                ("vx", np.float32),
                ("vy", np.float32),
                ("heading", np.float32),
                ("lane", np.int16),
            ]
        )
        ts, vi = np.nonzero(self.active)
        out = np.empty(ts.size, dtype=dtype)
        out["timestep"] = ts
        out["vehicle_id"] = self.vehicle_ids[vi]
        for name in ("x", "y", "vx", "vy", "heading", "lane"):
            out[name] = getattr(self, name)[ts, vi]
        return out


# ---------------------------------------------------------------------------
# Persistence: .npz so a trace is generated once and never regenerated.
# ---------------------------------------------------------------------------
_ARRAY_FIELDS = (
    "x", "y", "vx", "vy", "heading", "lane", "active",
    "vehicle_ids", "vclass", "length_m", "direction",
)


def save_trace(trace: Trace, path: Path) -> Path:
    ensure_dir(path.parent)
    arrays = {name: getattr(trace, name) for name in _ARRAY_FIELDS}
    meta = {
        "format_version": TRACE_FORMAT_VERSION,
        "dt": trace.dt,
        "scenario": trace.scenario,
        "backend": trace.backend,
        "routes": [list(r) for r in trace.routes] if trace.routes is not None else None,
        "meta": trace.meta,
    }
    # Write to a sibling temp file and rename into place. A process killed
    # mid-write (run3 was stopped while saving) used to leave a truncated
    # .npz under the real name, which crashed run4 13 updates later.
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        with open(tmp, "wb") as fh:
            np.savez_compressed(fh, _meta=np.array(json.dumps(meta)), **arrays)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    size_mb = path.stat().st_size / 1e6
    logger.info("Cached trace -> %s (%.1f MB)", path.name, size_mb)
    return path


class CorruptTraceError(ValueError):
    """A cached trace file exists but cannot be read (truncated or damaged)."""


def load_trace(path: Path) -> Trace:
    # np.load sniffs the magic bytes and reports non-zip garbage as "pickled
    # data" (a ValueError), indistinguishable from a format mismatch; check first.
    with open(path, "rb") as fh:
        magic = fh.read(4)
    if magic not in (b"PK\x03\x04", b"PK\x05\x06"):
        raise CorruptTraceError(f"{path.name}: not an .npz archive (magic {magic!r})")
    try:
        return _load_trace(path)
    except (zipfile.BadZipFile, EOFError, KeyError, OSError, json.JSONDecodeError) as exc:
        if isinstance(exc, FileNotFoundError):
            raise
        raise CorruptTraceError(f"{path.name}: {type(exc).__name__}: {exc}") from exc


def _load_trace(path: Path) -> Trace:
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["_meta"]))
        if meta.get("format_version") != TRACE_FORMAT_VERSION:
            raise ValueError(
                f"{path.name}: trace format v{meta.get('format_version')} != "
                f"v{TRACE_FORMAT_VERSION}; delete the cache and regenerate."
            )
        arrays = {name: z[name] for name in _ARRAY_FIELDS}
    routes = meta["routes"]
    return Trace(
        dt=float(meta["dt"]),
        scenario=meta["scenario"],
        backend=meta["backend"],
        routes=[tuple(r) for r in routes] if routes is not None else None,
        meta=meta["meta"],
        **arrays,
    )
