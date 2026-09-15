"""Bring real OpenStreetMap networks into the straight-axis geometry model.

User decision: rather than generalise the risk field to arbitrary road
polylines, pick real roads whose geometry already satisfies the model and map
them onto it, reporting the mapping error.

* **Highway.** A genuinely straight stretch of a real two-lane road (US-50,
  central Nevada: 10,394 m within 1.3 m of its chord). Positions are projected
  onto the chord: ``x`` = distance along it, ``y`` = signed lateral offset;
  velocities and headings are rotated with them. Vehicles outside the window
  are inactive. :class:`hazard.risk_field.HighwayRiskGeometry` then applies
  unchanged.
* **Grid.** A real grid city (Midtown Manhattan: 98.5% of street length within
  3 degrees of two perpendicular axes). The trace is rotated so the grid axes
  align with x and y, and each network edge becomes a grid edge record in the
  format :func:`hazard.model.hazard_from_config` expects. Off-axis edges are
  dropped from the hazard candidates and counted.

Everything here is pure geometry on arrays, testable without SUMO; only
:func:`read_net_edges` needs SUMO's ``sumolib``.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Sequence

import numpy as np

from mobility.trace import Trace


# ---------------------------------------------------------------------------
# Polylines
# ---------------------------------------------------------------------------
def _arc_lengths(pts: np.ndarray) -> np.ndarray:
    seg = np.hypot(*np.diff(pts, axis=0).T)
    return np.concatenate([[0.0], np.cumsum(seg)])


def _point_at(pts: np.ndarray, s_cum: np.ndarray, s: float) -> np.ndarray:
    i = int(np.clip(np.searchsorted(s_cum, s, side="right") - 1, 0, len(pts) - 2))
    f = (s - s_cum[i]) / max(s_cum[i + 1] - s_cum[i], 1e-12)
    return pts[i] + f * (pts[i + 1] - pts[i])


def straightest_window(points: Sequence[Sequence[float]], window_m: float
                       ) -> tuple[np.ndarray, np.ndarray, float]:
    """The ``window_m``-long stretch of a polyline closest to a straight line.

    Candidate windows start at every vertex and run ``window_m`` of arc length.
    Returns the chord endpoints ``(a, b)`` and the largest perpendicular
    distance of any polyline point inside the window from that chord.
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 2:
        raise ValueError("points must be an (n >= 2, 2) polyline")
    s_cum = _arc_lengths(pts)
    if s_cum[-1] < window_m:
        raise ValueError(f"polyline is {s_cum[-1]:.0f} m, shorter than the {window_m:.0f} m window")
    best = None
    starts = [s for s in s_cum if s + window_m <= s_cum[-1] + 1e-9]
    for s0 in starts:
        a = _point_at(pts, s_cum, s0)
        b = _point_at(pts, s_cum, s0 + window_m)
        inner = pts[(s_cum > s0) & (s_cum < s0 + window_m)]
        chord = b - a
        L = float(np.hypot(*chord))
        dev = 0.0
        if len(inner) and L > 0:
            rel = inner - a
            dev = float(np.max(np.abs(chord[0] * rel[:, 1] - chord[1] * rel[:, 0]) / L))
        if best is None or dev < best[2]:
            best = (a, b, dev)
    return best


def project_points(x: np.ndarray, y: np.ndarray, a: Sequence[float], b: Sequence[float]
                   ) -> tuple[np.ndarray, np.ndarray]:
    """(distance along a->b, signed lateral offset; left of a->b is positive)."""
    a = np.asarray(a, dtype=float)
    u = np.asarray(b, dtype=float) - a
    u = u / np.hypot(*u)
    rx, ry = np.asarray(x, dtype=float) - a[0], np.asarray(y, dtype=float) - a[1]
    return rx * u[0] + ry * u[1], u[0] * ry - u[1] * rx


def _rotate(vx: np.ndarray, vy: np.ndarray, theta: float) -> tuple[np.ndarray, np.ndarray]:
    c, s = math.cos(theta), math.sin(theta)
    return c * vx - s * vy, s * vx + c * vy


# ---------------------------------------------------------------------------
# Traces
# ---------------------------------------------------------------------------
def project_highway_trace(trace: Trace, a: Sequence[float], b: Sequence[float],
                          lateral_max_m: float = 30.0, max_dev_m: float | None = None) -> Trace:
    """Map a trace onto the straight-corridor frame of the chord ``a -> b``.

    ``x`` becomes distance along the chord, ``y`` the lateral offset. Samples
    outside ``[0, |ab|]`` or farther than ``lateral_max_m`` from the chord (side
    roads) become inactive. Each vehicle's direction is the sign of its mean
    along-chord velocity while active.
    """
    a = np.asarray(a, dtype=float)
    chord = np.asarray(b, dtype=float) - a
    L = float(np.hypot(*chord))
    theta = -math.atan2(chord[1], chord[0])
    s, lat = project_points(trace.x, trace.y, a, b)
    vx, vy = _rotate(np.asarray(trace.vx, float), np.asarray(trace.vy, float), theta)
    active = (np.asarray(trace.active, bool) & np.isfinite(s) & (s >= 0.0) & (s <= L)
              & (np.abs(lat) <= lateral_max_m))
    mean_v = np.where(active, vx, 0.0).sum(axis=0) / np.maximum(active.sum(axis=0), 1)
    direction = np.where(mean_v < 0, -1, 1).astype(np.int8)
    heading = (np.asarray(trace.heading, float) + theta + math.pi) % (2 * math.pi) - math.pi
    meta = {**trace.meta, "kind": "highway", "length_m": L, "directions": 2,
            "projection": {"type": "chord", "a": a.tolist(), "b": np.asarray(b, float).tolist(),
                           "length_m": L, "lateral_max_m": lateral_max_m,
                           "max_polyline_deviation_m": max_dev_m}}
    return dataclasses.replace(
        trace, x=np.where(active, s, np.nan).astype(np.float32),
        y=np.where(active, lat, np.nan).astype(np.float32),
        vx=np.where(active, vx, 0.0).astype(np.float32),
        vy=np.where(active, vy, 0.0).astype(np.float32),
        heading=heading.astype(np.float32), active=active, direction=direction, meta=meta)


def rotate_trace(trace: Trace, theta_deg: float, origin: Sequence[float] = (0.0, 0.0)) -> Trace:
    """Rotate positions, velocities and headings by ``theta_deg`` about ``origin``."""
    th = math.radians(theta_deg)
    ox, oy = float(origin[0]), float(origin[1])
    x, y = _rotate(np.asarray(trace.x, float) - ox, np.asarray(trace.y, float) - oy, th)
    vx, vy = _rotate(np.asarray(trace.vx, float), np.asarray(trace.vy, float), th)
    heading = (np.asarray(trace.heading, float) + th + math.pi) % (2 * math.pi) - math.pi
    meta = {**trace.meta, "rotation": {"theta_deg": theta_deg, "origin": [ox, oy]}}
    return dataclasses.replace(trace, x=x.astype(np.float32), y=y.astype(np.float32),
                               vx=vx.astype(np.float32), vy=vy.astype(np.float32),
                               heading=heading.astype(np.float32), meta=meta)


# ---------------------------------------------------------------------------
# Grid edges
# ---------------------------------------------------------------------------
def grid_edge_records(edges: Sequence[dict[str, Any]], theta_deg: float,
                      origin: Sequence[float] = (0.0, 0.0), axis_tolerance_deg: float = 5.0,
                      min_length_m: float = 20.0) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Rotated, axis-classified grid edges in the hazard model's format.

    ``edges`` items need ``sumo_id``, ``shape`` (list of (x, y)) and ``length``.
    Each output record carries ``id`` (int), ``p0``, ``p1``, ``axis`` ('x' or
    'y'), ``length``, ``heading`` and ``sumo_id``. Edges more than
    ``axis_tolerance_deg`` off both axes, or shorter than ``min_length_m``, are
    dropped; the returned stats say how much.
    """
    th = math.radians(theta_deg)
    ox, oy = float(origin[0]), float(origin[1])
    tol = math.radians(axis_tolerance_deg)
    out: list[dict[str, Any]] = []
    total = kept = 0.0
    for e in edges:
        shape = np.asarray(e["shape"], dtype=float)
        (x0, y0), (x1, y1) = shape[0], shape[-1]
        p0 = _rotate(np.array(x0 - ox), np.array(y0 - oy), th)
        p1 = _rotate(np.array(x1 - ox), np.array(y1 - oy), th)
        length = float(e["length"])
        total += length
        if length < min_length_m:
            continue
        heading = math.atan2(float(p1[1] - p0[1]), float(p1[0] - p0[0]))
        off_x = abs(math.sin(heading))       # 0 when parallel to x
        off_y = abs(math.cos(heading))       # 0 when parallel to y
        if off_x <= math.sin(tol):
            axis = "x"
        elif off_y <= math.sin(tol):
            axis = "y"
        else:
            continue
        kept += length
        out.append({"id": len(out), "p0": [float(p0[0]), float(p0[1])],
                    "p1": [float(p1[0]), float(p1[1])], "axis": axis, "length": length,
                    "heading": heading, "sumo_id": str(e["sumo_id"])})
    stats = {"n_edges_in": len(edges), "n_edges_kept": len(out),
             "length_in_m": total, "length_kept_m": kept,
             "length_kept_frac": kept / total if total else float("nan")}
    return out, stats


def read_net_edges(net_path: str, sumo_home: str | None = None,
                   types: Sequence[str] | None = None) -> list[dict[str, Any]]:
    """Non-internal edges of a SUMO network: id, type, lanes, length, shape, nodes."""
    import os
    import sys

    home = sumo_home or os.environ.get("SUMO_HOME")
    if home and os.path.join(home, "tools") not in sys.path:
        sys.path.append(os.path.join(home, "tools"))
    import sumolib  # noqa: PLC0415

    net = sumolib.net.readNet(str(net_path), withInternal=False)
    out = []
    for e in net.getEdges():
        etype = e.getType() or ""
        if types and not any(t in etype for t in types):
            continue
        out.append({"sumo_id": e.getID(), "type": etype, "lanes": e.getLaneNumber(),
                    "length": float(e.getLength()),
                    "shape": [tuple(map(float, p)) for p in e.getShape()],
                    "from": e.getFromNode().getID(), "to": e.getToNode().getID()})
    return out
