"""Parse a SUMO FCD XML export into the compact :class:`~mobility.trace.Trace`.

FCD XML is enormous (hundreds of MB for a 10 km corridor at 100 ms) and is
parsed exactly once; everything downstream reads the cached ``.npz``.

SUMO's ``angle`` attribute is navigational degrees (0 = north, increasing
clockwise). We convert to the project convention of mathematical radians
(0 = +x / east, increasing counter-clockwise)::

    heading_rad = radians(90 - angle_deg)
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np

from common.logging_utils import get_logger
from mobility.trace import Trace

logger = get_logger("mobility.fcd")


def sumo_angle_to_heading(angle_deg: float) -> float:
    """Navigational degrees -> mathematical radians in (-pi, pi]."""
    h = math.radians(90.0 - angle_deg)
    return (h + math.pi) % (2 * math.pi) - math.pi


def parse_fcd(
    fcd_path: Path,
    *,
    scenario_name: str,
    dt: float,
    warmup_s: float = 0.0,
    truck_types: tuple[str, ...] = ("truck", "trailer", "bus"),
    meta: dict[str, Any] | None = None,
) -> Trace:
    """Stream-parse an FCD export into a dense trace.

    Two passes: the first indexes vehicle ids and timesteps, the second fills
    the arrays. Streaming twice costs less wall-clock than holding the parsed
    DOM of a multi-hundred-MB file in memory.
    """
    fcd_path = Path(fcd_path)
    logger.info("Parsing FCD export %s (%.1f MB)", fcd_path.name, fcd_path.stat().st_size / 1e6)

    vehicle_ids: dict[str, int] = {}
    lane_ids: dict[str, int] = {}
    vtypes: dict[str, str] = {}
    lengths: dict[str, float] = {}
    times: list[float] = []

    for _, elem in ET.iterparse(fcd_path, events=("end",)):
        if elem.tag == "timestep":
            times.append(float(elem.get("time", "0")))
            for v in elem:
                vid = v.get("id")
                if vid is not None and vid not in vehicle_ids:
                    vehicle_ids[vid] = len(vehicle_ids)
                    vtypes[vid] = v.get("type", "car")
                lane = v.get("lane")
                if lane is not None and lane not in lane_ids:
                    lane_ids[lane] = len(lane_ids)
            elem.clear()

    if not times:
        raise ValueError(f"{fcd_path} contained no <timestep> elements.")

    t0 = min(times)
    keep_from = t0 + warmup_s
    kept_times = [t for t in times if t >= keep_from - 1e-9]
    step_of = {round(t, 4): i for i, t in enumerate(kept_times)}
    T, N = len(kept_times), len(vehicle_ids)
    logger.info("FCD: %d timesteps (%d after warmup), %d unique vehicles", len(times), T, N)

    x = np.full((T, N), np.nan, dtype=np.float32)
    y = np.full((T, N), np.nan, dtype=np.float32)
    vx = np.zeros((T, N), dtype=np.float32)
    vy = np.zeros((T, N), dtype=np.float32)
    heading = np.zeros((T, N), dtype=np.float32)
    lane = np.zeros((T, N), dtype=np.int16)
    active = np.zeros((T, N), dtype=bool)

    for _, elem in ET.iterparse(fcd_path, events=("end",)):
        if elem.tag != "timestep":
            continue
        t = round(float(elem.get("time", "0")), 4)
        s = step_of.get(t)
        if s is None:
            elem.clear()
            continue
        for v in elem:
            vid = v.get("id")
            if vid is None:
                continue
            i = vehicle_ids[vid]
            speed = float(v.get("speed", "0"))
            h = sumo_angle_to_heading(float(v.get("angle", "90")))
            x[s, i] = float(v.get("x", "0"))
            y[s, i] = float(v.get("y", "0"))
            vx[s, i] = speed * math.cos(h)
            vy[s, i] = speed * math.sin(h)
            heading[s, i] = h
            lane[s, i] = lane_ids.get(v.get("lane", ""), 0)
            active[s, i] = True
        elem.clear()

    order = sorted(vehicle_ids, key=lambda k: vehicle_ids[k])
    vclass = np.array(
        ["truck" if any(tt in vtypes[v].lower() for tt in truck_types) else "car" for v in order],
        dtype="U8",
    )
    length_m = np.array([lengths.get(v, 12.0 if c == "truck" else 4.5)
                         for v, c in zip(order, vclass)], dtype=np.float32)

    # Travel direction from the mean heading, for highway scenarios.
    mean_hx = np.nansum(np.where(active, np.cos(heading), 0.0), axis=0)
    direction = np.where(mean_hx >= 0, 1, -1).astype(np.int8)

    return Trace(
        dt=dt, x=x, y=y, vx=vx, vy=vy, heading=heading, lane=lane, active=active,
        vehicle_ids=np.array(order, dtype="U16"),
        vclass=vclass, length_m=length_m, direction=direction,
        scenario=scenario_name, backend="sumo",
        meta={**(meta or {}), "fcd_source": str(fcd_path), "lane_ids": lane_ids},
    )
