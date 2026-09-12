"""Pure-Python fallback mobility generator.

Used when no SUMO installation is detected, so the whole pipeline runs on a
bare Python environment. It is a genuine microscopic model, not a placeholder:
vehicles follow the Krauss (1998) car-following rule -- the same model SUMO
uses by default -- with SUMO-style dawdling, per-driver desired speeds and a
heterogeneous car/truck fleet.

What it is NOT: it has no lane changing, no OSM road geometry, no gap
acceptance at junctions and no calibrated demand profile. Results produced with
this backend must be labelled ``backend=fallback`` in the paper and must not be
presented as SUMO results. :mod:`mobility.generate` stamps the backend into
every trace and every results row for exactly this reason.

Density control
---------------
Vehicle count per lane is regulated to the commanded density rather than being
left to emerge from an inflow rate. The density sweep in Phase 7 treats density
as the independent variable, so it has to be the quantity we actually hold
fixed; letting it drift would confound every cell of the sweep.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from common.logging_utils import get_logger
from mobility.trace import Trace

logger = get_logger("mobility.fallback")

KMH_TO_MS = 1.0 / 3.6


@dataclass
class FleetParams:
    """Per-vehicle-class parameters, resolved from the scenario YAML."""

    names: list[str]
    shares: np.ndarray
    speed_min_ms: np.ndarray
    speed_max_ms: np.ndarray
    length_m: np.ndarray
    accel: np.ndarray
    decel: np.ndarray

    @classmethod
    def from_config(cls, veh_cfg: dict[str, Any]) -> "FleetParams":
        classes = veh_cfg["classes"]
        names = list(classes.keys())
        shares = np.array([classes[n]["share"] for n in names], dtype=float)
        if not math.isclose(shares.sum(), 1.0, rel_tol=1e-6):
            raise ValueError(f"Vehicle class shares must sum to 1.0, got {shares.sum():.4f}")
        return cls(
            names=names,
            shares=shares,
            speed_min_ms=np.array([classes[n]["speed_kmh_min"] for n in names]) * KMH_TO_MS,
            speed_max_ms=np.array([classes[n]["speed_kmh_max"] for n in names]) * KMH_TO_MS,
            length_m=np.array([classes[n]["length_m"] for n in names], dtype=float),
            accel=np.array([classes[n]["max_accel_ms2"] for n in names], dtype=float),
            decel=np.array([classes[n]["max_decel_ms2"] for n in names], dtype=float),
        )


def _krauss_velocity(
    v: np.ndarray,
    v_lead: np.ndarray,
    gap: np.ndarray,
    v_desired: np.ndarray,
    accel: np.ndarray,
    decel: np.ndarray,
    tau: float,
    sigma: float,
    dt: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """One Krauss car-following update, vectorised over a lane.

    Safe velocity (Krauss 1998, as implemented in SUMO)::

        v_safe = v_lead + (gap - v_lead * tau) / (tau + (v_lead + v) / (2 * b))

    The follower then takes ``v_des = min(v_max, v + a*dt, v_safe)`` and dawdles
    off a random fraction of one acceleration step, which is what keeps the
    stream from collapsing onto a deterministic platoon.

    ``gap`` is bumper-to-bumper clearance already net of ``minGap``; it may be
    ``inf`` for a leaderless vehicle.
    """
    denom = tau + (v_lead + v) / (2.0 * decel)
    with np.errstate(invalid="ignore", divide="ignore"):
        v_safe = v_lead + (gap - v_lead * tau) / np.maximum(denom, 1e-9)
    v_safe = np.where(np.isfinite(gap), v_safe, np.inf)
    v_des = np.minimum.reduce([v_desired, v + accel * dt, v_safe])
    v_des = np.maximum(v_des, 0.0)
    # SUMO-style dawdling: lose up to sigma * a * dt of speed at random.
    v_next = v_des - sigma * accel * dt * rng.random(v.shape)
    return np.clip(v_next, 0.0, v_desired)


class _LaneStream:
    """One directed lane, held as a position-ordered list of global slots.

    Vehicles only ever enter at the upstream end and leave at the downstream
    end, so the ordering is maintained by appending and popping, never sorting.
    """

    __slots__ = ("order",)

    def __init__(self) -> None:
        self.order: list[int] = []  # most downstream first


def _sample_fleet(
    n: int, fleet: FleetParams, speed_dev: float, speed_factor: float, rng: np.random.Generator
) -> dict[str, np.ndarray]:
    """Sample ``n`` vehicles' static attributes."""
    cls_idx = rng.choice(len(fleet.names), size=n, p=fleet.shares)
    base = rng.uniform(fleet.speed_min_ms[cls_idx], fleet.speed_max_ms[cls_idx])
    # Per-driver desired-speed deviation, truncated at +/-2 sigma exactly as
    # SUMO truncates its speedDev distribution. A wider truncation produces
    # implausibly slow trucks which, on a single-lane carriageway with no
    # overtaking, dominate the whole speed distribution.
    dev = np.clip(rng.normal(1.0, speed_dev, size=n), 1.0 - 2 * speed_dev, 1.0 + 2 * speed_dev)
    v_desired = base * dev * speed_factor
    return {
        "class_index": cls_idx,
        "vclass": np.array([fleet.names[i] for i in cls_idx], dtype="U8"),
        "v_desired": v_desired,
        "length_m": fleet.length_m[cls_idx],
        "accel": fleet.accel[cls_idx],
        "decel": fleet.decel[cls_idx],
    }


class _SlotPool:
    """Grow-on-demand storage for vehicle slots over the whole run."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.n = 0
        self.pos = np.zeros(capacity)          # metres along direction of travel
        self.vel = np.zeros(capacity)
        self.v_desired = np.zeros(capacity)
        self.length_m = np.zeros(capacity)
        self.accel = np.zeros(capacity)
        self.decel = np.zeros(capacity)
        self.vclass = np.empty(capacity, dtype="U8")
        self.lane = np.zeros(capacity, dtype=np.int16)
        self.direction = np.zeros(capacity, dtype=np.int8)
        self.edge = np.zeros(capacity, dtype=np.int32)      # grid only
        self.routes: list[list[int]] = []

    def _grow(self) -> None:
        new_cap = max(16, self.capacity * 2)
        for name in ("pos", "vel", "v_desired", "length_m", "accel", "decel",
                     "vclass", "lane", "direction", "edge"):
            arr = getattr(self, name)
            grown = np.zeros(new_cap, dtype=arr.dtype)
            grown[: self.capacity] = arr
            setattr(self, name, grown)
        self.capacity = new_cap

    def add(self, **attrs: Any) -> int:
        if self.n >= self.capacity:
            self._grow()
        i = self.n
        self.n += 1
        for k, v in attrs.items():
            getattr(self, k)[i] = v
        return i


# ---------------------------------------------------------------------------
# Highway corridor
# ---------------------------------------------------------------------------
def generate_highway_trace(
    scenario: dict[str, Any],
    density_veh_km_lane: float,
    rng: np.random.Generator,
    *,
    duration_s: float | None = None,
    speed_factor: float = 1.0,
    headway_factor: float = 1.0,
) -> Trace:
    """Generate a straight bidirectional highway corridor trace.

    Parameters
    ----------
    density_veh_km_lane:
        Commanded density, held fixed by the insertion controller.
    speed_factor, headway_factor:
        Weather effects on *driving behaviour* (not on the radio). Supplied by
        the caller from the weather config.
    """
    geo, veh, sim = scenario["geometry"], scenario["vehicles"], scenario["simulation"]
    L = float(geo["length_m"])
    dt = float(sim["timestep_s"])
    duration = float(sim["duration_s"] if duration_s is None else duration_s)
    warmup = float(sim["warmup_s"])
    n_rec = int(round(duration / dt))
    n_warm = int(round(warmup / dt))

    fleet = FleetParams.from_config(veh)
    tau = float(veh["reaction_time_s"]) * headway_factor
    sigma = float(veh["driver_imperfection"])
    min_gap = float(veh["min_gap_m"]) * headway_factor
    speed_dev = float(veh["speed_dev"])

    lanes_per_dir = int(geo["lanes_per_direction"])
    directions = [1, -1] if int(geo["directions"]) == 2 else [1]
    lane_w = float(geo["lane_width_m"])
    median = float(geo.get("median_width_m", 0.0))

    # (direction, lane_in_direction) -> flat lane id
    lane_keys = [(d, j) for d in directions for j in range(lanes_per_dir)]
    n_lanes = len(lane_keys)
    target_per_lane = max(1, int(round(density_veh_km_lane * L / 1000.0)))

    pool = _SlotPool(capacity=max(64, target_per_lane * n_lanes * 3))
    streams = [_LaneStream() for _ in range(n_lanes)]

    def _spawn(lane_id: int, s: float, v: float) -> int:
        d, j = lane_keys[lane_id]
        a = _sample_fleet(1, fleet, speed_dev, speed_factor, rng)
        return pool.add(
            pos=s,
            vel=min(v, float(a["v_desired"][0])),
            v_desired=float(a["v_desired"][0]),
            length_m=float(a["length_m"][0]),
            accel=float(a["accel"][0]),
            decel=float(a["decel"][0]),
            vclass=str(a["vclass"][0]),
            lane=lane_id,
            direction=d,
        )

    # --- initial placement: jittered uniform spacing at desired speed --------
    spacing = L / target_per_lane
    for lane_id in range(n_lanes):
        s = L  # fill from the downstream end backwards
        for _ in range(target_per_lane):
            s -= spacing * rng.uniform(0.7, 1.3)
            if s <= 0.0:
                break
            slot = _spawn(lane_id, s, v=0.0)
            pool.vel[slot] = pool.v_desired[slot]
            streams[lane_id].order.append(slot)

    # --- recording buffers ---------------------------------------------------
    rec_pos: list[np.ndarray] = []
    rec_vel: list[np.ndarray] = []
    rec_act: list[np.ndarray] = []
    n_steps_total = n_warm + n_rec

    for step in range(n_steps_total):
        for lane_id, stream in enumerate(streams):
            order = stream.order
            if not order:
                continue
            idx = np.asarray(order, dtype=np.intp)
            pos, vel = pool.pos[idx], pool.vel[idx]
            # Leader of element k is element k-1 (further downstream).
            gap = np.empty_like(pos)
            v_lead = np.empty_like(pos)
            gap[0] = np.inf
            v_lead[0] = 0.0
            if idx.size > 1:
                gap[1:] = pos[:-1] - pos[1:] - pool.length_m[idx[:-1]] - min_gap
                v_lead[1:] = vel[:-1]
            new_vel = _krauss_velocity(
                vel, v_lead, np.maximum(gap, 0.0),
                pool.v_desired[idx], pool.accel[idx], pool.decel[idx],
                tau, sigma, dt, rng,
            )
            pool.vel[idx] = new_vel
            pool.pos[idx] = pos + new_vel * dt

            # Depart at the downstream end.
            while order and pool.pos[order[0]] > L:
                order.pop(0)
            # Insert at the upstream end to hold density.
            if len(order) < target_per_lane:
                tail_ok = True
                if order:
                    last = order[-1]
                    need = min_gap + pool.length_m[last] + 5.0
                    tail_ok = pool.pos[last] > need
                if tail_ok and rng.random() < 0.5:
                    v_in = pool.vel[order[-1]] if order else 0.0
                    slot = _spawn(lane_id, s=0.0, v=v_in if order else 25.0)
                    if not order:
                        pool.vel[slot] = pool.v_desired[slot]
                    order.append(slot)

        if step >= n_warm:
            act = np.zeros(pool.capacity, dtype=bool)
            for stream in streams:
                if stream.order:
                    act[np.asarray(stream.order, dtype=np.intp)] = True
            rec_act.append(act)
            rec_pos.append(pool.pos.copy())
            rec_vel.append(pool.vel.copy())

    n_slots = pool.n
    T = len(rec_act)

    def _stack(chunks: list[np.ndarray]) -> np.ndarray:
        out = np.zeros((T, n_slots), dtype=np.float32)
        for t, c in enumerate(chunks):
            out[t] = c[:n_slots]
        return out

    s_arr = _stack(rec_pos)
    v_arr = _stack(rec_vel)
    active = np.zeros((T, n_slots), dtype=bool)
    for t, c in enumerate(rec_act):
        active[t] = c[:n_slots]

    direction = pool.direction[:n_slots].astype(np.int8)
    lane_id_arr = pool.lane[:n_slots].astype(np.int16)

    # Map along-direction coordinate s to the Cartesian frame.
    dir_sign = direction.astype(np.float32)
    x = np.where(dir_sign > 0, s_arr, L - s_arr).astype(np.float32)
    lane_in_dir = np.array([lane_keys[i][1] for i in lane_id_arr], dtype=np.float32)
    y_off = (median / 2.0 + (0.5 + lane_in_dir) * lane_w) * dir_sign
    y = np.broadcast_to(y_off, (T, n_slots)).astype(np.float32).copy()

    vx = (v_arr * dir_sign).astype(np.float32)
    vy = np.zeros_like(vx)
    heading = np.broadcast_to(
        np.where(dir_sign > 0, 0.0, np.pi).astype(np.float32), (T, n_slots)
    ).copy()

    x[~active] = np.nan
    y[~active] = np.nan

    return Trace(
        dt=dt,
        x=x, y=y, vx=vx, vy=vy, heading=heading,
        lane=np.broadcast_to(lane_id_arr, (T, n_slots)).copy(),
        active=active,
        vehicle_ids=np.array([f"v{i}" for i in range(n_slots)], dtype="U16"),
        vclass=pool.vclass[:n_slots].astype("U8"),
        length_m=pool.length_m[:n_slots].astype(np.float32),
        direction=direction,
        scenario=scenario["name"],
        backend="fallback",
        meta={
            "kind": "highway",
            "length_m": L,
            "lane_width_m": lane_w,
            "median_width_m": median,
            "lanes_per_direction": lanes_per_dir,
            "directions": int(geo["directions"]),
            "density_veh_km_lane": float(density_veh_km_lane),
            "target_per_lane": target_per_lane,
            "speed_factor": speed_factor,
            "headway_factor": headway_factor,
            "car_following": "Krauss1998+dawdling",
            "warmup_s": warmup,
        },
    )


# ---------------------------------------------------------------------------
# Urban grid
# ---------------------------------------------------------------------------
def generate_grid_trace(
    scenario: dict[str, Any],
    density_veh_km_lane: float,
    rng: np.random.Generator,
    *,
    duration_s: float | None = None,
    speed_factor: float = 1.0,
    headway_factor: float = 1.0,
) -> Trace:
    """Manhattan grid with fixed-time two-phase signals.

    Directed edges are independent car-following queues. At an intersection a
    vehicle picks a random outgoing edge (no U-turn) and is held at the stop
    line while its phase is red, modelled as a stationary virtual leader.
    """
    geo, veh, sim = scenario["geometry"], scenario["vehicles"], scenario["simulation"]
    rows, cols = int(geo["grid_rows"]), int(geo["grid_cols"])
    block = float(geo["block_length_m"])
    dt = float(sim["timestep_s"])
    duration = float(sim["duration_s"] if duration_s is None else duration_s)
    n_rec = int(round(duration / dt))
    n_warm = int(round(float(sim["warmup_s"]) / dt))
    cycle = float(geo.get("cycle_time_s", 60.0))
    green_frac = float(geo.get("green_fraction", 0.5))
    signalised = bool(geo.get("signalised", True))
    lane_w = float(geo["lane_width_m"])

    fleet = FleetParams.from_config(veh)
    tau = float(veh["reaction_time_s"]) * headway_factor
    sigma = float(veh["driver_imperfection"])
    min_gap = float(veh["min_gap_m"]) * headway_factor
    speed_dev = float(veh["speed_dev"])

    # --- build the directed edge list ---------------------------------------
    # Node (r, c) sits at (c * block, r * block).
    def node_xy(r: int, c: int) -> tuple[float, float]:
        return c * block, r * block

    edges: list[dict[str, Any]] = []
    edge_index: dict[tuple[tuple[int, int], tuple[int, int]], int] = {}
    for r in range(rows):
        for c in range(cols):
            for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                r2, c2 = r + dr, c + dc
                if not (0 <= r2 < rows and 0 <= c2 < cols):
                    continue
                x0, y0 = node_xy(r, c)
                x1, y1 = node_xy(r2, c2)
                eid = len(edges)
                edge_index[((r, c), (r2, c2))] = eid
                edges.append({
                    "id": eid,
                    "from": (r, c),
                    "to": (r2, c2),
                    "p0": (x0, y0),
                    "p1": (x1, y1),
                    "length": block,
                    "heading": math.atan2(y1 - y0, x1 - x0),
                    "axis": "h" if dr == 0 else "v",  # horizontal / vertical street
                })
    n_edges = len(edges)
    # Successors, excluding the U-turn back down the same street.
    succ: list[list[int]] = []
    for e in edges:
        opts = [
            edge_index[(e["to"], nxt)]
            for nxt in (
                (e["to"][0] + dr, e["to"][1] + dc) for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0))
            )
            if (e["to"], nxt) in edge_index and nxt != e["from"]
        ]
        succ.append(opts or [edge_index[(e["to"], e["from"])]])

    total_lane_km = n_edges * block / 1000.0
    n_target = max(1, int(round(density_veh_km_lane * total_lane_km)))

    pool = _SlotPool(capacity=max(64, n_target * 3))
    streams = [_LaneStream() for _ in range(n_edges)]
    routes: dict[int, list[int]] = {}

    def _spawn(edge_id: int, s: float) -> int:
        a = _sample_fleet(1, fleet, speed_dev, speed_factor, rng)
        slot = pool.add(
            pos=s,
            vel=float(a["v_desired"][0]) * 0.5,
            v_desired=float(a["v_desired"][0]),
            length_m=float(a["length_m"][0]),
            accel=float(a["accel"][0]),
            decel=float(a["decel"][0]),
            vclass=str(a["vclass"][0]),
            lane=edge_id,
            direction=0,
            edge=edge_id,
        )
        routes[slot] = [edge_id]
        return slot

    for _ in range(n_target):
        eid = int(rng.integers(n_edges))
        streams[eid].order.append(_spawn(eid, s=float(rng.uniform(0, block))))
    for st in streams:
        st.order.sort(key=lambda i: -pool.pos[i])

    rec_x, rec_y, rec_vx, rec_vy, rec_head, rec_edge, rec_act = [], [], [], [], [], [], []

    for step in range(n_warm + n_rec):
        t = step * dt
        h_green = (t % cycle) < green_frac * cycle if signalised else True
        transfers: list[tuple[int, int, float]] = []  # (slot, from_edge, overshoot)

        for eid, stream in enumerate(streams):
            order = stream.order
            if not order:
                continue
            idx = np.asarray(order, dtype=np.intp)
            pos, vel = pool.pos[idx], pool.vel[idx]
            gap = np.empty_like(pos)
            v_lead = np.empty_like(pos)
            e = edges[eid]
            green = h_green if e["axis"] == "h" else (not h_green)
            if signalised and not green:
                # Stop line acts as a stationary leader at the end of the edge.
                gap[0] = max(e["length"] - pos[0] - min_gap, 0.0)
                v_lead[0] = 0.0
            else:
                gap[0] = np.inf
                v_lead[0] = 0.0
            if idx.size > 1:
                gap[1:] = pos[:-1] - pos[1:] - pool.length_m[idx[:-1]] - min_gap
                v_lead[1:] = vel[:-1]
            new_vel = _krauss_velocity(
                vel, v_lead, np.maximum(gap, 0.0),
                pool.v_desired[idx], pool.accel[idx], pool.decel[idx],
                tau, sigma, dt, rng,
            )
            pool.vel[idx] = new_vel
            pool.pos[idx] = pos + new_vel * dt
            while order and pool.pos[order[0]] >= e["length"]:
                slot = order.pop(0)
                transfers.append((slot, eid, float(pool.pos[slot] - e["length"])))

        for slot, from_edge, overshoot in transfers:
            nxt = int(rng.choice(succ[from_edge]))
            pool.edge[slot] = nxt
            pool.lane[slot] = nxt
            pool.pos[slot] = min(overshoot, edges[nxt]["length"] - 1.0)
            routes[slot].append(nxt)
            # Joining at the back of the receiving queue keeps the invariant
            # that `order` is sorted downstream-first; a vehicle entering just
            # past the stop line is behind everyone already on that edge.
            streams[nxt].order.append(slot)
            streams[nxt].order.sort(key=lambda i: -pool.pos[i])

        if step >= n_warm:
            n = pool.capacity
            xs = np.full(n, np.nan, dtype=np.float32)
            ys = np.full(n, np.nan, dtype=np.float32)
            vxs = np.zeros(n, dtype=np.float32)
            vys = np.zeros(n, dtype=np.float32)
            hds = np.zeros(n, dtype=np.float32)
            act = np.zeros(n, dtype=bool)
            for eid, stream in enumerate(streams):
                if not stream.order:
                    continue
                idx = np.asarray(stream.order, dtype=np.intp)
                e = edges[eid]
                (x0, y0), (x1, y1) = e["p0"], e["p1"]
                frac = np.clip(pool.pos[idx] / e["length"], 0.0, 1.0)
                xs[idx] = x0 + (x1 - x0) * frac
                ys[idx] = y0 + (y1 - y0) * frac
                hd = e["heading"]
                vxs[idx] = pool.vel[idx] * math.cos(hd)
                vys[idx] = pool.vel[idx] * math.sin(hd)
                hds[idx] = hd
                act[idx] = True
                # Offset right-hand traffic off the street centreline.
                xs[idx] += math.sin(hd) * lane_w * 0.5
                ys[idx] -= math.cos(hd) * lane_w * 0.5
            rec_x.append(xs); rec_y.append(ys); rec_vx.append(vxs)
            rec_vy.append(vys); rec_head.append(hds); rec_act.append(act)
            rec_edge.append(pool.edge.copy())

    n_slots, T = pool.n, len(rec_act)

    def _stack(chunks: list[np.ndarray], dtype: Any) -> np.ndarray:
        out = np.zeros((T, n_slots), dtype=dtype)
        for t, c in enumerate(chunks):
            out[t] = c[:n_slots]
        return out

    return Trace(
        dt=dt,
        x=_stack(rec_x, np.float32), y=_stack(rec_y, np.float32),
        vx=_stack(rec_vx, np.float32), vy=_stack(rec_vy, np.float32),
        heading=_stack(rec_head, np.float32),
        lane=_stack(rec_edge, np.int16),
        active=_stack(rec_act, bool),
        vehicle_ids=np.array([f"v{i}" for i in range(n_slots)], dtype="U16"),
        vclass=pool.vclass[:n_slots].astype("U8"),
        length_m=pool.length_m[:n_slots].astype(np.float32),
        direction=np.zeros(n_slots, dtype=np.int8),
        routes=[tuple(str(e) for e in routes.get(i, ())) for i in range(n_slots)],
        scenario=scenario["name"],
        backend="fallback",
        meta={
            "kind": "grid",
            "grid_rows": rows,
            "grid_cols": cols,
            "block_length_m": block,
            "n_edges": n_edges,
            "total_lane_km": total_lane_km,
            "density_veh_km_lane": float(density_veh_km_lane),
            "n_target": n_target,
            "cycle_time_s": cycle,
            "green_fraction": green_frac,
            "speed_factor": speed_factor,
            "headway_factor": headway_factor,
            "car_following": "Krauss1998+dawdling",
            "edges": [
                {"id": e["id"], "p0": e["p0"], "p1": e["p1"], "axis": e["axis"],
                 "length": e["length"], "heading": e["heading"]}
                for e in edges
            ],
        },
    )


def generate_fallback_trace(
    scenario: dict[str, Any], density_veh_km_lane: float, rng: np.random.Generator, **kw: Any
) -> Trace:
    """Dispatch on ``scenario['kind']``."""
    kind = scenario.get("kind", "highway")
    if kind == "highway":
        return generate_highway_trace(scenario, density_veh_km_lane, rng, **kw)
    if kind == "grid":
        return generate_grid_trace(scenario, density_veh_km_lane, rng, **kw)
    raise ValueError(f"Unknown scenario kind {kind!r}")
