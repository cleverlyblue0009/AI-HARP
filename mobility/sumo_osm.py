"""SUMO demand and trace post-processing for real OSM networks (and SUMO grids).

Called from :func:`mobility.sumo_runner.generate_sumo_trace`:

* :func:`write_osm_routes` -- demand on an imported OSM network. Highway: the
  typed main-road edges are chained into one directed route per carriageway and
  loaded with the same free-flow flows as the synthetic corridor. Grid: SUMO's
  ``randomTrips.py`` with an insertion density sized from the commanded density.
* :func:`postprocess_trace` -- the mapping onto our geometry model (user
  decision: straight real road + projection, rotated real grid), and the grid
  metadata every SUMO grid trace needs: the ``edges`` list for hazard placement
  and, for synthetic grids, ``grid_rows`` / ``grid_cols`` / ``block_length_m``,
  without which ``sim.buildings.build_building_grid`` silently falls back to a
  5 x 5 grid.
"""

from __future__ import annotations

import dataclasses
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from mobility.osm_geometry import (
    grid_edge_records, project_highway_trace, rotate_trace, straightest_window,
)
from mobility.trace import Trace


# ---------------------------------------------------------------------------
# Highway routes
# ---------------------------------------------------------------------------
def _bearing(e: dict[str, Any]) -> float:
    (x0, y0), (x1, y1) = e["shape"][0], e["shape"][-1]
    return math.atan2(y1 - y0, x1 - x0)


def highway_chains(edges: Sequence[dict[str, Any]]) -> list[list[str]]:
    """Directed chains of main-road edges, one per carriageway, longest first.

    A successor starts where an edge ends and is not its U-turn reverse; with
    several, the most collinear continues the chain.
    """
    by_from: dict[str, list[dict[str, Any]]] = {}
    for e in edges:
        by_from.setdefault(e["from"], []).append(e)

    def successors(e):
        return [s for s in by_from.get(e["to"], []) if s["to"] != e["from"]]

    has_pred = {s["sumo_id"] for e in edges for s in successors(e)}
    chains = []
    for start in (e for e in edges if e["sumo_id"] not in has_pred):
        chain, seen, cur = [start["sumo_id"]], {start["sumo_id"]}, start
        while True:
            nxt = [s for s in successors(cur) if s["sumo_id"] not in seen]
            if not nxt:
                break
            b = _bearing(cur)
            cur = min(nxt, key=lambda s: abs((_bearing(s) - b + math.pi) % (2 * math.pi) - math.pi))
            chain.append(cur["sumo_id"])
            seen.add(cur["sumo_id"])
        chains.append(chain)
    length = {e["sumo_id"]: e["length"] for e in edges}
    return sorted(chains, key=lambda c: -sum(length[i] for i in c))


def _typed(scenario: dict[str, Any], net_edges: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    types = scenario["sumo"].get("osm_highway_types", ["primary", "trunk"])
    return [e for e in net_edges if any(t in e["type"] for t in types)]


def write_osm_routes(scenario: dict[str, Any], density: float, seed: int, work: Path,
                     speed_factor: float, net: Path, net_edges: Sequence[dict[str, Any]],
                     tools: Any, warmup_s: float) -> Path:
    from mobility.sumo_runner import _vtype_xml, _write

    geo, sim = scenario["geometry"], scenario["simulation"]
    classes = scenario["vehicles"]["classes"]
    v_mean_kmh = float(np.average([(c["speed_kmh_min"] + c["speed_kmh_max"]) / 2
                                   for c in classes.values()],
                                  weights=[c["share"] for c in classes.values()])) * speed_factor
    duration = float(sim["duration_s"]) + float(warmup_s)

    if scenario.get("kind") == "grid":
        # Vehicles in the network ~ insertion rate x mean trip time. Trip length
        # is taken as the mean Manhattan half-span of the network; the achieved
        # density is measured afterwards and reported (_check_density).
        xs = [p[0] for e in net_edges for p in e["shape"]]
        ys = [p[1] for e in net_edges for p in e["shape"]]
        trip_km = max(((max(xs) - min(xs)) + (max(ys) - min(ys))) / 2.0 / 1000.0, 0.2)
        lanes = float(np.average([e["lanes"] for e in net_edges], weights=[e["length"] for e in net_edges]))
        insertion = density * lanes * v_mean_kmh / trip_km
        routes = work / "demand.rou.xml"
        home = Path(tools.sumo_home) if getattr(tools, "sumo_home", None) else Path(tools.sumo).parent.parent
        cmd = [sys.executable, str(home / "tools" / "randomTrips.py"), "-n", str(net),
               "-o", str(work / "trips.trips.xml"), "-r", str(routes), "-b", "0",
               "-e", f"{duration:.1f}", "--insertion-density", f"{insertion:.3f}",
               "--fringe-factor", "5", "--validate", "-s", str(int(seed)),
               "--trip-attributes", 'departLane="best" departSpeed="max"']
        subprocess.run(cmd, cwd=work, check=True, capture_output=True, text=True)
        return routes

    chains = highway_chains(_typed(scenario, net_edges))[:2]
    if len(chains) < 2:
        raise ValueError(f"expected two carriageway chains on the OSM highway, found {chains}")
    q_veh_h = density * v_mean_kmh * int(geo["lanes_per_direction"])
    body = [_vtype_xml(scenario, speed_factor)]
    for k, chain in enumerate(chains):
        body.append(f'  <route id="r{k}" edges="{" ".join(chain)}"/>')
        for name, c in classes.items():
            body.append(f'  <flow id="f{k}_{name}" route="r{k}" type="{name}" begin="0" '
                        f'end="{duration:.1f}" vehsPerHour="{q_veh_h * c["share"]:.1f}" '
                        f'departLane="best" departSpeed="max"/>')
    return _write(work / "demand.rou.xml", "<routes>\n" + "\n".join(body) + "\n</routes>")


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------
def postprocess_trace(trace: Trace, scenario: dict[str, Any], net_source: str,
                      net_edges: Sequence[dict[str, Any]]) -> tuple[Trace, float | None]:
    """Map a SUMO trace onto the geometry model; returns it and its lane-km.

    ``lane_km`` is ``None`` when the scenario geometry already defines it.
    """
    geo = scenario["geometry"]
    if scenario.get("kind") == "grid":
        osm = net_source == "osm"
        theta = float(geo.get("grid_rotation_deg", 0.0)) if osm else 0.0
        records, stats = grid_edge_records(net_edges, theta)
        if osm:
            trace = rotate_trace(trace, theta)
        meta = {**trace.meta, "kind": "grid", "edges": records, "grid_edge_stats": stats}
        lane_km = None
        if osm:
            lanes = {e["sumo_id"]: int(e.get("lanes", 1)) for e in net_edges}
            lane_km = sum(r["length"] * lanes.get(r["sumo_id"], 1) for r in records) / 1000.0
        else:
            meta.update(grid_rows=int(geo["grid_rows"]), grid_cols=int(geo["grid_cols"]),
                        block_length_m=float(geo["block_length_m"]))
        return dataclasses.replace(trace, meta=meta), lane_km

    if net_source != "osm":
        return trace, None
    typed = _typed(scenario, net_edges)
    if not typed:
        raise ValueError("no main-road edges of the configured osm_highway_types in the network")
    longest = max(typed, key=lambda e: e["length"])
    a, b, dev = straightest_window(longest["shape"], float(geo["length_m"]))
    trace = project_highway_trace(trace, a, b, max_dev_m=dev)
    lanes, dirs = int(geo["lanes_per_direction"]), int(geo.get("directions", 2))
    trace.meta.update(lanes_per_direction=lanes, lane_width_m=float(geo["lane_width_m"]))
    return trace, float(trace.meta["length_m"]) / 1000.0 * lanes * dirs
