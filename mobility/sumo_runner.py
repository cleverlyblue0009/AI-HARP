"""SUMO scenario generation and execution (Phase 1, preferred backend).

Detection is deliberately strict and the outcome is always logged: a run must
never be ambiguous about whether its mobility came from SUMO or from the
pure-Python fallback.

Network sources, in priority order:

1. ``scenario['sumo']['osm_extract']`` -- a real OSM extract, imported with
   ``netconvert --osm-files``. This is what the paper should use.
2. Synthetic: a straight two-lane corridor (highway) or ``netgenerate --grid``
   (urban). A runnable stand-in so the pipeline works before the extract is
   dropped in; traces built this way carry ``network_source='synthetic'``.

Warm-up, and why it is computed rather than configured
------------------------------------------------------
The fallback generator *pre-places* vehicles along the corridor, so it is at
steady state from step 0. SUMO cannot do that: it injects vehicles at the
boundary, and the corridor is not full until the first of them has driven its
whole length. Recording before that yields a density far below the commanded
one -- silently, because nothing downstream re-checks density.

Measured on first contact: a 2 km corridor at 20 veh/km/lane with a 10 s warm-up
recorded 18.6 concurrent vehicles against an expected 76. For the real 10 km
corridor the transit time is ~457 s against a configured warm-up of 20 s, so
every SUMO run would have been a fraction of its nominal density and would have
been incomparable with the fallback results.

So the warm-up is ``max(configured, corridor_transit_time * safety)``, and
:func:`_check_density` verifies the achieved density afterwards and warns
loudly if it still misses. Do not lower either.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from common.config import PROJECT_ROOT, ensure_dir
from common.logging_utils import get_logger
from mobility.fcd import parse_fcd
from mobility.trace import Trace

logger = get_logger("mobility.sumo")

BUILD_DIR = PROJECT_ROOT / "cache" / "sumo_build"
KMH_TO_MS = 1.0 / 3.6

#: Multiple of the corridor transit time to warm up for. 1.5 leaves margin
#: for slow vehicles (trucks traverse more slowly than the fleet mean) and
#: for the queue that forms behind them.
WARMUP_TRANSITS = 1.5

#: Achieved density may miss the command by this fraction before we warn.
DENSITY_TOLERANCE = 0.25

#: Grid demand is an estimate (randomTrips insertion rate x an assumed trip
#: length), so a grid trace that misses by more than this is re-run once with
#: its demand scaled by commanded / achieved. On urban_nlos d=20 the estimate
#: alone gave 33.5 veh/km/lane (+67%). Corridor flows are exact and never re-run.
CALIBRATION_TOLERANCE = 0.10

#: SUMO runs per grid trace at most: the estimate, one proportional correction,
#: then one interpolation between the two measurements.
MAX_DEMAND_ATTEMPTS = 3


@dataclass(frozen=True)
class SumoTools:
    """Resolved paths to the SUMO binaries we need."""

    sumo: str
    netconvert: str
    netgenerate: str
    sumo_home: str | None


def find_sumo() -> SumoTools | None:
    """Locate a usable SUMO installation, or return None."""
    sumo_home = os.environ.get("SUMO_HOME")
    search: list[Path] = []
    if sumo_home:
        search.append(Path(sumo_home) / "bin")

    def _which(name: str) -> str | None:
        for d in search:
            for ext in ("", ".exe"):
                cand = d / f"{name}{ext}"
                if cand.exists():
                    return str(cand)
        return shutil.which(name)

    sumo, netconvert, netgenerate = _which("sumo"), _which("netconvert"), _which("netgenerate")
    if not (sumo and netconvert and netgenerate):
        return None
    return SumoTools(sumo=sumo, netconvert=netconvert, netgenerate=netgenerate,
                     sumo_home=sumo_home)


def _run(cmd: list[str], cwd: Path) -> None:
    logger.debug("exec: %s", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"SUMO command failed ({proc.returncode}): {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )


def _fleet_mean_speed_ms(scenario: dict[str, Any], speed_factor: float) -> float:
    classes = scenario["vehicles"]["classes"]
    return float(np.average(
        [(c["speed_kmh_min"] + c["speed_kmh_max"]) / 2 for c in classes.values()],
        weights=[c["share"] for c in classes.values()],
    )) * KMH_TO_MS * speed_factor


def required_warmup_s(scenario: dict[str, Any], speed_factor: float = 1.0) -> float:
    """Time for a vehicle to traverse the corridor, times a safety factor.

    Below this the corridor is still filling when recording starts. Grid
    scenarios circulate rather than traverse, so one block-row is used as the
    characteristic length.
    """
    geo = scenario["geometry"]
    v = max(_fleet_mean_speed_ms(scenario, speed_factor), 1e-6)
    if geo.get("route_length_m"):
        # Real networks: vehicles enter at the route's start, not the window's.
        span = float(geo["route_length_m"])
    elif scenario.get("kind") == "grid":
        span = float(geo.get("block_length_m", 200.0)) * float(geo.get("grid_cols", 5))
    else:
        span = float(geo.get("length_m", 0.0))
    return span / v * WARMUP_TRANSITS


def _lane_km(scenario: dict[str, Any], override: float | None = None) -> float:
    """Lane-kilometres the density is measured over (override: real networks)."""
    if override is not None:
        return float(override)
    geo = scenario["geometry"]
    if scenario.get("kind") == "grid":
        return (float(geo.get("block_length_m", 200.0))
                * float(geo.get("grid_rows", 5)) * float(geo.get("grid_cols", 5))
                * 4.0 / 1000.0)
    return (float(geo.get("length_m", 0.0)) / 1000.0
            * float(geo.get("lanes_per_direction", 1))
            * float(geo.get("directions", 2)))


def mask_to_corridor(trace: Trace, length_m: float) -> Trace:
    """Vehicles on the exit sections (x < 0 or x > length) are not in the scenario."""
    import dataclasses

    x = np.asarray(trace.x, dtype=float)
    inside = np.asarray(trace.active, bool) & np.isfinite(x) & (x >= 0.0) & (x <= length_m)
    return dataclasses.replace(trace, active=inside,
                               x=np.where(inside, trace.x, np.nan).astype(np.float32),
                               y=np.where(inside, trace.y, np.nan).astype(np.float32))


def _achieved_density(trace: Trace, scenario: dict[str, Any], override: float | None = None) -> float:
    lk = _lane_km(scenario, override)
    return float(trace.active.sum(axis=1).mean()) / lk if lk > 0 else float("nan")


def _check_density(trace: Trace, scenario: dict[str, Any], commanded: float,
                   lane_km_override: float | None = None) -> None:
    """Warn if the achieved density misses the command.

    The guard that would have caught the warm-up bug on its own: nothing else
    downstream re-checks that SUMO actually produced the density it was asked
    for, and a quietly empty corridor looks like a valid result.
    """
    lane_km = _lane_km(scenario, lane_km_override)
    if lane_km <= 0:
        return
    achieved = float(trace.active.sum(axis=1).mean()) / lane_km
    trace.meta["achieved_density_veh_km_lane"] = round(achieved, 2)
    rel = abs(achieved - commanded) / max(commanded, 1e-9)
    if rel > DENSITY_TOLERANCE:
        logger.warning(
            "SUMO density MISS: commanded %.3g veh/km/lane, achieved %.3g (%.0f%% off). "
            "Usually too short a warm-up (need >= %.0f s for this corridor) or demand "
            "that cannot be inserted. Do not compare this trace with fallback results.",
            commanded, achieved, rel * 100.0, required_warmup_s(scenario),
        )
    else:
        logger.info("SUMO density OK: commanded %.3g, achieved %.3g veh/km/lane",
                    commanded, achieved)


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def _vtype_xml(scenario: dict[str, Any], speed_factor: float) -> str:
    veh = scenario["vehicles"]
    rows = []
    for name, c in veh["classes"].items():
        vmax = c["speed_kmh_max"] * KMH_TO_MS * speed_factor
        rows.append(
            f'  <vType id="{name}" vClass="{"truck" if name == "truck" else "passenger"}" '
            f'length="{c["length_m"]}" maxSpeed="{vmax:.2f}" accel="{c["max_accel_ms2"]}" '
            f'decel="{c["max_decel_ms2"]}" sigma="{veh["driver_imperfection"]}" '
            f'tau="{veh["reaction_time_s"]}" minGap="{veh["min_gap_m"]}" '
            f'speedDev="{veh["speed_dev"]}" carFollowModel="{scenario["sumo"]["car_following_model"]}"/>'
        )
    return "\n".join(rows)


#: Speed-sign file a congested OSM highway leaves beside its routes: an
#: imported network's edge speeds cannot be rewritten as the synthetic builder
#: rewrites its own, so the equilibrium speed is imposed with a
#: variableSpeedSign over the route's lanes.
VSS_FILE = "vss.add.xml"

#: Extra gap, metres, between pre-placed congested vehicles beyond the safe
#: minimum (positions are written to 0.1 m).
INSERTION_MARGIN_M = 0.5

#: Length of the speed-limited exit section added beyond each end of a
#: congested corridor. [ASSUMED] Long enough to hold the discharge queue's head
#: outside the recorded window.
EXIT_SECTION_M = 500.0


@dataclass(frozen=True)
class CongestionPlan:
    """How a synthetic SUMO corridor is made to hold a commanded density.

    A corridor fed only from its ends cannot hold dense congestion: at rural
    d=80 SUMO stalled at ~30 veh/km/lane (user decision: pre-populate + a
    downstream bottleneck). Under Krauss car-following a steady queue at speed
    v has spacing ~ L_veh + minGap + v * tau, so the speed that holds density k
    is ``v_eq = (1000/k - L_veh - minGap) / tau``. When that is below the free
    speed the corridor is congested: vehicles are pre-placed at the commanded
    spacing, the whole corridor (and the exit sections beyond it) is limited to
    ``v_eq``, and inflow is ``k * v_eq * lanes`` inserted at ``v_eq``, so density,
    speed and flow are held together. (rural d=80: v_eq = 3.9 m/s = 14 km/h; the
    fallback's d=80 traffic measured 14.8 km/h.)

    An exit-only bottleneck was tried first and does not work: the queue spills
    back far too slowly to fill a 10 km corridor within the warm-up, and the
    pre-placed vehicles ahead of it accelerate away (rural d=80 seed 0: 40.7
    veh/km/lane at 48.3 km/h).
    """

    congested: bool
    v_eq_ms: float
    v_free_ms: float
    spacing_m: float


def congestion_plan(scenario: dict[str, Any], density: float, speed_factor: float = 1.0
                    ) -> CongestionPlan:
    veh = scenario["vehicles"]
    classes = veh["classes"]
    w = [c["share"] for c in classes.values()]
    L_veh = float(np.average([c["length_m"] for c in classes.values()], weights=w))
    v_free = _fleet_mean_speed_ms(scenario, speed_factor)
    spacing = 1000.0 / max(density, 1e-9)
    v_eq = (spacing - L_veh - float(veh["min_gap_m"])) / float(veh["reaction_time_s"])
    congested = v_eq < v_free
    return CongestionPlan(congested=bool(congested), v_eq_ms=float(max(v_eq, 0.5)),
                          v_free_ms=float(v_free), spacing_m=float(spacing))


def _build_highway_network(scenario: dict[str, Any], tools: SumoTools, work: Path,
                           plan: CongestionPlan | None = None) -> Path:
    """Synthetic straight corridor: `junctions` splits, plus exit bottlenecks when congested."""
    geo = scenario["geometry"]
    L = float(geo["length_m"])
    lanes = int(geo["lanes_per_direction"])
    n_seg = max(1, int(geo.get("junctions", 1)) + 1)
    vmax = max(c["speed_kmh_max"] for c in scenario["vehicles"]["classes"].values()) * KMH_TO_MS

    nodes = ["<nodes>"]
    edges = ["<edges>"]
    for i in range(n_seg + 1):
        kind = "priority" if 0 < i < n_seg else "unregulated"
        nodes.append(f'  <node id="n{i}" x="{i * L / n_seg:.2f}" y="0.0" type="{kind}"/>')
    congested = plan is not None and plan.congested
    edge_speed = plan.v_eq_ms if congested else vmax
    for i in range(n_seg):
        edges.append(
            f'  <edge id="e{i}" from="n{i}" to="n{i+1}" numLanes="{lanes}" '
            f'speed="{edge_speed:.2f}" priority="2"/>'
        )
        edges.append(
            f'  <edge id="-e{i}" from="n{i+1}" to="n{i}" numLanes="{lanes}" '
            f'speed="{edge_speed:.2f}" priority="2"/>'
        )
    if congested:
        nodes.append(f'  <node id="nout" x="{L + EXIT_SECTION_M:.2f}" y="0.0" type="unregulated"/>')
        nodes.append(f'  <node id="nin" x="{-EXIT_SECTION_M:.2f}" y="0.0" type="unregulated"/>')
        edges.append(f'  <edge id="xf" from="n{n_seg}" to="nout" numLanes="{lanes}" '
                     f'speed="{plan.v_eq_ms:.2f}" priority="2"/>')
        edges.append(f'  <edge id="xb" from="n0" to="nin" numLanes="{lanes}" '
                     f'speed="{plan.v_eq_ms:.2f}" priority="2"/>')
    nodes.append("</nodes>")
    edges.append("</edges>")

    _write(work / "corridor.nod.xml", "\n".join(nodes))
    _write(work / "corridor.edg.xml", "\n".join(edges))
    net = work / "net.net.xml"
    _run([tools.netconvert, "-n", "corridor.nod.xml", "-e", "corridor.edg.xml",
          "-o", net.name, "--no-turnarounds", "true"], cwd=work)
    return net


def _build_grid_network(scenario: dict[str, Any], tools: SumoTools, work: Path) -> Path:
    geo = scenario["geometry"]
    net = work / "net.net.xml"
    cmd = [
        tools.netgenerate, "--grid",
        "--grid.x-number", str(geo["grid_cols"]),
        "--grid.y-number", str(geo["grid_rows"]),
        "--grid.length", str(geo["block_length_m"]),
        "--default.lanenumber", str(geo["lanes_per_direction"]),
        "--default.speed", "13.89",
        "-o", net.name,
    ]
    if geo.get("signalised", True):
        cmd += ["--tls.guess", "true", "--tls.cycle.time", str(int(geo.get("cycle_time_s", 60)))]
    _run(cmd, cwd=work)
    return net


def _build_network(scenario: dict[str, Any], tools: SumoTools, work: Path,
                   plan: CongestionPlan | None = None) -> tuple[Path, str]:
    osm = scenario["sumo"].get("osm_extract")
    if osm:
        osm_path = Path(osm)
        if not osm_path.is_absolute():
            osm_path = PROJECT_ROOT / osm_path
        if not osm_path.exists():
            raise FileNotFoundError(f"osm_extract {osm_path} not found.")
        net = work / "net.net.xml"
        _run([tools.netconvert, "--osm-files", str(osm_path), "-o", net.name,
              "--geometry.remove", "--roundabouts.guess", "--ramps.guess",
              "--junctions.join", "--tls.guess-signals", "--tls.discard-simple"], cwd=work)
        return net, "osm"
    if scenario.get("kind") == "grid":
        return _build_grid_network(scenario, tools, work), "synthetic"
    return _build_highway_network(scenario, tools, work, plan), "synthetic"


def _write_routes(
    scenario: dict[str, Any], density: float, seed: int, work: Path, speed_factor: float
) -> Path:
    """Demand as per-direction flows sized to hit the commanded density.

    q [veh/h] = k [veh/km/lane] * v_free [km/h] * lanes, the fundamental
    relation of traffic flow in the free-flow regime.
    """
    geo = scenario["geometry"]
    sim = scenario["simulation"]
    lanes = int(geo["lanes_per_direction"])
    classes = scenario["vehicles"]["classes"]
    v_mean_kmh = float(
        np.average(
            [(c["speed_kmh_min"] + c["speed_kmh_max"]) / 2 for c in classes.values()],
            weights=[c["share"] for c in classes.values()],
        )
    ) * speed_factor
    duration = float(sim["duration_s"]) + max(float(sim["warmup_s"]),
                                              required_warmup_s(scenario, speed_factor))
    q_veh_h = density * v_mean_kmh * lanes

    n_seg = max(1, int(geo.get("junctions", 1)) + 1)
    body = [_vtype_xml(scenario, speed_factor)]
    if scenario.get("kind") == "grid":
        # randomTrips.py is the right tool here; when it is unavailable we fall
        # back to flows over explicit through-routes generated by SUMO itself.
        body.append(
            f'  <flow id="fgrid" begin="0" end="{duration:.1f}" vehsPerHour="{q_veh_h:.1f}" '
            f'type="car" from="A0A1" to="A1A2" departLane="best" departSpeed="max"/>'
        )
    else:
        plan = congestion_plan(scenario, density, speed_factor)
        seg_len = float(geo["length_m"]) / n_seg
        rng = np.random.default_rng(int(seed) + 7919)
        names = list(classes)
        shares = np.array([classes[n]["share"] for n in names], dtype=float)
        vehicles: list[str] = []
        for d, prefix in ((1, ""), (-1, "-")):
            order = list(range(n_seg)) if d == 1 else list(reversed(range(n_seg)))
            exit_edge = ["xf" if d == 1 else "xb"] if plan.congested else []
            route_edges = [f"{prefix}e{i}" for i in order] + exit_edge
            body.append(f'  <route id="r{d}" edges="{" ".join(route_edges)}"/>')
            # Congested: q = k * v_eq * lanes, inserted at v_eq, so the inflow
            # arrives at the commanded density rather than at free-flow spacing.
            q_dir = (density * plan.v_eq_ms * 3.6 * lanes) if plan.congested else q_veh_h
            depart_speed = f"{plan.v_eq_ms:.2f}" if plan.congested else "max"
            for name, c in classes.items():
                body.append(
                    f'  <flow id="f{d}_{name}" route="r{d}" type="{name}" begin="0" '
                    f'end="{duration:.1f}" vehsPerHour="{q_dir * c["share"]:.1f}" '
                    f'departLane="best" departSpeed="{depart_speed}"/>'
                )
            if plan.congested:
                # Pre-place the corridor travelling at the equilibrium speed, so
                # the queue exists from t = 0. Each vehicle sits its LEADER's
                # length + minGap + v_eq * tau behind it: a uniform 12.5 m (the
                # fleet-average spacing) cannot fit a 12 m truck, SUMO refused
                # those insertions and rural d=80 recorded 22.9 veh/km/lane.
                # `s` is the front position along the direction of travel,
                # laid out from the downstream end backwards.
                veh_cfg = scenario["vehicles"]
                # +0.5 m: departPos is written to 0.1 m, and a gap exactly at the
                # safe minimum was still refused.
                gap = (float(veh_cfg["min_gap_m"]) + plan.v_eq_ms * float(veh_cfg["reaction_time_s"])
                       + INSERTION_MARGIN_M)
                L_corr = float(geo["length_m"])
                for lane in range(lanes):
                    s, k = L_corr - 1.0, 0
                    while True:
                        cls = names[int(rng.choice(len(names), p=shares / shares.sum()))]
                        length = float(classes[cls]["length_m"])
                        seg = min(int(s // seg_len), n_seg - 1)
                        if s - seg * seg_len < length:
                            # SUMO only inserts a vehicle wholly on its edge: one
                            # that would straddle a junction starts at the end of
                            # the previous edge instead.
                            s = seg * seg_len - 0.5
                            seg -= 1
                        if seg < 0 or s - length < 0.0:
                            break
                        pos = s - seg * seg_len
                        edge_idx = order.index(seg) if d == 1 else order.index(n_seg - 1 - seg)
                        rest = [f"{prefix}e{i}" for i in order[edge_idx:]] + exit_edge
                        vehicles.append(
                            f'  <vehicle id="p{d}_{k}_{lane}" type="{cls}" depart="0" '
                            f'departPos="{pos:.1f}" departLane="{lane}" '
                            f'departSpeed="{plan.v_eq_ms:.2f}"><route edges="{" ".join(rest)}"/></vehicle>'
                        )
                        s -= float(classes[cls]["length_m"]) + gap
                        k += 1
        body.extend(vehicles)
    return _write(work / "demand.rou.xml", "<routes>\n" + "\n".join(body) + "\n</routes>")


def generate_sumo_trace(
    scenario: dict[str, Any],
    density_veh_km_lane: float,
    seed: int,
    *,
    tools: SumoTools,
    speed_factor: float = 1.0,
    keep_fcd: bool = False,
    tag: str = "",
) -> Trace:
    """Build the network + demand, run SUMO, and parse the FCD export."""
    sim, scfg = scenario["simulation"], scenario["sumo"]
    # The build directory must separate everything that changes the run, not just
    # (scenario, density, seed): two weathers of one cell differ only in their
    # speed/headway factors, and with parallel jobs they ran SUMO in the SAME
    # directory, clobbering each other's net/routes/fcd (WinError 32 killed the
    # sparse SUMO sweep 80 s in). `tag` carries the trace cache key, which
    # already distinguishes weather, backend and pipeline version.
    work = ensure_dir(BUILD_DIR / (f"{scenario['name']}_d{density_veh_km_lane:g}_s{seed}"
                                   + (f"_{tag}" if tag else "")))

    configured_warmup = float(sim["warmup_s"])
    needed = required_warmup_s(scenario, speed_factor)
    warmup = max(configured_warmup, needed)
    if warmup > configured_warmup:
        logger.info(
            "SUMO warm-up raised %.0f s -> %.0f s (one corridor transit x %.1f); "
            "below this the corridor is still filling when recording starts.",
            configured_warmup, warmup, WARMUP_TRANSITS,
        )

    # Highways -- synthetic or real -- are held at the commanded density; grids
    # are fed by randomTrips and calibrated from their measured density instead.
    plan = (congestion_plan(scenario, density_veh_km_lane, speed_factor)
            if scenario.get("kind") != "grid" else None)
    net, net_source = _build_network(scenario, tools, work, plan)
    # Real OSM networks and every SUMO grid need the network's edges: OSM for
    # demand and projection, grids for the hazard's edge list.
    net_edges = None
    if net_source == "osm" or scenario.get("kind") == "grid":
        from mobility.osm_geometry import read_net_edges

        home = tools.sumo_home or str(Path(tools.sumo).resolve().parent.parent)
        net_edges = read_net_edges(str(Path(work) / Path(net).name), home)
    # Every SUMO grid uses randomTrips demand. The synthetic grid's single
    # A0A1 -> A1A2 flow reached 0.25 veh/km/lane against a commanded 20 on
    # urban_nlos (99% off): no vehicle ever reached the hazard and RWCR was 0.
    osm_or_grid = net_source == "osm" or scenario.get("kind") == "grid"
    duration = float(sim["duration_s"]) + warmup
    fcd = work / "fcd.xml"
    demand_scale, attempts = 1.0, []

    for attempt in range(MAX_DEMAND_ATTEMPTS):
        if osm_or_grid:
            from mobility.sumo_osm import write_osm_routes

            routes = write_osm_routes(scenario, density_veh_km_lane, seed, work, speed_factor,
                                      Path(work) / Path(net).name, net_edges, tools, warmup,
                                      demand_scale=demand_scale, plan=plan)
        else:
            routes = _write_routes(scenario, density_veh_km_lane, seed, work, speed_factor)

        # A congested OSM highway cannot have its edge speeds rewritten the way
        # the synthetic builder does, so write_osm_routes leaves a speed-sign
        # file beside the routes instead.
        vss = work / VSS_FILE
        extra = f'\n    <additional-files value="{VSS_FILE}"/>' if vss.exists() else ""
        _write(work / "run.sumocfg", f"""<configuration>
  <input>
    <net-file value="{Path(net).name}"/>
    <route-files value="{Path(routes).name}"/>{extra}
  </input>
  <time>
    <begin value="0"/>
    <end value="{duration:.1f}"/>
    <step-length value="{scfg['step_length']}"/>
  </time>
  <processing>
    <max-depart-delay value="10"/>
    <max-num-vehicles value="-1"/>
    <time-to-teleport value="-1"/>
    <ignore-route-errors value="true"/>
  </processing>
  <random_number>
    <seed value="{int(seed) + int(scfg.get('seed_offset', 0))}"/>
  </random_number>
</configuration>""")

        logger.info("Running SUMO (%s network, density=%g veh/km/lane, seed=%d, demand x%.3f)",
                    net_source, density_veh_km_lane, seed, demand_scale)
        # NOTE: the FCD sampling period is `--device.fcd.period`, NOT
        # `--fcd-output.period` (which does not exist and makes SUMO exit 1).
        # Verified against `sumo --help` for 1.19.0.
        # Record FCD only after the warm-up: on the 16.5 km US-50 route the full
        # export was 933 MB, ~90% of it warm-up that parse_fcd discards anyway.
        _run([tools.sumo, "-c", "run.sumocfg", "--fcd-output", fcd.name,
              "--device.fcd.begin", f"{warmup:.1f}",
              "--device.fcd.period", str(scfg["fcd_period"]), "--no-step-log", "true",
              "--no-warnings", "true"], cwd=work)

        trace = parse_fcd(
            fcd,
            scenario_name=scenario["name"],
            dt=float(sim["timestep_s"]),
            warmup_s=warmup,
            meta={
                "kind": scenario.get("kind", "highway"),
                "length_m": float(scenario["geometry"].get("length_m", 0.0)),
                "lane_width_m": float(scenario["geometry"]["lane_width_m"]),
                "median_width_m": float(scenario["geometry"].get("median_width_m", 0.0)),
                "density_veh_km_lane": float(density_veh_km_lane),
                "network_source": net_source,
                "speed_factor": speed_factor,
                "car_following": scfg["car_following_model"],
            },
        )
        trace.meta["warmup_s"] = warmup
        if plan is not None:
            trace = mask_to_corridor(trace, float(scenario["geometry"]["length_m"]))
            trace.meta["congestion_plan"] = {**plan.__dict__, "exit_section_m": EXIT_SECTION_M
                                             if plan.congested else 0.0}
        lane_km = None
        if net_edges is not None:
            from mobility.sumo_osm import postprocess_trace

            trace, lane_km = postprocess_trace(trace, scenario, net_source, net_edges)
        achieved = _achieved_density(trace, scenario, lane_km)
        attempts.append({"demand_scale": round(demand_scale, 4),
                         "achieved_density_veh_km_lane": round(achieved, 3)})
        miss = abs(achieved - density_veh_km_lane) / max(density_veh_km_lane, 1e-9)
        if not (scenario.get("kind") == "grid" and attempt < MAX_DEMAND_ATTEMPTS - 1
                and achieved > 0 and miss > CALIBRATION_TOLERANCE):
            break
        if len(attempts) < 2:
            demand_scale *= density_veh_km_lane / achieved
        else:
            # Density is not proportional to demand once the grid congests (one
            # proportional step took urban_nlos d=20 from 33.5 to 16.4):
            # interpolate between the two most recent measurements instead.
            (s0, a0), (s1, a1) = [(p["demand_scale"], p["achieved_density_veh_km_lane"])
                                  for p in attempts[-2:]]
            demand_scale = (s1 + (density_veh_km_lane - a1) * (s1 - s0) / (a1 - a0)
                            if a1 != a0 else s1 * density_veh_km_lane / a1)
            demand_scale = max(demand_scale, 1e-3)
        logger.info("SUMO grid demand calibration: achieved %.3g vs %.3g veh/km/lane; "
                    "re-running with demand x%.3f", achieved, density_veh_km_lane, demand_scale)

    trace.meta["demand_calibration"] = attempts
    _check_density(trace, scenario, density_veh_km_lane, lane_km_override=lane_km)
    if not keep_fcd:
        fcd.unlink(missing_ok=True)
    return trace
