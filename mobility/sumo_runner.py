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

.. warning::
   This module is **untested against a live SUMO installation** in the current
   development environment (no SUMO present). Treat the first real SUMO run as
   an integration test, not as known-good code.
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


def _build_highway_network(scenario: dict[str, Any], tools: SumoTools, work: Path) -> Path:
    """Synthetic straight corridor: one lane per direction, `junctions` splits."""
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
    for i in range(n_seg):
        edges.append(
            f'  <edge id="e{i}" from="n{i}" to="n{i+1}" numLanes="{lanes}" '
            f'speed="{vmax:.2f}" priority="2"/>'
        )
        edges.append(
            f'  <edge id="-e{i}" from="n{i+1}" to="n{i}" numLanes="{lanes}" '
            f'speed="{vmax:.2f}" priority="2"/>'
        )
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


def _build_network(scenario: dict[str, Any], tools: SumoTools, work: Path) -> tuple[Path, str]:
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
    return _build_highway_network(scenario, tools, work), "synthetic"


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
    duration = float(sim["duration_s"]) + float(sim["warmup_s"])
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
        for d, prefix in ((1, ""), (-1, "-")):
            route = " ".join(
                f"{prefix}e{i}" for i in (range(n_seg) if d == 1 else reversed(range(n_seg)))
            )
            body.append(f'  <route id="r{d}" edges="{route}"/>')
            for name, c in classes.items():
                body.append(
                    f'  <flow id="f{d}_{name}" route="r{d}" type="{name}" begin="0" '
                    f'end="{duration:.1f}" vehsPerHour="{q_veh_h * c["share"]:.1f}" '
                    f'departLane="best" departSpeed="max"/>'
                )
    return _write(work / "demand.rou.xml", "<routes>\n" + "\n".join(body) + "\n</routes>")


def generate_sumo_trace(
    scenario: dict[str, Any],
    density_veh_km_lane: float,
    seed: int,
    *,
    tools: SumoTools,
    speed_factor: float = 1.0,
    keep_fcd: bool = False,
) -> Trace:
    """Build the network + demand, run SUMO, and parse the FCD export."""
    sim, scfg = scenario["simulation"], scenario["sumo"]
    work = ensure_dir(BUILD_DIR / f"{scenario['name']}_d{density_veh_km_lane:g}_s{seed}")

    net, net_source = _build_network(scenario, tools, work)
    routes = _write_routes(scenario, density_veh_km_lane, seed, work, speed_factor)
    duration = float(sim["duration_s"]) + float(sim["warmup_s"])
    fcd = work / "fcd.xml"

    _write(work / "run.sumocfg", f"""<configuration>
  <input>
    <net-file value="{net.name}"/>
    <route-files value="{routes.name}"/>
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

    logger.info("Running SUMO (%s network, density=%g veh/km/lane, seed=%d)",
                net_source, density_veh_km_lane, seed)
    _run([tools.sumo, "-c", "run.sumocfg", "--fcd-output", fcd.name,
          "--fcd-output.period", str(scfg["fcd_period"]), "--no-step-log", "true",
          "--no-warnings", "true"], cwd=work)

    trace = parse_fcd(
        fcd,
        scenario_name=scenario["name"],
        dt=float(sim["timestep_s"]),
        warmup_s=float(sim["warmup_s"]),
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
    if not keep_fcd:
        fcd.unlink(missing_ok=True)
    return trace
