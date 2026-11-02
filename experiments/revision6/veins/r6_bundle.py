"""Revision 6: add the risk-field / hazard lines the revision-6 Veins app needs to an exported bundle (r6.txt).

    python -m experiments.revision6.veins.r6_bundle --seeds 200-209 [--cells headline]

Values are taken from the custom simulator's own objects for that (cell, seed) -- the RiskField (with its geometry)
and the Hazard of the run, and the canonical trace's static carriageway sign and vehicle class per vehicle -- and
written with repr() so the app parses the identical doubles. Nothing is recomputed or approximated here.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from common.config import PROJECT_ROOT
from experiments.revision5.sweep import HEADLINE
from experiments.run_sim import RunSpec, run_single
from hazard.model import DecayKind
from hazard.risk_field import GridRiskGeometry

BUNDLES = PROJECT_ROOT / "cache" / "veins"
DECAY = {DecayKind.NONE: 0, DecayKind.EXPONENTIAL: 1, DecayKind.LINEAR_TTL: 2}


def lines(scenario: str, density: float, seed: int) -> tuple[str, dict]:
    _, res = run_single(RunSpec(scenario=scenario, density_veh_km_lane=density, seed=seed, policy="flooding",
                                backend="sumo"), return_result=True)
    rf, hz, tr = res.risk, res.hazard, res.trace
    g = rf.geometry
    grid = isinstance(g, GridRiskGeometry)
    out = [f"risk {'grid' if grid else 'highway'} {rf.eta_full_s!r} {rf.eta_decay_tau_s!r} {rf.inside_span_weight!r} "
           f"{rf.min_speed_ms!r} {rf.severity_gamma!r} {rf.opposing_direction_relevance!r}",
           f"hazard {float(hz.span_start_m)!r} {float(hz.span_end_m)!r} {int(hz.affected_direction)} {DECAY[hz.decay]} "
           f"{float(hz.severity0)!r} {float(hz.onset_time_s)!r} {float(hz.tau_s or 0.0)!r} {float(hz.ttl_s or 0.0)!r}"]
    if grid:
        hx, hy = g.hazard_xy if g.hazard_xy is not None else (hz.span_mid_m, 0.0)
        heading = g.hazard_heading if g.hazard_heading is not None else hz.heading_rad
        out.append(f"gridgeom {float(hx)!r} {float(hy)!r} {float(heading or 0.0)!r} {int(heading is not None)} "
                   f"{float(g.span_tolerance_m)!r} {float(g.heading_tolerance_rad)!r}")
    direction = np.asarray(tr.direction)
    out.append(f"anydir {int(np.any(direction.astype(float) != 0.0))}")
    out += [f"dir {i} {int(direction[i])} {str(tr.vclass[i]).replace(' ', '_')}" for i in range(len(direction))]
    meta = {"grid": grid, "n_vehicles": int(len(direction)), "origin": [int(res.origin_index), int(res.origin_step)]}
    return "\n".join(out) + "\n", meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", required=True, help="e.g. 200-209")
    a = ap.parse_args(argv)
    import numpy
    print(f"[env] {sys.executable} numpy {numpy.__version__}", flush=True)
    lo, hi = (int(x) for x in a.seeds.split("-"))
    for sc, d in HEADLINE:
        for s in range(lo, hi + 1):
            bd = BUNDLES / f"{sc}_d{d:g}_s{s}"
            if not (bd / "bundle.json").exists():
                print(f"[r6] {bd.name}: no bundle", flush=True)
                continue
            txt, meta = lines(sc, d, s)
            b = json.loads((bd / "bundle.json").read_text())
            if [b["origin"]["index"], b["origin"]["step"]] != meta["origin"] or len(b["vehicles"]) != meta["n_vehicles"]:
                raise SystemExit(f"{bd.name}: custom run does not match the bundle (origin/vehicles)")
            p = bd / "r6.txt"
            if p.exists() and p.read_text() != txt:
                raise SystemExit(f"{bd.name}: r6.txt exists and differs; refusing to overwrite")
            p.write_text(txt, newline="\n")
            print(f"[r6] {bd.name}: {'grid' if meta['grid'] else 'highway'}, {meta['n_vehicles']} vehicles", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
