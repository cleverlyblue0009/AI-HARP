"""Simulator validation against a published curve (paper Section: simulator credibility).

::

    D:/aiharp-env/python.exe -m experiments.validate_amador --jobs 3
    D:/aiharp-env/python.exe -m experiments.validate_amador --time-one      # cost check

Reference: O. Amador, M. Urueña, M. Calderon, I. Soto, "Evaluation and
improvement of ETSI ITS Contention-Based Forwarding (CBF) of warning messages
in highway scenarios", Vehicular Communications 34 (2022) 100454 (arXiv
2403.05994). Simulated with Artery (Veins / OMNeT++, Vanetza, SUMO). Table 3,
ETSI column: average PDR per density.

Every setting below carries a provenance tag:

  [PAPER]    stated in the paper
  [DERIVED]  computed from a stated quantity
  [ASSUMED]  not stated in the paper; our choice, and a source of deviation

The paper does NOT state receiver sensitivity, noise, CAM rate or vehicle
speeds, and our engine differs structurally in ways listed in DEVIATIONS. The
deviation reported is therefore "our simulator configured as closely as the
paper allows", not a like-for-like replication.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from common.config import PROJECT_ROOT, load_yaml

#: Table 3, ETSI CBF column, density (veh/km/lane) -> PDR.              [PAPER]
REFERENCE_PDR = {10.0: 0.9998, 20.0: 0.9961, 30.0: 0.9280, 40.0: 0.9371, 50.0: 0.9372}
REFERENCE = ("Amador et al., Veh. Commun. 34 (2022) 100454, Table 3 (ETSI CBF); "
             "Artery/Veins/OMNeT++; 5 runs x 30 DENMs per density")

ROAD_M = 5000.0                 # [PAPER] 5 km straightaway
LANES_PER_DIR = 4               # [PAPER] 8 lanes, 4 per direction
TX_POWER_DBM = 10 * math.log10(20.0)   # [PAPER] 20 mW = 13.01 dBm
RANGE_M = 778.0                 # [PAPER] maximum transmission range measured
DATA_RATE_MBPS = 6.0            # [PAPER]
BANDWIDTH_HZ = 10e6             # [PAPER]
FREQ_HZ = 5.9e9                 # [PAPER]
DENM_BYTES = 301                # [PAPER]
CAM_BYTES = 285                 # [PAPER]
HOP_LIMIT = 10                  # [PAPER] ETSI MIB default
LIFETIME_S = 10.0               # [PAPER]
AREA_BEHIND_M = 4000.0          # [PAPER] rectangle 4 km behind the source ...
AREA_AHEAD_M = 100.0            # [PAPER] ... and 100 m in front, all 8 lanes
TIMESTEP_S = 0.01               # [DERIVED] resolves CBF's 1-100 ms timers (user decision)
SOURCE_X_M = 4500.0             # [ASSUMED] "near the beginning": leaves 4 km upstream
ONSET_S = 5.0                   # [ASSUMED]
SPEED_KMH = (100.0, 130.0)      # [ASSUMED] paper: SUMO defaults, not stated
CAM_HZ = 10.0                   # [ASSUMED] paper: TC2 CAMs, rate not stated

DEVIATIONS = (
    "Receiver sensitivity/noise not stated: derived so free-space range = 778 m.",
    "Vehicle speeds not stated (SUMO defaults): cars only, 100-130 km/h assumed.",
    "CAM rate not stated: modelled as our background beacon load at 10 Hz with DCC.",
    "One DENM per run from a moving detecting vehicle, not 30 DENMs at 1 Hz from a "
    "stationary one; statistics pooled over seeds instead.",
    "Epoch-based engine at 10 ms: every transmission in an epoch is concurrent; "
    "802.11p backoff is abstracted (sim/mac.py), not simulated per 13 us slot.",
    "Mobility: our fallback generator (or SUMO if present), not the paper's SUMO network.",
)


def derived_receiver(phy_cfg: dict[str, Any]) -> tuple[float, float]:
    """(sensitivity_dbm, noise_figure_db) giving a 778 m free-space range.   [DERIVED]

    PL(d) = 20 log10(4 pi d f / c); sensitivity = Ptx - PL(778 m) with 0 dBi
    antennas. The noise figure is lowered so that noise + SINR threshold equals
    that sensitivity; otherwise the SINR test would cap the range first.
    """
    pl = 20 * math.log10(4 * math.pi * RANGE_M * FREQ_HZ / 299_792_458.0)
    sens = TX_POWER_DBM - pl
    noise_no_nf = float(phy_cfg["radio"]["thermal_noise_density_dbm_hz"]) + 10 * math.log10(BANDWIDTH_HZ)
    nf = sens - float(phy_cfg["receiver"]["sinr_threshold_db"]) - noise_no_nf
    return sens, nf


def configs(seed_scenario: str = "rural_highway") -> dict[str, Any]:
    """Scenario, PHY, hazard and engine configs for the validation run."""
    scn = load_yaml(f"scenario_{'rural' if seed_scenario == 'rural_highway' else seed_scenario}.yaml")
    scn = copy.deepcopy(scn)
    scn["name"] = "amador_highway"
    scn["geometry"].update(length_m=ROAD_M, lanes_per_direction=LANES_PER_DIR, junctions=0)
    scn["vehicles"]["classes"] = {"car": {**scn["vehicles"]["classes"]["car"], "share": 1.0,
                                          "speed_kmh_min": SPEED_KMH[0],
                                          "speed_kmh_max": SPEED_KMH[1]}}
    scn["simulation"].update(timestep_s=TIMESTEP_S, duration_s=ONSET_S + LIFETIME_S + 1.0)
    scn["sumo"].update(step_length=TIMESTEP_S, fcd_period=TIMESTEP_S)

    phy = copy.deepcopy(load_yaml("phy.yaml"))
    sc = "rural_highway"                      # per-scenario keys the PHY builder reads
    phy["radio"].update(tx_power_dbm=TX_POWER_DBM, tx_antenna_gain_dbi=0.0,
                        rx_antenna_gain_dbi=0.0, data_rate_mbps=DATA_RATE_MBPS,
                        bandwidth_hz=BANDWIDTH_HZ, carrier_frequency_hz=FREQ_HZ)
    sens, nf = derived_receiver(phy)
    phy["radio"]["noise_figure_db"] = nf
    phy["receiver"]["sensitivity_dbm"] = sens
    pl = phy["path_loss"]
    pl["exponent"][sc] = 2.0                  # [PAPER] alpha = 2.0
    pl["exponent_far"][sc] = 2.0
    pl["breakpoint_distance_m"] = {sc: 1e9}   # single slope
    pl["shadowing_std_db"][sc] = 0.0          # [PAPER] simple path loss, no shadowing stated
    phy["fading"]["m"][sc] = 1e6              # [DERIVED] no fading stated: m -> inf
    phy["mac"].update(frame_bytes=DENM_BYTES, background_beacon_bytes=CAM_BYTES,
                      background_beacon_hz=CAM_HZ, carrier_sense_threshold_dbm=sens)

    hz = copy.deepcopy(load_yaml("hazard.yaml"))
    hz["default_instance"].update(position_frac=SOURCE_X_M / ROAD_M, onset_time_s=ONSET_S,
                                  extent_m=10.0, detection_range_m=20.0, affected_direction=0)

    exp = copy.deepcopy(load_yaml("experiment.yaml"))
    exp["simulation"].update(max_hops=HOP_LIMIT, message_ttl_s=LIFETIME_S)
    return {"scenario": scn, "phy": phy, "hazard": hz, "experiment": exp,
            "derived": {"sensitivity_dbm": sens, "noise_figure_db": nf}}


def pdr(result: Any, trace: Any, area: tuple[float, float], lifetime_steps: int) -> dict[str, float]:
    """The paper's PDR: vehicles that received / vehicles in the area at generation.

    ``paper`` counts every receiver that was inside the area when it received
    (so vehicles entering during forwarding count, and PDR may exceed 1, as in
    the paper). ``strict`` counts only vehicles present at generation.
    """
    o = int(result.origin_step)
    if o < 0:
        return {"paper": float("nan"), "strict": float("nan"), "n_area": 0}
    x = np.asarray(trace.x)
    active = np.asarray(trace.active, dtype=bool)
    lo, hi = area
    at_gen = active[o] & (x[o] >= lo) & (x[o] <= hi)
    at_gen[result.origin_index] = False          # the source is not a receiver
    n_area = int(at_gen.sum())
    inf = np.asarray(result.informed_step)
    got = (inf >= 0) & (inf <= o + lifetime_steps)
    got[result.origin_index] = False
    idx = np.flatnonzero(got)
    x_rx = x[inf[idx], idx]
    in_area_at_rx = np.zeros_like(got)
    in_area_at_rx[idx] = (x_rx >= lo) & (x_rx <= hi)
    return {"paper": float(in_area_at_rx.sum() / n_area) if n_area else float("nan"),
            "strict": float((got & at_gen).sum() / n_area) if n_area else float("nan"),
            "n_area": n_area}


def run_one(args: tuple[float, int]) -> dict[str, Any]:
    density, seed = args
    logging.disable(logging.INFO)
    from agents.registry import build_policy
    from common.seeding import SeedBundle
    from hazard.model import hazard_from_config
    from hazard.risk_field import build_risk_field
    from mobility.generate import get_trace
    from sim.engine import DisseminationEngine, SimSettings
    from sim.mac import build_mac
    from sim.phy import build_phy

    c = configs()
    t0 = time.time()
    trace = get_trace(c["scenario"], density, seed, weather="clear", phy_cfg=c["phy"])
    t_mob = time.time() - t0
    hazard = hazard_from_config(c["hazard"], trace.meta, overrides={"type": "crash"})
    phy = build_phy(c["phy"], "rural_highway", "clear", seed, trace_meta=trace.meta)
    mac = build_mac(c["phy"], phy)
    risk = build_risk_field(c["hazard"], trace, hazard)
    x_src = SOURCE_X_M
    area = (x_src - AREA_BEHIND_M, x_src + AREA_AHEAD_M)
    policy = build_policy("etsi_cbf", area_x_min_m=area[0], area_x_max_m=area[1])
    eng = DisseminationEngine(trace, phy, mac, risk, hazard, policy, SeedBundle(master_seed=seed),
                              SimSettings.from_config(c["experiment"]))
    res = eng.run()
    p = pdr(res, trace, area, int(round(LIFETIME_S / TIMESTEP_S)))
    return {"density": density, "seed": seed, **p, "n_transmissions": int(res.n_transmissions),
            "origin_x": float(np.asarray(trace.x)[res.origin_step, res.origin_index])
            if res.origin_step >= 0 else float("nan"),
            "comm_range_m": float(eng.comm_range_m), "t_mobility_s": round(t_mob, 1),
            "t_total_s": round(time.time() - t0, 1)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Validate against Amador et al. 2022, Table 3")
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--densities", nargs="*", type=float, default=sorted(REFERENCE_PDR))
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--time-one", action="store_true",
                    help="run one seed at the lowest and highest density and report cost")
    ap.add_argument("--out-dir", default="results/validation")
    args = ap.parse_args(argv)

    c = configs()
    print(f"derived receiver: sensitivity {c['derived']['sensitivity_dbm']:.2f} dBm, "
          f"noise figure {c['derived']['noise_figure_db']:.2f} dB (range-matched to {RANGE_M} m)")
    if args.time_one:
        for d in (min(args.densities), max(args.densities)):
            r = run_one((d, 0))
            print(json.dumps(r))
        return 0

    work = [(d, s) for d in args.densities for s in range(args.seeds)]
    rows = []
    if args.jobs > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            for r in ex.map(run_one, work):
                rows.append(r)
                print(f"  d={r['density']:g} seed={r['seed']} pdr={r['paper']:.3f} "
                      f"({r['t_total_s']}s)", flush=True)
    else:
        for w in work:
            rows.append(run_one(w))

    out = PROJECT_ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    summary = {"reference": REFERENCE, "deviations": DEVIATIONS, "derived": c["derived"],
               "per_density": {}}
    lines = [f"Validation against {REFERENCE}", "",
             f"{'density':>8} {'ours PDR':>16} {'strict':>8} {'paper':>7} {'diff':>8} {'n':>4}"]
    for d in args.densities:
        v = np.array([r["paper"] for r in rows if r["density"] == d and np.isfinite(r["paper"])])
        s = np.array([r["strict"] for r in rows if r["density"] == d and np.isfinite(r["strict"])])
        ref = REFERENCE_PDR.get(d, float("nan"))
        mu, se = (float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v)))) if len(v) > 1 else (float("nan"), float("nan"))
        summary["per_density"][f"{d:g}"] = {"pdr_mean": mu, "pdr_se": se,
                                            "pdr_strict_mean": float(s.mean()) if len(s) else float("nan"),
                                            "paper": ref, "diff": mu - ref, "n": int(len(v))}
        lines.append(f"{d:8g} {mu:9.4f} +/- {se:.4f} {float(s.mean()) if len(s) else float('nan'):8.4f} "
                     f"{ref:7.4f} {mu - ref:+8.4f} {len(v):4d}")
    diffs = [abs(v["diff"]) for v in summary["per_density"].values() if np.isfinite(v["diff"])]
    summary["mean_abs_diff"] = float(np.mean(diffs)) if diffs else float("nan")
    lines += ["", f"mean |ours - paper| = {summary['mean_abs_diff']:.4f}", "", "Deviations:"]
    lines += [f"  - {x}" for x in DEVIATIONS]
    (out / "amador2022_runs.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    (out / "amador2022.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    (out / "amador2022.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
