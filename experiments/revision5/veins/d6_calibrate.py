"""DEVIATIONS D6: calibrate the sub-epoch timing mode on the Veins audit seeds 210-219, freeze it, re-run.

    python -m experiments.revision5.veins.d6_calibrate run   --jobs 4   (16 grid points x 200 paired runs, resumable)
    python -m experiments.revision5.veins.d6_calibrate score            (loss per point; freezes the minimum once)
    python -m experiments.revision5.veins.d6_calibrate rerun --jobs 4   (frozen mode: 4 cells x seeds 200-229 x 55)

Grid, data, loss and tie rule exactly as fixed in DEVIATIONS D6 before any calibration number existed:
  * carrier-sense threshold {-95, -92, -88, -85} dBm x capture SINR threshold {4, 6, 8, 10} dB;
  * 5 settings (flooding; p_persistence_03 p=0.3; slotted_1p n=8; weighted_p max_p=0.5; counter_based C=3) x the
    4 headline cells x seeds 210-219, custom-on-SUMO vs the Veins replays of the same (cell, seed, setting);
  * loss = max over cells of the largest |dPDR| over the 50 m bins below the nominal range (D3 definition, both
    simulators pooled over the cell's 50 runs) + mean over cells of the relative |d busy ratio| (D3 definition:
    |median Veins - median custom| / median custom); ties -> the point closest to (-92 dBm, 8 dB).
The frozen point is written once to configs/phy_patch_d6_frozen.json and never revisited (score refuses to change it).
"""

from __future__ import annotations

import argparse
import json
import math
import sys

import numpy as np

from common.config import PROJECT_ROOT
from experiments.revision5.analysis.h3 import NOMINAL, _num, key, veins_rows
from experiments.revision5.sweep import HEADLINE

VEINS = PROJECT_ROOT / "results" / "revision5" / "veins"
CAL_DIR = VEINS / "d6_calibration"
FROZEN = PROJECT_ROOT / "configs" / "phy_patch_d6_frozen.json"
RERUN = VEINS / "custom_h3_subepoch.jsonl"
CS = (-95.0, -92.0, -88.0, -85.0)
SINR = (4.0, 6.0, 8.0, 10.0)
SETTINGS = ["flooding:", "p_persistence_03:0.3", "slotted_1p:8", "weighted_p:0.5", "counter_based:3"]
SEEDS = list(range(210, 220))
N_POINT = len(HEADLINE) * len(SEEDS) * len(SETTINGS)       # 200


def patch(cs: float, sinr: float) -> dict:
    return {"mac": {"timing": "sub_epoch", "carrier_sense_threshold_dbm": cs}, "receiver": {"sinr_threshold_db": sinr}}


def out_path(cs: float, sinr: float):
    return CAL_DIR / f"cs{cs:g}_sinr{sinr:g}.jsonl"


def _count(p) -> int:
    return sum(1 for _ in p.open(encoding="utf-8")) if p.exists() else 0


def run(jobs: int) -> int:
    from experiments.revision5.veins import custom_h3
    CAL_DIR.mkdir(parents=True, exist_ok=True)
    for cs in CS:
        for sn in SINR:
            p = out_path(cs, sn)
            if _count(p) >= N_POINT:
                continue
            print(f"[d6] point cs={cs:g} dBm sinr={sn:g} dB ({_count(p)}/{N_POINT} done)", flush=True)
            custom_h3.main(["--jobs", str(jobs), "--seeds", f"{SEEDS[0]}-{SEEDS[-1]}", "--phy-patch",
                            json.dumps(patch(cs, sn)), "--out", str(p), "--settings", *SETTINGS])
    print("[d6] calibration runs complete", flush=True)
    return 0


def _pdr(recs) -> dict[int, tuple[float, int]]:
    a, s = {}, {}
    for r in recs:
        for b, n in r["bins_attempts"].items():
            a[int(b)] = a.get(int(b), 0) + n
            s[int(b)] = s.get(int(b), 0) + r["bins_success"].get(b, r["bins_success"].get(str(b), 0))
    return {b: (s[b] / a[b], a[b]) for b in a}


def point_loss(recs: list[dict], vrows: list[dict], vraw: dict) -> dict:
    want = {key(*s.split(":")) for s in SETTINGS}
    cells = {}
    for sc, d in HEADLINE:
        R = NOMINAL[sc]
        cv = [r for r in recs if r["scenario"] == sc and float(r["density"]) == d]
        vv = [v for (s2, d2, sd, k), v in vraw.items()
              if s2 == sc and d2 == d and sd in SEEDS and k in want and v is not None]
        pv, pc = _pdr(vv), _pdr(cv)
        diffs = {b: abs(pv[b][0] - pc[b][0]) for b in sorted(set(pv) | set(pc))
                 if b * 50.0 < R and pv.get(b, (0, 0))[1] and pc.get(b, (0, 0))[1]}
        empty = [b for b in sorted(set(pv) | set(pc)) if b * 50.0 < R and b not in diffs]
        cbr_v = np.array([x for x in (_num(r["channel_busy_ratio"]) for r in vrows
                                      if r["scenario"] == sc and float(r["density"]) == d and int(r["seed"]) in SEEDS
                                      and key(r["policy"], r["value"]) in want) if math.isfinite(x)])
        cbr_c = np.array([r["beacon_cbr"] + r["dissemination_cbr"] for r in cv])
        mv, mc = float(np.median(cbr_v)), float(np.median(cbr_c))
        cells[f"{sc} d={d:g}"] = {"pdr_max_abs_diff": max(diffs.values()), "pdr_bins_compared": len(diffs),
                                  "pdr_bins_empty_in_one": empty, "cbr_veins_median": mv, "cbr_custom_median": mc,
                                  "cbr_relative_diff": abs(mv - mc) / mc, "n_runs": [len(vv), len(cv)],
                                  "n_cbr": [int(cbr_v.size), int(cbr_c.size)]}
    pdr_term = max(c["pdr_max_abs_diff"] for c in cells.values())
    cbr_term = float(np.mean([c["cbr_relative_diff"] for c in cells.values()]))
    return {"loss": pdr_term + cbr_term, "pdr_term": pdr_term, "cbr_term": cbr_term, "cells": cells}


def score() -> int:
    from experiments.revision5.analysis.common import write
    vrows, vraw = veins_rows()
    pts, missing = [], []
    for cs in CS:
        for sn in SINR:
            p = out_path(cs, sn)
            recs = [json.loads(l) for l in p.open(encoding="utf-8")] if p.exists() else []
            if len(recs) < N_POINT:
                missing.append([cs, sn, len(recs)])
                continue
            pts.append({"cs_dbm": cs, "sinr_db": sn, **point_loss(recs, vrows, vraw)})
    if missing:
        print(f"[d6] incomplete points (cs, sinr, runs): {missing}; nothing frozen", flush=True)
        return 1
    best = min(pts, key=lambda q: (round(q["loss"], 12), math.hypot(q["cs_dbm"] + 92.0, q["sinr_db"] - 8.0)))
    frozen = patch(best["cs_dbm"], best["sinr_db"])
    if FROZEN.exists():
        prev = json.loads(FROZEN.read_text(encoding="utf-8"))
        if prev["phy_patch"] != frozen:
            print(f"[d6] REFUSING: frozen point exists and differs ({prev['phy_patch']} vs {frozen})", flush=True)
            return 2
    else:
        FROZEN.write_text(json.dumps({"phy_patch": frozen, "loss": best["loss"], "source": "DEVIATIONS D6",
                                      "calibration": "results/revision5/analysis/d6_calibration.json"}, indent=2) + "\n",
                          encoding="utf-8")
    print(write("d6_calibration.json", {"points": pts, "frozen": {"cs_dbm": best["cs_dbm"], "sinr_db": best["sinr_db"],
                                                                     "loss": best["loss"], "phy_patch": frozen}},
                {"cs_grid": CS, "sinr_grid": SINR, "settings": SETTINGS}, SEEDS, len(pts) * N_POINT))
    for q in sorted(pts, key=lambda q: q["loss"]):
        print(f"cs {q['cs_dbm']:6g} sinr {q['sinr_db']:4g}  loss {q['loss']:.4f}  (pdr {q['pdr_term']:.4f}, "
              f"cbr {q['cbr_term']:.4f})")
    print("frozen:", frozen)
    return 0


def rerun(jobs: int) -> int:
    from experiments.revision5.veins import custom_h3
    if not FROZEN.exists():
        print("[d6] no frozen point yet (run score first)", flush=True)
        return 1
    fp = json.loads(FROZEN.read_text(encoding="utf-8"))["phy_patch"]
    return custom_h3.main(["--jobs", str(jobs), "--seeds", "200-229", "--phy-patch", json.dumps(fp),
                           "--out", str(RERUN)])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=("run", "score", "rerun"))
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args(argv)
    import numpy
    print(f"[env] {sys.executable} numpy {numpy.__version__}", flush=True)
    return {"run": lambda: run(a.jobs), "score": score, "rerun": lambda: rerun(a.jobs)}[a.mode]()


if __name__ == "__main__":
    raise SystemExit(main())
