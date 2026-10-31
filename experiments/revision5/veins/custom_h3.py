"""H3 network layer, custom-simulator side: the paired runs (headline cells x EVAL seeds 200-209 x 55 classical
settings, SUMO mobility) re-run with SimSettings.record_rx_pairs (DEVIATIONS D3).

    python -m experiments.revision5.veins.custom_h3 --jobs 2         (sets SUMO_HOME itself)

Per run (one JSON line in results/revision5/veins/custom_h3.jsonl, resumable by task_hash):
  * pdr_bins: per 50 m distance bin, in-nominal-range (tx, rx) attempts and decodes;
  * single_hop_delays_s: (informed_step(r) - informed_step(informed_by(r))) * dt for every non-origin informed vehicle;
  * loss split (n_rx_attempts, n_rx_success, n_fail_sensitivity, n_fail_sinr, n_fail_beacon) and beacon_cbr /
    dissemination_cbr (the channel load the custom MAC model applies);
  * rwcr / tx_per_at_risk_informed, checked equal to the committed EVAL row of the same task (classical__EVAL__
    headline__default__sumo.csv) -- the recording draws nothing, so a mismatch means a broken run and is reported.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from common.config import PROJECT_ROOT

OUT = PROJECT_ROOT / "results" / "revision5" / "veins" / "custom_h3.jsonl"
BIN_M = 50.0


def _same(x: float, y: float) -> bool:
    import math
    if math.isnan(x) or math.isnan(y):
        return math.isnan(x) and math.isnan(y)
    return x == y or abs(x - y) <= 1e-12


def _one(t: dict) -> dict:
    import numpy as np

    from common.config import load_yaml
    from experiments.revision1.provenance import git_commit
    from experiments.run_sim import RunSpec, run_single
    exp = load_yaml("experiment.yaml")
    exp["simulation"]["record_rx_pairs"] = True
    spec = RunSpec(scenario=t["scenario"], density_veh_km_lane=t["density"], weather=t["weather"], policy=t["policy"],
                   policy_params=t["params"], seed=t["seed"], hazard_type=t["hazard_type"], backend="sumo")
    m, res = run_single(spec, exp_cfg=exp, return_result=True)
    pairs = res.meta.get("rx_pairs", [])
    att: dict[int, int] = {}
    ok: dict[int, int] = {}
    for _step, _tx, _rx, d, dec in pairs:
        b = int(d // BIN_M)
        att[b] = att.get(b, 0) + 1
        ok[b] = ok.get(b, 0) + int(dec)
    inf = res.informed_step
    by = res.informed_by
    delays = [float((inf[r] - inf[by[r]]) * res.trace.dt) for r in np.flatnonzero(inf >= 0)
              if r != res.origin_index and by[r] >= 0]
    return {"task_hash": t["task_hash"], "scenario": t["scenario"], "density": t["density"], "seed": t["seed"],
            "policy": t["policy"], "param": t["param"], "value": t["value"],
            "pdr_bins_m": BIN_M, "bins_attempts": att, "bins_success": ok, "single_hop_delays_s": delays,
            "n_rx_attempts": res.n_rx_attempts, "n_rx_success": res.n_rx_success,
            "n_fail_sensitivity": res.n_fail_sensitivity, "n_fail_sinr": res.n_fail_sinr,
            "n_fail_beacon": res.n_fail_beacon, "beacon_cbr": m.get("beacon_cbr"),
            "dissemination_cbr": m.get("dissemination_cbr"), "rwcr": m.get("rwcr"),
            "tx_per_at_risk_informed": m.get("tx_per_at_risk_informed"), "transmissions": m.get("transmissions"),
            "config_hash": m.get("config_hash"), "git_commit": git_commit(), "backend": res.trace.backend}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--seeds", default="200-209")
    a = ap.parse_args(argv)
    os.environ.setdefault("SUMO_HOME", "D:/sumo/sumo-1.19.0")
    import numpy
    print(f"[env] {sys.executable} numpy {numpy.__version__}", flush=True)
    from experiments.revision1.runner import load_rows, task
    from experiments.revision1.sweeps import knob_settings
    from experiments.revision5.sweep import HEADLINE
    lo, hi = (int(x) for x in a.seeds.split("-"))
    tasks = [task(sc, d, pol, s, param=par, value=val, backend="sumo")
             for sc, d in HEADLINE for s in range(lo, hi + 1) for pol, par, val in knob_settings()]
    ref = {r["task_hash"]: r for r in load_rows(PROJECT_ROOT / "results" / "revision5" / "runs" /
                                                 "classical__EVAL__headline__default__sumo.csv")}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["task_hash"] for l in OUT.open(encoding="utf-8")} if OUT.exists() else set()
    todo = [t for t in tasks if t["task_hash"] not in done]
    print(f"[custom_h3] {len(todo)} of {len(tasks)} runs to do", flush=True)
    mism = 0
    with OUT.open("a", encoding="utf-8") as fh, ProcessPoolExecutor(a.jobs) as ex:
        futs = {ex.submit(_one, t): t for t in todo}
        for i, f in enumerate(as_completed(futs), 1):
            t = futs[f]
            try:
                rec = f.result()
            except Exception as exc:  # noqa: BLE001 -- logged, retried on the next launch
                print(f"[custom_h3] FAILED {t['task_hash']}: {exc!r}", flush=True)
                continue
            r = ref.get(t["task_hash"])
            rec["matches_eval_row"] = r is not None and all(_same(float(r[k]), rec[k])
                                                             for k in ("rwcr", "tx_per_at_risk_informed"))
            mism += not rec["matches_eval_row"]
            fh.write(json.dumps(rec, default=str) + "\n")
            fh.flush()
            if i % 200 == 0:
                print(f"[custom_h3] {i}/{len(todo)} (mismatches vs EVAL rows so far: {mism})", flush=True)
    print(f"[custom_h3] done; mismatches vs EVAL rows: {mism}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
