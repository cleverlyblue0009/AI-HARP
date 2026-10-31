"""H3 network layer, custom-simulator side: the paired runs (headline cells x EVAL seeds 200-209 x 55 classical
settings, SUMO mobility) re-run with SimSettings.record_rx_pairs (DEVIATIONS D3).

    python -m experiments.revision5.veins.custom_h3 --jobs 2         (sets SUMO_HOME itself)

DEVIATIONS D6 (sub-epoch timing mode): --phy-patch '{"mac": {"timing": "sub_epoch", ...}}' deep-merges into
configs/phy.yaml, --out names the output file, --settings restricts to "policy:value" settings; every record then also
carries the patch and the full metric row (the EVAL-row check only applies to the unpatched default).

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


def _merge(base: dict, patch: dict) -> dict:
    for k, v in patch.items():
        base[k] = _merge(base.get(k, {}), v) if isinstance(v, dict) else v
    return base


def _one(t: dict, patch: dict | None = None) -> dict:
    import numpy as np

    from common.config import load_yaml
    from experiments.revision1.provenance import git_commit
    from experiments.run_sim import RunSpec, run_single
    exp = load_yaml("experiment.yaml")
    exp["simulation"]["record_rx_pairs"] = True
    spec = RunSpec(scenario=t["scenario"], density_veh_km_lane=t["density"], weather=t["weather"], policy=t["policy"],
                   policy_params=t["params"], seed=t["seed"], hazard_type=t["hazard_type"], backend="sumo")
    phy = _merge(load_yaml("phy.yaml"), patch) if patch else None
    m, res = run_single(spec, phy_cfg=phy, exp_cfg=exp, return_result=True)
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
            "config_hash": m.get("config_hash"), "git_commit": git_commit(), "backend": res.trace.backend,
            **({"phy_patch": patch, "metrics": {k: v for k, v in m.items() if isinstance(v, (int, float, str))},
                "sub_epoch": {k: v for k, v in res.meta.items() if k.startswith("sub_epoch")}} if patch else {})}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--seeds", default="200-209")
    ap.add_argument("--phy-patch", default="", help="JSON deep-merged into phy.yaml (D6)")
    ap.add_argument("--out", default="", help="output JSONL (default custom_h3.jsonl)")
    ap.add_argument("--settings", nargs="*", default=None, help='restrict to "policy:value" (value "" for none)')
    a = ap.parse_args(argv)
    os.environ.setdefault("SUMO_HOME", "D:/sumo/sumo-1.19.0")
    import numpy
    print(f"[env] {sys.executable} numpy {numpy.__version__}", flush=True)
    from experiments.revision1.runner import load_rows, task
    from experiments.revision1.sweeps import knob_settings
    from experiments.revision5.sweep import HEADLINE
    lo, hi = (int(x) for x in a.seeds.split("-"))
    patch = json.loads(a.phy_patch) if a.phy_patch else None
    out = Path(a.out) if a.out else OUT
    keep = None if a.settings is None else set(a.settings)
    tasks = [task(sc, d, pol, s, param=par, value=val, backend="sumo")
             for sc, d in HEADLINE for s in range(lo, hi + 1) for pol, par, val in knob_settings()
             if keep is None or f"{pol}:{'' if val is None else val}" in keep]
    if patch:
        print(f"[custom_h3] phy patch {json.dumps(patch, sort_keys=True)} -> {out}", flush=True)
    ref = {r["task_hash"]: r for r in load_rows(PROJECT_ROOT / "results" / "revision5" / "runs" /
                                                 "classical__EVAL__headline__default__sumo.csv")}
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["task_hash"] for l in out.open(encoding="utf-8")} if out.exists() else set()
    todo = [t for t in tasks if t["task_hash"] not in done]
    print(f"[custom_h3] {len(todo)} of {len(tasks)} runs to do", flush=True)
    mism = 0
    with out.open("a", encoding="utf-8") as fh, ProcessPoolExecutor(a.jobs) as ex:
        futs = {ex.submit(_one, t, patch): t for t in todo}
        for i, f in enumerate(as_completed(futs), 1):
            t = futs[f]
            try:
                rec = f.result()
            except Exception as exc:  # noqa: BLE001 -- logged, retried on the next launch
                print(f"[custom_h3] FAILED {t['task_hash']}: {exc!r}", flush=True)
                continue
            r = ref.get(t["task_hash"])
            rec["matches_eval_row"] = None if patch else r is not None and all(_same(float(r[k]), rec[k])
                                                             for k in ("rwcr", "tx_per_at_risk_informed"))
            mism += rec["matches_eval_row"] is False
            fh.write(json.dumps(rec, default=str) + "\n")
            fh.flush()
            if i % 200 == 0:
                print(f"[custom_h3] {i}/{len(todo)} (mismatches vs EVAL rows so far: {mism})", flush=True)
    print(f"[custom_h3] done; mismatches vs EVAL rows: {mism}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
