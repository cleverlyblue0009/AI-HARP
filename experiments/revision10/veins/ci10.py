"""Revision 10: one Veins shard on a Linux runner (results/revision10/ADDENDUM.md §7).

    python -m experiments.revision10.veins.ci10 h7 --seed 230 --workers 4     urban NLOS d=20, R-A
    python -m experiments.revision10.veins.ci10 s3 --seed 245 --workers 4     urban NLOS d=20, R-B
    python -m experiments.revision10.veins.ci10 s5 --seed 233 --workers 4     rural d=20, R-A

experiments/revision9/veins/ci9.py (H6) with: output under results/revision10/veins/<mode>/, the cell and seed list fixed by
the mode, STRUCT9 parameters from results/revision9/train, and ONE added check -- the shard's regenerated bundle must have
the content digest recorded in results/revision10/gates.json for that (cell, seed). Every shard first restores the
AI-HARP checkpoints (hash-checked), passes gate check 1 (707d6533...) and gate check 2 (220dd1fd...), and contributes
nothing if any of the three fails. 60 replays (10 runs x delta -2..3), every decision log re-sent offline
(0 mismatches), scoring as revision 8/9.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import sys

from common.config import PROJECT_ROOT
from experiments.revision8.veins.ci_veins import (APP, BUNDLES, DELTAS, GATE2_REF, GATE_SHA, bundles, replay,
                                                  restore_checkpoints, score_dir, sha_events)
from experiments.revision9.veins.ci9 import RUNS, bridged9

TRAIN = "results/revision9/train"
SEEDS = json.loads((PROJECT_ROOT / "results" / "revision10" / "seeds.json").read_text())
MODES = {"h7": (("urban_nlos", 20.0), SEEDS["R_A"]), "s3": (("urban_nlos", 20.0), SEEDS["R_B"]),
         "s5": (("rural_highway", 20.0), SEEDS["R_A"])}


def main(argv=None) -> int:
    import numpy
    from experiments.revision10.veins.gate10 import digest
    from experiments.revision9.veins.decision_server9 import trace
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=list(MODES))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)
    (sc, d), allowed = MODES[a.mode]
    seed = a.seed
    if seed not in allowed:
        raise SystemExit(f"seed {seed} is not in the pre-registered set for {a.mode}: {allowed}")
    print(f"[env] {sys.executable} numpy {numpy.__version__}", flush=True)
    out = PROJECT_ROOT / "results" / "revision10" / "veins" / a.mode / f"{sc}_d{d:g}_s{seed}"
    out.mkdir(parents=True, exist_ok=True)
    for run in RUNS:
        t = json.loads((PROJECT_ROOT / TRAIN / run / "theta.json").read_text())
        assert t["generations"] == 80, (run, t["generations"])
    want = json.loads((PROJECT_ROOT / "results" / "revision10" / "gates.json").read_text())["bundles"]["per_bundle"]
    ck = restore_checkpoints()
    bundles({(("rural_highway", 20.0), 200), ((sc, d), seed)})
    bdir = BUNDLES / f"{sc}_d{d:g}_s{seed}"
    dg = digest(bdir)
    app_sha = hashlib.sha256((APP / "aiharp").read_bytes()).hexdigest()
    g0 = BUNDLES / "rural_highway_d20_s200"
    g1 = replay(g0, "slotted_1p", "8", 200, out / "gate")
    ev1 = out / "gate" / f"{g1['name']}.events.csv.gz"
    s1 = sha_events(ev1) if ev1.exists() else None
    bridged9([{"bdir": g0, "seed": 200, "policy": "ai_harp", "knobs": "ref_s0_b1",
               "cfg": {"policy": "ai_harp", "run": "ref_s0", "bias": 1.0, "tau": 0.5}, "raw": out / "gate2",
               "logs": out / "gate2_logs"}], 1, out)
    ev2 = out / "gate2" / "rural_highway_d20_s200__ai_harp_ref_s0_b1__s200.events.csv.gz"
    s2 = sha_events(ev2) if ev2.exists() else None
    ref2 = sha_events(GATE2_REF)
    ref_b = want.get(bdir.name, {}).get("content_sha256")
    gate = {"check1": {"events_sha256": s1, "reference_sha256": GATE_SHA, "pass": s1 == GATE_SHA},
            "check2": {"events_sha256": s2, "reference_sha256": ref2, "pass": s2 == ref2},
            "bundle": {"content_sha256": dg["content_sha256"], "reference_sha256": ref_b,
                       "trace_sha256": dg["trace_sha256"], "pass": dg["content_sha256"] == ref_b},
            "checkpoints": ck, "app_sha256": app_sha}
    gate["pass"] = gate["check1"]["pass"] and gate["check2"]["pass"] and gate["bundle"]["pass"]
    (out / "gate.json").write_text(json.dumps(gate, indent=1))
    print(json.dumps(gate), flush=True)
    for x in ("gate", "gate2", "gate2_logs"):
        shutil.rmtree(out / x, ignore_errors=True)
    if not gate["pass"]:
        return 2
    items = [{"bdir": bdir, "seed": seed, "policy": "struct", "knobs": f"{run}_d{dl:g}",
              "cfg": {"policy": "struct", "run": run, "delta": dl, "train_dir": TRAIN},
              "raw": out / "raw", "logs": out / "bridge_logs"} for run in RUNS for dl in DELTAS]
    bridged9(items, a.workers, out)
    tc = []
    for p in sorted((out / "bridge_logs").glob("*.jsonl")):
        tc.append(trace(p))
        with open(p, "rb") as fi, gzip.open(str(p) + ".gz", "wb") as fo:
            shutil.copyfileobj(fi, fo)
        p.unlink()
    (out / "trace_check.json").write_text(json.dumps({"logs": tc, "all_pass": all(x["pass"] for x in tc),
                                                       "n": len(tc)}, indent=1))
    rows, rejected = score_dir(out / "raw", app_sha)
    with open(out / "rows.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")
    (out / "merge.json").write_text(json.dumps({"merged": len(rows), "rejected": len(rejected), "expected": len(items),
                                                "rejected_replays": rejected, "app_sha256": app_sha,
                                                "train_dir": TRAIN}, indent=1))
    ok = len(rows) == len(items) and len(tc) == len(items) and all(x["pass"] for x in tc)
    print(f"[ci10] {a.mode} {sc} d={d:g} s={seed}: merged {len(rows)}, trace {'pass' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
