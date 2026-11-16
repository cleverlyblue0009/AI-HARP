"""Revision 9: one Veins shard on a Linux runner (results/revision9/ADDENDUM.md, compute as revision-8 D14).

    python -m experiments.revision9.veins.ci9 h6 --cell rural_highway 20 --seed 223 --workers 4
    python -m experiments.revision9.veins.ci9 c1 --cell urban_nlos 20 --seed 227 --workers 4

h6: STRUCT9 plain_s0-4 and rand_s0-4 (results/revision9/train); c1: revision-8 STRUCT plain_s0-4 and rand_s0-4
(results/revision8/train). 60 replays per shard (10 runs x delta -2..3). Every shard first:
  * restores the committed AI-HARP checkpoints, each hash checked against results/revision5/checkpoint_shas.json;
  * gate check 1: classical slotted_1p n=8, rural d=20, seed 200, event log byte-identical to 707d6533...;
  * gate check 2: AI-HARP ref_s0 bias +1, every new option off, byte-identical to 220dd1fd...
    (results/revision7/veins/raw/r7_gate/);
and contributes nothing if either fails. Afterwards every decision log is re-sent offline (0 mismatches required) and
replays are scored if exit 0, identical SUMO trace and this runner's gated binary.
Outputs: results/revision9/veins/<mode>/<cell>_s<seed>/: gate.json, trace_check.json, rows.jsonl, merge.json, raw/,
bridge_logs/ (never committed).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

from common.config import PROJECT_ROOT
from experiments.revision8.veins.ci_veins import (APP, BROKER, BUNDLES, DELTAS, GATE2_REF, GATE_SHA, bundles, replay,
                                                  restore_checkpoints, score_dir, sha_events)

RUNS = [f"{arm}_s{k}" for arm in ("plain", "rand") for k in range(5)]
TRAIN = {"h6": "results/revision9/train", "c1": "results/revision8/train"}


def bridged9(items, workers: int, out: Path) -> list:
    lock, it, done = threading.Lock(), iter(items), []

    def worker(k: int) -> None:
        app_port, srv_port = 5801 + k, 5901 + k
        cfgp = out / f"worker{k}.json"
        broker = subprocess.Popen([sys.executable, str(BROKER), str(app_port), str(srv_port)],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(2)
        srv = subprocess.Popen([sys.executable, "-u", "-m", "experiments.revision9.veins.decision_server9", "--connect",
                                f"127.0.0.1:{srv_port}", "--config", str(cfgp)], cwd=PROJECT_ROOT,
                               stdout=open(out / f"server{k}.log", "a"), stderr=subprocess.STDOUT)
        try:
            while True:
                with lock:
                    t = next(it, None)
                if t is None:
                    return
                bdir = t["bdir"]
                name = f"{bdir.name}__{t['policy']}_{t['knobs']}__s{t['seed']}"
                t["logs"].mkdir(parents=True, exist_ok=True)
                nveh = len(json.loads((bdir / "bundle.json").read_text())["vehicles"])
                sc_, rest = bdir.name.split("_d")
                cfgp.write_text(json.dumps({"name": name, "scenario": sc_, "density": float(rest.split("_s")[0]),
                                            "seed": t["seed"], "n_vehicles": nveh,
                                            "log": str(t["logs"] / f"{name}.jsonl"), **t["cfg"]}))
                r = replay(bdir, t["policy"], t["knobs"], t["seed"], t["raw"], "--bridge", f"127.0.0.1:{app_port}")
                with lock:
                    done.append(r)
                    print(f"{time.strftime('%H:%M:%S')} {len(done)}/{len(items)} {name} exit {r.get('exit_code')}",
                          flush=True)
        finally:
            srv.terminate()
            broker.terminate()

    th = [threading.Thread(target=worker, args=(k,)) for k in range(workers)]
    for x in th:
        x.start()
    for x in th:
        x.join()
    return done


def main(argv=None) -> int:
    import gzip
    import shutil

    import numpy
    from experiments.revision9.veins.decision_server9 import trace
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=["h6", "c1"])
    ap.add_argument("--cell", nargs=2, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)
    sc, d, seed = a.cell[0], float(a.cell[1]), a.seed
    print(f"[env] {sys.executable} numpy {numpy.__version__}", flush=True)
    out = PROJECT_ROOT / "results" / "revision9" / "veins" / a.mode / f"{sc}_d{d:g}_s{seed}"
    out.mkdir(parents=True, exist_ok=True)
    for run in RUNS:                                      # every STRUCT run must exist and be complete
        t = json.loads((PROJECT_ROOT / TRAIN[a.mode] / run / "theta.json").read_text())
        assert t["generations"] == 80, (run, t["generations"])
    ck = restore_checkpoints()
    bundles({(("rural_highway", 20.0), 200), ((sc, d), seed)})
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
    gate = {"check1": {"events_sha256": s1, "reference_sha256": GATE_SHA, "pass": s1 == GATE_SHA},
            "check2": {"events_sha256": s2, "reference_sha256": ref2, "pass": s2 == ref2},
            "checkpoints": ck, "app_sha256": app_sha}
    gate["pass"] = gate["check1"]["pass"] and gate["check2"]["pass"]
    (out / "gate.json").write_text(json.dumps(gate, indent=1))
    print(json.dumps(gate), flush=True)
    shutil.rmtree(out / "gate", ignore_errors=True)
    shutil.rmtree(out / "gate2", ignore_errors=True)
    shutil.rmtree(out / "gate2_logs", ignore_errors=True)
    if not gate["pass"]:
        return 2
    bdir = BUNDLES / f"{sc}_d{d:g}_s{seed}"
    items = [{"bdir": bdir, "seed": seed, "policy": "struct", "knobs": f"{run}_d{dl:g}",
              "cfg": {"policy": "struct", "run": run, "delta": dl, "train_dir": TRAIN[a.mode]},
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
                                                "train_dir": TRAIN[a.mode]}, indent=1))
    ok = len(rows) == len(items) and len(tc) == len(items) and all(x["pass"] for x in tc)
    print(f"[ci9] {a.mode} {sc} d={d:g} s={seed}: merged {len(rows)}, trace {'pass' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
