"""Revision 8 (DEVIATIONS.md D12-D14): one H5 Tier B shard on a Linux runner -- one (cell, replay seed), all 60 STRUCT
replays (plain_s0-4 and rand_s0-4 x delta in {-2..3}).

    python -m experiments.revision8.veins.ci_veins --cell rural_highway 20 --seed 213 --workers 4

Runs natively on Linux (no WSL): OMNeT++ 6.0.3 / Veins 5.3 / SUMO 1.19.0 from experiments/revision5/veins/install_wsl.sh
in ~/veins-r5, the revision-8 app in ~/veins-r8/aiharp (experiments/revision8/veins/build_app.sh).
  1. Bundles: exports the shard's bundle and the gate bundle (rural d=20, seed 200) with the source-built SUMO
     (experiments.revision5.veins.export, identity-checked against the canonical trace), then r6.txt.
  2. Gate (D13/D14): slotted_1p n=8, rural d=20, seed 200, every new option off: the event log must be byte-identical to
     the revision-6 gate log (707d6533...). If not, the shard stops and contributes nothing.
  3. Replays: one bridge broker + decision_server8 per worker (as queue8 on the laptop), run_veins8.py per replay.
  4. Trace check: every decision log of the shard is re-sent offline to a fresh Session8 (validate8.trace); all must
     reproduce exactly.
  5. Scoring: only replays with exit code 0, an identical SUMO trace and the gated binary, with the existing scorer.
Outputs in results/revision8/veins/ci/<cell>_s<seed>/: gate.json, trace_check.json, rows.jsonl, merge.json, raw/.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from common.config import PROJECT_ROOT

HOME = Path.home()
VR5 = HOME / "veins-r5"
APP = HOME / "veins-r8" / "aiharp"
RUNNER = PROJECT_ROOT / "experiments" / "revision8" / "veins" / "run_veins8.py"
BROKER = PROJECT_ROOT / "experiments" / "revision6" / "veins" / "bridge_broker.py"
BUNDLES = PROJECT_ROOT / "cache" / "veins"
GATE_SHA = "707d65331f9ecee307f321b38f6c6ded9989bec47320a4a844616875ccf2fba8"
RUNS = [f"{arm}_s{k}" for arm in ("plain", "rand") for k in range(5)]
DELTAS = (-2.0, -1.0, 0.0, 1.0, 2.0, 3.0)


def sh(cmd, **kw):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run([str(c) for c in cmd], **kw)


def sha_events(p: Path) -> str:
    with gzip.open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def bundles(cells_seeds) -> None:
    env = dict(os.environ, SUMO_HOME=str(VR5 / "sumo-1.19.0"))
    for (sc, d), s in cells_seeds:
        if not (BUNDLES / f"{sc}_d{d:g}_s{s}" / "bundle.json").exists():
            r = sh([sys.executable, "-m", "experiments.revision5.veins.export", sc, f"{d:g}", s], env=env)
            if r.returncode:
                raise SystemExit(f"export failed for {sc} d={d:g} s={s}")
    seeds = sorted({s for _, s in cells_seeds})
    for s in seeds:
        sh([sys.executable, "-m", "experiments.revision6.veins.r6_bundle", "--seeds", f"{s}-{s}"], env=env, check=True)


def replay(bundle: Path, policy: str, knobs: str, seed: int, out: Path, *flags) -> dict:
    sh([sys.executable, RUNNER, bundle, policy, knobs, seed, out, "--app", APP, *flags], capture_output=True)
    name = f"{bundle.name}__{policy}_{knobs.replace(',', '_')}__s{seed}"
    p = out / f"{name}.run.json"
    return {"name": name, **(json.loads(p.read_text()) if p.exists() else {"exit_code": None})}


def main(argv=None) -> int:
    import numpy
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cell", nargs=2, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)
    sc, d, seed = a.cell[0], float(a.cell[1]), a.seed
    print(f"[env] {sys.executable} numpy {numpy.__version__}", flush=True)
    out = PROJECT_ROOT / "results" / "revision8" / "veins" / "ci" / f"{sc}_d{d:g}_s{seed}"
    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    bundles({(("rural_highway", 20.0), 200), ((sc, d), seed)})
    app_sha = hashlib.sha256((APP / "aiharp").read_bytes()).hexdigest()

    # 2. gate
    g = replay(BUNDLES / "rural_highway_d20_s200", "slotted_1p", "8", 200, out / "gate")
    ev = out / "gate" / f"{g['name']}.events.csv.gz"
    gsha = sha_events(ev) if ev.exists() else None
    gate = {"replay": g["name"], "exit_code": g.get("exit_code"), "events_sha256": gsha, "reference_sha256": GATE_SHA,
            "app_sha256": app_sha, "pass": gsha == GATE_SHA}
    (out / "gate.json").write_text(json.dumps(gate, indent=1))
    print(json.dumps(gate), flush=True)
    if not gate["pass"]:
        return 2

    # 3. replays, one broker + decision server per worker
    logs = out / "bridge_logs"
    logs.mkdir(exist_ok=True)
    tasks = [(run, dl) for run in RUNS for dl in DELTAS]
    bdir = BUNDLES / f"{sc}_d{d:g}_s{seed}"
    nveh = len(json.loads((bdir / "bundle.json").read_text())["vehicles"])
    lock, it, done = threading.Lock(), iter(tasks), []

    def worker(k: int) -> None:
        app_port, srv_port = 5801 + k, 5901 + k
        cfg = out / f"worker{k}.json"
        broker = subprocess.Popen([sys.executable, str(BROKER), str(app_port), str(srv_port)],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(2)
        srv = subprocess.Popen([sys.executable, "-u", "-m", "experiments.revision8.veins.decision_server8", "--connect",
                                f"127.0.0.1:{srv_port}", "--config", str(cfg)], cwd=PROJECT_ROOT,
                               stdout=open(out / f"server{k}.log", "a"), stderr=subprocess.STDOUT)
        try:
            while True:
                with lock:
                    t = next(it, None)
                if t is None:
                    return
                run, dl = t
                knobs = f"{run}_d{dl:g}"
                name = f"{bdir.name}__struct_{knobs}__s{seed}"
                log = logs / f"{name}.jsonl"
                cfg.write_text(json.dumps({"name": name, "policy": "struct", "run": run, "delta": dl, "scenario": sc,
                                           "density": d, "seed": seed, "n_vehicles": nveh, "log": str(log)}))
                r = replay(bdir, "struct", knobs, seed, raw, "--bridge", f"127.0.0.1:{app_port}")
                with lock:
                    done.append(r)
                    print(f"{time.strftime('%H:%M:%S')} {len(done)}/{len(tasks)} {name} exit {r.get('exit_code')}",
                          flush=True)
        finally:
            srv.terminate()
            broker.terminate()

    th = [threading.Thread(target=worker, args=(k,)) for k in range(a.workers)]
    for x in th:
        x.start()
    for x in th:
        x.join()

    # 4. trace check of every decision log
    from experiments.revision8.veins.validate8 import trace
    tc = []
    for p in sorted(logs.glob("*.jsonl")):
        res = trace(p)
        tc.append(res)
        with open(p, "rb") as fi, gzip.open(str(p) + ".gz", "wb") as fo:
            shutil.copyfileobj(fi, fo)
        p.unlink()
    (out / "trace_check.json").write_text(json.dumps({"logs": tc, "all_pass": all(x["pass"] for x in tc),
                                                       "n": len(tc)}, indent=1))

    # 5. scoring (accepted: exit 0, identical trace, this runner's gated binary)
    from experiments.revision6.veins import veins6_score as s6
    rows, rejected = [], {}
    for p in sorted(raw.glob("*.run.json")):
        rec = json.loads(p.read_text())
        name = p.name[: -len(".run.json")]
        ok = rec.get("exit_code") == 0 and rec.get("fcd_check", {}).get("identical") and rec.get("app_sha256") == app_sha
        if not ok:
            rejected[name] = {"exit_code": rec.get("exit_code"), "fcd": rec.get("fcd_check", {}).get("identical")}
            continue
        rows.append({"name": name, "row": s6.score(p)})
    with open(out / "rows.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")
    (out / "merge.json").write_text(json.dumps({"merged": len(rows), "rejected": len(rejected), "expected": len(tasks),
                                                "rejected_replays": rejected, "app_sha256": app_sha}, indent=1))
    print(f"[ci_veins] {sc} d={d:g} s={seed}: merged {len(rows)}, rejected {len(rejected)}, trace check "
          f"{'pass' if all(x['pass'] for x in tc) else 'FAIL'} ({len(tc)} logs)", flush=True)
    return 0 if len(rows) == len(tasks) and all(x["pass"] for x in tc) else 3


if __name__ == "__main__":
    raise SystemExit(main())
