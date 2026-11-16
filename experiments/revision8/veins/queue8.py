"""Revision 8: the Veins replays of H5 and Experiments C and D, resumably (Windows side; launches WSL jobs).
results/revision8/ADDENDUM.md sections 6 and 8; results/revision8/LAPTOP_RUNBOOK.md.

    python -m experiments.revision8.veins.queue8 gate             validation gate without STRUCT -> app_gate.json
    python -m experiments.revision8.veins.queue8 gate_struct      STRUCT replay + trace check -> app_gate_struct.json (H5 needs it)
    python -m experiments.revision8.veins.queue8 h5B --jobs 8     STRUCT plain/rand s0-4, delta -2..3, Tier B 210-219 (1,200; D12)
    python -m experiments.revision8.veins.queue8 h5A --jobs 8     the same on Tier A 200-209 (1,200; not run under D12)
    python -m experiments.revision8.veins.queue8 expC --jobs 8    AI-HARP ref_s0-4, geometric table, rural d=20, Tier A (300; D12)
    python -m experiments.revision8.veins.queue8 expD --jobs 8    AI-HARP ref_s0-4, lag removed, rural d=20, Tier A (300; D12)
    python -m experiments.revision8.veins.queue8 status

Cells: H5 rural d=20 and urban NLOS d=20; Experiments C and D rural d=20 only (D12). Binary ~/veins-r8/aiharp (sha256 in app_gate.json, written only if every gate check
passes; no replay mode starts without it). A replay is done when its record shows exit code 0, an identical trace and
that binary; anything else is kept and retried. Outputs: cache/veins_runs/r8_<mode>; decision logs
D:\\aiharp-r8-bridge-logs\\<mode>.
"""

from __future__ import annotations

import argparse
import ctypes
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
from experiments.revision6.veins import queue6 as q6

ROOT = PROJECT_ROOT / "cache" / "veins_runs"
OUT = PROJECT_ROOT / "results" / "revision8" / "veins"
GATE = OUT / "app_gate.json"
LOGS = Path(r"D:\aiharp-r8-bridge-logs")
APP = "~/veins-r8/aiharp"
APP_ABS = "/home/upasana/veins-r8/aiharp/aiharp"
RUNNER = f"{q6.WSL_ROOT}/experiments/revision8/veins/run_veins8.py"
CELLS = (("rural_highway", 20.0), ("urban_nlos", 20.0))
TIER_A, TIER_B = list(range(200, 210)), list(range(210, 220))       # D12: Tier B cut to 210-219
CD_CELLS = (("rural_highway", 20.0),)                                  # D12: Experiments C and D in rural d=20 only
DELTAS = (-2.0, -1.0, 0.0, 1.0, 2.0, 3.0)
STRUCT_RUNS = [f"{arm}_s{k}" for arm in ("plain", "rand") for k in range(5)]      # fixed in the addendum
REF_RUNS = [f"ref_s{k}" for k in range(5)]
GATE_SLOTTED_SHA = "707d65331f9ecee307f321b38f6c6ded9989bec47320a4a844616875ccf2fba8"   # revision-6 gate log
SC, D, S = "rural_highway", 20.0, 200


def keep_awake() -> None:
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)


def tasks(mode: str) -> list[tuple]:
    """(scenario, density, seed, policy, knobs-tag, server cfg extras, runner flags)."""
    if mode in ("h5A", "h5B"):
        seeds = TIER_A if mode == "h5A" else TIER_B
        return [(sc, d, s, "struct", f"{run}_d{dl:g}", {"policy": "struct", "run": run, "delta": dl}, ())
                for run in STRUCT_RUNS for dl in DELTAS for sc, d in CELLS for s in seeds]
    flags = {"expC": ("--send-heard",), "expD": ("--step-offset", "1")}[mode]
    extra = {"neighbour_table": "geometric"} if mode == "expC" else {}
    return [(sc, d, s, "ai_harp", f"{run}_b{b:g}", {"policy": "ai_harp", "run": run, "bias": b, "tau": 0.5, **extra},
             flags) for run in REF_RUNS for b in q6.BIASES for sc, d in CD_CELLS for s in TIER_A]


def name_of(t) -> str:
    return q6.name_of(*t[:5])


def done(out: Path, name: str, app_sha: str) -> bool:
    p = out / f"{name}.run.json"
    if not p.exists():
        return False
    r = json.loads(p.read_text())
    return r.get("exit_code") == 0 and bool(r.get("fcd_check", {}).get("identical")) and r.get("app_sha256") == app_sha


def runner_cmd(bdir: Path, pol: str, kn: str, s: int, out: Path, *flags: str) -> list[str]:
    return ["wsl", "-d", "Ubuntu", "--", "python3", RUNNER, q6.wsl(bdir), pol, kn, str(s), q6.wsl(out), "--app", APP,
            *flags]


class Worker8:
    """A WSL broker and a revision-8 decision server, reused across this worker's replays."""

    def __init__(self, k: int, logs: Path):
        self.k, self.app_port, self.srv_port = k, 5801 + k, 5901 + k
        self.logs = logs
        self.cfg = q6.BRIDGE / f"worker8_{k}.json"
        q6.BRIDGE.mkdir(parents=True, exist_ok=True)
        (PROJECT_ROOT / "logs" / "revision8").mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, SUMO_HOME=os.environ.get("SUMO_HOME", r"D:\sumo\sumo-1.19.0"))
        log = open(PROJECT_ROOT / "logs" / "revision8" / f"server8_worker{k}.log", "a")
        self.broker = subprocess.Popen(["wsl", "-d", "Ubuntu", "--", "python3", q6.BROKER, str(self.app_port),
                                        str(self.srv_port)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(3)
        self.server = subprocess.Popen([q6.PY, "-u", "-m", "experiments.revision8.veins.decision_server8", "--connect",
                                        f"127.0.0.1:{self.srv_port}", "--config", str(self.cfg)], cwd=PROJECT_ROOT,
                                       env=env, stdout=log, stderr=subprocess.STDOUT)

    def run(self, t, out: Path) -> str:
        sc, d, s, pol, kn, extra, flags = t
        name = name_of(t)
        bdir = q6.BUNDLES / f"{sc}_d{d:g}_s{s}"
        nveh = len(json.loads((bdir / "bundle.json").read_text())["vehicles"])
        plain = q6.BRIDGE / f"{name}.jsonl"
        plain.unlink(missing_ok=True)
        self.cfg.write_text(json.dumps({"name": name, "scenario": sc, "density": d, "seed": s, "n_vehicles": nveh,
                                        "log": str(plain), **extra}))
        subprocess.run(runner_cmd(bdir, pol, kn, s, out, "--bridge", f"127.0.0.1:{self.app_port}", *flags),
                       capture_output=True, text=True)
        if plain.exists():
            self.logs.mkdir(parents=True, exist_ok=True)
            for _ in range(20):
                try:
                    with open(plain, "rb") as fi, gzip.open(self.logs / f"{name}.jsonl.gz", "wb") as fo:
                        shutil.copyfileobj(fi, fo)
                    plain.unlink()
                    break
                except PermissionError:
                    time.sleep(1)
        return name

    def close(self) -> None:
        for p in (self.server, self.broker):
            p.terminate()


# ------------------------------------------------------------------------------------------------- gate --
def sha_events(path: Path) -> str:
    with gzip.open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def wsl_sha() -> str:
    r = subprocess.run(["wsl", "-d", "Ubuntu", "--", "sha256sum", APP_ABS], capture_output=True, text=True)
    return r.stdout.split()[0]


def gate(only_struct: bool = False) -> dict:
    """Before any reported replay: every new option off reproduces stored logs byte for byte; one STRUCT replay, one
    Exp C replay and one Exp D replay run cleanly; the STRUCT and Exp C decision logs are reproduced offline exactly."""
    from experiments.revision8.veins import validate8 as v8
    out = ROOT / "r8_gate"
    out.mkdir(parents=True, exist_ok=True)
    checks: dict[str, dict] = {}
    if not only_struct:
        name = q6.name_of(SC, D, S, "slotted_1p", "8")
        if not (out / f"{name}.events.csv.gz").exists():
            subprocess.run(runner_cmd(q6.BUNDLES / f"{SC}_d{D:g}_s{S}", "slotted_1p", "8", S, out), capture_output=True,
                           text=True)
        sha = sha_events(out / f"{name}.events.csv.gz")
        checks["slotted_1p_off_vs_revision6_gate"] = {"replay": name, "events_sha256": sha,
                                                      "reference_sha256": GATE_SLOTTED_SHA,
                                                      "pass": sha == GATE_SLOTTED_SHA}
    w = Worker8(0, LOGS / "gate")
    try:
        cases = {
            "ai_harp_off_vs_revision6_replay": (SC, D, S, "ai_harp", "ref_s0_b1",
                                                {"policy": "ai_harp", "run": "ref_s0", "bias": 1.0, "tau": 0.5}, ()),
            "struct_replay": (SC, D, S, "struct", "plain_s0_d0", {"policy": "struct", "run": "plain_s0", "delta": 0.0}, ()),
            "expC_replay": (SC, D, S, "ai_harp", "ref_s0_b1", {"policy": "ai_harp", "run": "ref_s0", "bias": 1.0,
                                                               "tau": 0.5, "neighbour_table": "geometric"},
                            ("--send-heard",)),
            "expD_replay": (SC, D, S, "ai_harp", "ref_s0_b1", {"policy": "ai_harp", "run": "ref_s0", "bias": 1.0,
                                                               "tau": 0.5}, ("--step-offset", "1")),
        }
        # D12 scheduling: "gate" (no STRUCT; Experiments C/D may start) and "gate_struct" (once plain_s0 is trained;
        # required before H5) are run separately; together they are the registered gate.
        cases = {k: v for k, v in cases.items() if (k == "struct_replay") == only_struct}
        for label, t in cases.items():
            o = out / label
            o.mkdir(exist_ok=True)
            w.logs = LOGS / "gate" / label
            nm = name_of(t)
            if not (o / f"{nm}.run.json").exists():
                w.run(t, o)
            rec = json.loads((o / f"{nm}.run.json").read_text())
            c = {"replay": nm, "exit_code": rec["exit_code"], "trace_identical": rec["fcd_check"].get("identical"),
                 "events_sha256": sha_events(o / f"{nm}.events.csv.gz")}
            if label == "ai_harp_off_vs_revision6_replay":
                c["reference_sha256"] = sha_events(q6.RUNS / f"{nm}.events.csv.gz")
                c["pass"] = c["events_sha256"] == c["reference_sha256"]
            else:
                c["pass"] = c["exit_code"] == 0 and bool(c["trace_identical"])
                if label in ("struct_replay", "expC_replay"):
                    tr = v8.trace(w.logs / f"{nm}.jsonl.gz")
                    c["trace_check"] = tr
                    c["pass"] = c["pass"] and tr["pass"]
                if label == "struct_replay":         # STRUCT never gets a duplicate callback
                    with gzip.open(o / f"{nm}.events.csv.gz", "rt") as f:
                        dups = sum(1 for l in f if l.startswith("D,") and ",receive," in l)
                    c["receive_decisions"] = dups
            checks[label] = c
    finally:
        w.close()
    return checks


def main(argv=None) -> int:
    import numpy
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=["gate", "gate_struct", "h5B", "h5A", "expC", "expD", "status"])
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args(argv)
    print(f"[env] {sys.executable} numpy {numpy.__version__}", flush=True)
    os.environ.setdefault("SUMO_HOME", r"D:\sumo\sumo-1.19.0")
    OUT.mkdir(parents=True, exist_ok=True)
    if a.mode in ("gate", "gate_struct"):
        checks = gate(only_struct=a.mode == "gate_struct")
        ok = all(c["pass"] for c in checks.values())
        print(json.dumps(checks, indent=1, default=str))
        from experiments.revision1.provenance import stamp
        sha = wsl_sha()
        if a.mode == "gate_struct" and GATE.exists() and json.loads(GATE.read_text())["app_sha256"] != sha:
            ok = False
            checks["same_binary_as_gate"] = {"pass": False}
        payload = {"provenance": stamp(f"revision8/veins/{a.mode}", {"replay": f"{SC} d={D:g} seed {S}"}), "pass": ok,
                   "app_sha256": sha, "checks": checks}
        stem = "app_gate" if a.mode == "gate" else "app_gate_struct"
        (OUT / (f"{stem}.json" if ok else f"{stem}_FAILED.json")).write_text(json.dumps(payload, indent=1, default=str))
        return 0 if ok else 1
    if not GATE.exists():
        raise SystemExit("no results/revision8/veins/app_gate.json: run the gate first; no replay before it passes")
    if a.mode in ("h5A", "h5B") and not (OUT / "app_gate_struct.json").exists():
        raise SystemExit("no results/revision8/veins/app_gate_struct.json: run `queue8 gate_struct` (needs plain_s0 trained)")
    app_sha = json.loads(GATE.read_text())["app_sha256"]
    for sc, d in CELLS:
        for s in TIER_A + TIER_B:                     # r6.txt needed for 200-219 (D12)
            if not (q6.BUNDLES / f"{sc}_d{d:g}_s{s}" / "r6.txt").exists():
                raise SystemExit(f"{sc} d={d:g} s={s}: no r6.txt (python -m experiments.revision6.veins.r6_bundle)")
    if a.mode == "status":
        for m in ("h5B", "h5A", "expC", "expD"):
            ts = tasks(m)
            print(f"{m}: {sum(done(ROOT / f'r8_{m}', name_of(t), app_sha) for t in ts)} / {len(ts)} done", flush=True)
        return 0
    keep_awake()
    out = ROOT / f"r8_{a.mode}"
    out.mkdir(parents=True, exist_ok=True)
    ts = tasks(a.mode)
    todo = [t for t in ts if not done(out, name_of(t), app_sha)]
    print(f"[queue8] {a.mode}: {len(ts)} replays, {len(todo)} to do, {a.jobs} workers", flush=True)
    t0 = time.time()
    workers = [Worker8(k, LOGS / a.mode) for k in range(a.jobs)]
    lock, it, n = threading.Lock(), iter(todo), [0]

    def loop(w):
        while True:
            with lock:
                t = next(it, None)
            if t is None:
                return
            name = w.run(t, out)
            with lock:
                n[0] += 1
                print(f"{time.strftime('%H:%M:%S')} {n[0]}/{len(todo)} {name} "
                      f"{'ok' if done(out, name, app_sha) else 'FAILED (kept, will retry)'}", flush=True)
    try:
        th = [threading.Thread(target=loop, args=(w,)) for w in workers]
        for x in th:
            x.start()
        for x in th:
            x.join()
    finally:
        for w in workers:
            w.close()
    print(f"[queue8] {a.mode} pass finished in {(time.time() - t0) / 3600:.2f} wall-hours", flush=True)
    left = [t for t in ts if not done(out, name_of(t), app_sha)]
    if not left:
        (out / "_DONE").write_text(time.strftime("%Y-%m-%dT%H:%M:%S"))
    print(f"[queue8] {len(left)} replays still not done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
