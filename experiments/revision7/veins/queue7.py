"""Revision 7: the Veins replays of Experiments A and B, resumably (Windows side; launches WSL jobs). results/revision7/ADDENDUM.md.

    python -m experiments.revision7.veins.queue7 capped --jobs 8     A: learned policy with max_decisions_per_epoch = 1
    python -m experiments.revision7.veins.queue7 d9 --jobs 11        B: greedy_farthest and dvcast with the D9 fix
    python -m experiments.revision7.veins.queue7 status

Reuses experiments/revision6/veins (runner, broker, decision server, queue6.Worker) with the revision-7 binary
(~/veins-r7/aiharp, sha256 in results/revision7/veins/app_gate.json). A replay is done when its record shows exit code 0, an
identical trace and that binary; anything else is kept and retried. Outputs: cache/veins_runs/r7_capped, r7_d9.
"""

from __future__ import annotations

import argparse
import ctypes
import gzip
import json
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from common.config import PROJECT_ROOT
from experiments.revision6.veins import queue6 as q6

ROOT = PROJECT_ROOT / "cache" / "veins_runs"
GATE = PROJECT_ROOT / "results" / "revision7" / "veins" / "app_gate.json"
BRIDGE_LOGS = Path(r"D:\aiharp-r7-bridge-logs")
APP = "~/veins-r7/aiharp"
RUNS = [f"ref_s{k}" for k in range(5)]
SEEDS = list(range(200, 210))
CELLS_A = (("rural_highway", 20.0), ("urban_nlos", 20.0))
CELLS_B = (("rural_highway", 2.0), ("rural_highway", 20.0), ("urban_nlos", 20.0))
FAMILIES_B = ("greedy_farthest", "dvcast")


def keep_awake() -> None:
    """Hold off idle sleep while this process runs; no power setting is changed. Lid-close and requested sleep still apply."""
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)   # ES_CONTINUOUS | ES_SYSTEM_REQUIRED


def d9_settings() -> list[tuple[str, str]]:
    """(policy, knob) of both families as the revision-5 Veins rows ran them."""
    out = set()
    for line in (PROJECT_ROOT / "results" / "revision5" / "veins" / "veins_rows.jsonl").open(encoding="utf-8"):
        r = json.loads(line)["row"]
        if r["policy"] in FAMILIES_B:
            out.add((r["policy"], f"{float(r['knob']):g}"))
    return sorted(out, key=lambda x: (x[0], float(x[1])))


def tasks_capped() -> list[tuple]:
    return [(sc, d, s, "ai_harp", f"{run}_b{b:g}", (run, b))
            for run in RUNS for b in q6.BIASES for sc, d in CELLS_A for s in SEEDS]


def tasks_d9() -> list[tuple]:
    return [(sc, d, s, pol, kn, None) for sc, d in CELLS_B for s in SEEDS for pol, kn in d9_settings()]


def done(out: Path, name: str, app_sha: str) -> bool:
    p = out / f"{name}.run.json"
    if not p.exists():
        return False
    r = json.loads(p.read_text())
    return r.get("exit_code") == 0 and bool(r.get("fcd_check", {}).get("identical")) and r.get("app_sha256") == app_sha


def runner_cmd(bdir: Path, pol: str, kn: str, s: int, out: Path, *flags: str) -> list[str]:
    return ["wsl", "-d", "Ubuntu", "--", "python3", q6.RUNNER, q6.wsl(bdir), pol, kn, str(s), q6.wsl(out), "--app", APP, *flags]


class Worker7(q6.Worker):
    """queue6.Worker's replay, on the revision-7 binary."""

    logs = BRIDGE_LOGS

    def run7(self, t, out: Path, *flags: str) -> str:
        sc, d, s, pol, kn, (run, b) = t
        name = q6.name_of(sc, d, s, pol, kn)
        bdir = q6.BUNDLES / f"{sc}_d{d:g}_s{s}"
        nveh = len(json.loads((bdir / "bundle.json").read_text())["vehicles"])
        plain = q6.BRIDGE / f"{name}.jsonl"
        plain.unlink(missing_ok=True)
        self.cfg.write_text(json.dumps({"name": name, "run": run, "bias": b, "tau": 0.5, "scenario": sc, "density": d,
                                        "seed": s, "n_vehicles": nveh, "log": str(plain)}))
        subprocess.run(runner_cmd(bdir, pol, kn, s, out, "--bridge", f"127.0.0.1:{self.app_port}", *flags),
                       capture_output=True, text=True)
        if plain.exists():
            self.logs.mkdir(parents=True, exist_ok=True)
            for _ in range(20):                      # the server closes the log right after the app disconnects
                try:
                    with open(plain, "rb") as fi, gzip.open(self.logs / f"{name}.jsonl.gz", "wb") as fo:
                        shutil.copyfileobj(fi, fo)
                    plain.unlink()
                    break
                except PermissionError:
                    time.sleep(1)
        return name


def main(argv=None) -> int:
    import numpy
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=["capped", "d9", "status"])
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--reverse", action="store_true", help="capped: work from the end of the list (a second queue beside a running one)")
    ap.add_argument("--first-worker", type=int, default=0, help="capped: first worker number, so a second queue has its own ports and config files")
    a = ap.parse_args(argv)
    print(f"[env] {sys.executable} numpy {numpy.__version__}", flush=True)
    app_sha = json.loads(GATE.read_text())["app_sha256"]
    plans = {"capped": (tasks_capped(), ROOT / "r7_capped"), "d9": (tasks_d9(), ROOT / "r7_d9")}
    if a.mode == "status":
        for m, (ts, out) in plans.items():
            print(f"{m}: {sum(done(out, q6.name_of(*t[:5]), app_sha) for t in ts)} / {len(ts)} done", flush=True)
        return 0
    keep_awake()
    ts, out = plans[a.mode]
    out.mkdir(parents=True, exist_ok=True)
    todo = [t for t in ts if not done(out, q6.name_of(*t[:5]), app_sha)]
    if a.reverse:
        todo.reverse()
    print(f"[queue7] {a.mode}: {len(ts)} replays, {len(todo)} to do, {a.jobs} workers", flush=True)
    t0 = time.time()
    if a.mode == "d9":
        def one(t):
            sc, d, s, pol, kn, _ = t
            subprocess.run(runner_cmd(q6.BUNDLES / f"{sc}_d{d:g}_s{s}", pol, kn, s, out, "--beacon-cam-velocity"),
                           capture_output=True, text=True)
            return q6.name_of(*t[:5])
        with ThreadPoolExecutor(a.jobs) as ex:
            for i, name in enumerate(ex.map(one, todo), 1):
                print(f"{time.strftime('%H:%M:%S')} {i}/{len(todo)} {name} "
                      f"{'ok' if done(out, name, app_sha) else 'FAILED (kept, will retry)'}", flush=True)
    else:
        workers = [Worker7(k) for k in range(a.first_worker, a.first_worker + a.jobs)]
        lock, it, n = threading.Lock(), iter(todo), [0]

        def loop(w):
            while True:
                with lock:
                    t = next(it, None)
                if t is None:
                    return
                if a.reverse:                        # never touch a replay the other queue finished or is running
                    nm = q6.name_of(*t[:5])
                    if done(out, nm, app_sha) or (q6.BRIDGE / f"{nm}.jsonl").exists():
                        continue
                name = w.run7(t, out, "--max-decisions-per-epoch", "1")
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
    print(f"[queue7] {a.mode} pass finished in {(time.time() - t0) / 3600:.2f} wall-hours", flush=True)
    left = [t for t in ts if not done(out, q6.name_of(*t[:5]), app_sha)]
    if not left:
        (out / "_DONE").write_text(time.strftime("%Y-%m-%dT%H:%M:%S"))
    print(f"[queue7] {len(left)} replays still not done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
