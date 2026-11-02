"""Revision 6 Step 4: run the rule and policy replays in Veins, resumably (Windows side; launches WSL jobs).

    python -m experiments.revision6.veins.queue6 rules  --seeds 200-209 --jobs 6
    python -m experiments.revision6.veins.queue6 policy --runs ref_s0 ref_s1 ref_s2 ref_s3 ref_s4 --seeds 200-209 --jobs 4
    python -m experiments.revision6.veins.queue6 status --seeds 200-209

Cells: the 4 headline cells. Rules: the 39 pre-registered settings (experiments/revision5/sweep.py rule_settings).
Policy: each training run x suppression bias {-2, ..., 3} (or --biases, for the pre-registered cut) at tau 0.5.
A replay is done when its run record shows exit code 0, an identical trace and the gated app binary
(results/revision6/veins/app_gate.json); anything else is kept and retried. Policy workers: worker k owns a WSL broker
(app port 5801+k, server port 5901+k; bridge_broker.py) and a decision server in the pinned interpreter
(decision_server.py); before each replay the worker's configuration file is rewritten; the server's request log is
gzipped to D:\\aiharp-r6-bridge-logs afterwards. Outputs: cache/veins_runs/r6/.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from common.config import PROJECT_ROOT
from experiments.revision5.sweep import HEADLINE, rule_settings

BUNDLES = PROJECT_ROOT / "cache" / "veins"
RUNS = PROJECT_ROOT / "cache" / "veins_runs" / "r6"
BRIDGE = PROJECT_ROOT / "cache" / "veins_runs" / "r6_bridge"
BRIDGE_LOGS = Path(r"D:\aiharp-r6-bridge-logs")
GATE = PROJECT_ROOT / "results" / "revision6" / "veins" / "app_gate.json"
WSL_ROOT = "/mnt/c/Users/UPASANA/AI-HARP"
RUNNER = f"{WSL_ROOT}/experiments/revision6/veins/run_veins6.py"
BROKER = f"{WSL_ROOT}/experiments/revision6/veins/bridge_broker.py"
BIASES = (-2.0, -1.0, 0.0, 1.0, 2.0, 3.0)
PY = sys.executable


def wsl(path: Path) -> str:
    return WSL_ROOT + "/" + path.relative_to(PROJECT_ROOT).as_posix()


def rule_knobs() -> list[tuple[str, str]]:
    out = []
    for pol, _, val in rule_settings():
        out.append((pol, ",".join(repr(float(v)) if pol == "rar" else str(v) for v in json.loads(val))))
    return out


def name_of(sc, d, seed, pol, knobs) -> str:
    return f"{sc}_d{d:g}_s{seed}__{pol}_{knobs.replace(',', '_')}__s{seed}"


def done(name: str, app_sha: str) -> bool:
    p = RUNS / f"{name}.run.json"
    if not p.exists():
        return False
    r = json.loads(p.read_text())
    return r.get("exit_code") == 0 and bool(r.get("fcd_check", {}).get("identical")) and r.get("app_sha256") == app_sha


def seeds_of(s: str) -> list[int]:
    lo, hi = (int(x) for x in s.split("-"))
    return list(range(lo, hi + 1))


def tasks(mode: str, seeds, runs=(), biases=BIASES) -> list[tuple]:
    out = []
    if mode == "rules":
        for sc, d in HEADLINE:
            for s in seeds:
                for pol, kn in rule_knobs():
                    out.append((sc, d, s, pol, kn, None))
    else:
        for run in runs:
            for b in biases:
                for sc, d in HEADLINE:
                    for s in seeds:
                        out.append((sc, d, s, "ai_harp", f"{run}_b{b:g}", (run, b)))
    return out


def run_rule(t) -> str:
    sc, d, s, pol, kn, _ = t
    subprocess.run(["wsl", "-d", "Ubuntu", "--", "python3", RUNNER, wsl(BUNDLES / f"{sc}_d{d:g}_s{s}"), pol, kn, str(s),
                    wsl(RUNS)], capture_output=True, text=True)
    return name_of(sc, d, s, pol, kn)


class Worker:
    """One policy worker: a WSL broker and a decision server, reused across its replays."""

    def __init__(self, k: int):
        self.k, self.app_port, self.srv_port = k, 5801 + k, 5901 + k
        self.cfg = BRIDGE / f"worker{k}.json"
        BRIDGE.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, SUMO_HOME=os.environ.get("SUMO_HOME", r"D:\sumo\sumo-1.19.0"))
        log = open(PROJECT_ROOT / "logs" / "revision6" / f"server_worker{k}.log", "a")
        self.broker = subprocess.Popen(["wsl", "-d", "Ubuntu", "--", "python3", BROKER, str(self.app_port),
                                        str(self.srv_port)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(3)
        self.server = subprocess.Popen([PY, "-u", "-m", "experiments.revision6.veins.decision_server", "--connect",
                                        f"127.0.0.1:{self.srv_port}", "--config", str(self.cfg)], cwd=PROJECT_ROOT,
                                       env=env, stdout=log, stderr=subprocess.STDOUT)

    def run(self, t) -> str:
        sc, d, s, pol, kn, (run, b) = t
        name = name_of(sc, d, s, pol, kn)
        bdir = BUNDLES / f"{sc}_d{d:g}_s{s}"
        nveh = len(json.loads((bdir / "bundle.json").read_text())["vehicles"])
        plain = BRIDGE / f"{name}.jsonl"
        plain.unlink(missing_ok=True)
        self.cfg.write_text(json.dumps({"name": name, "run": run, "bias": b, "tau": 0.5, "scenario": sc, "density": d,
                                        "seed": s, "n_vehicles": nveh, "log": str(plain)}))
        subprocess.run(["wsl", "-d", "Ubuntu", "--", "python3", RUNNER, wsl(bdir), "ai_harp", kn, str(s), wsl(RUNS),
                        "--bridge", f"127.0.0.1:{self.app_port}"], capture_output=True, text=True)
        if plain.exists():
            BRIDGE_LOGS.mkdir(parents=True, exist_ok=True)
            for _ in range(20):                      # the server closes the log right after the app disconnects
                try:
                    with open(plain, "rb") as fi, gzip.open(BRIDGE_LOGS / f"{name}.jsonl.gz", "wb") as fo:
                        shutil.copyfileobj(fi, fo)
                    plain.unlink()
                    break
                except PermissionError:
                    time.sleep(1)
        return name

    def close(self) -> None:
        for p in (self.server, self.broker):
            p.terminate()


def main(argv=None) -> int:
    import numpy
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=["rules", "policy", "status"])
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--runs", nargs="*", default=[f"ref_s{k}" for k in range(5)])
    ap.add_argument("--biases", type=float, nargs="*", default=list(BIASES))
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args(argv)
    print(f"[env] {sys.executable} numpy {numpy.__version__}", flush=True)
    app_sha = json.loads(GATE.read_text())["app_sha256"]
    seeds = seeds_of(a.seeds)
    for sc, d in HEADLINE:
        for s in seeds:
            if not (BUNDLES / f"{sc}_d{d:g}_s{s}" / "r6.txt").exists():
                raise SystemExit(f"{sc} d={d:g} s={s}: no r6.txt (python -m experiments.revision6.veins.r6_bundle)")
    if a.mode == "status":
        for m, kw in (("rules", {}), ("policy", {"runs": a.runs, "biases": a.biases})):
            ts = tasks(m, seeds, **kw)
            print(f"{m}: {sum(done(name_of(*t[:5]), app_sha) for t in ts)} / {len(ts)} done", flush=True)
        return 0
    ts = tasks(a.mode, seeds, a.runs, a.biases)
    todo = [t for t in ts if not done(name_of(*t[:5]), app_sha)]
    print(f"[queue6] {a.mode}: {len(ts)} replays, {len(todo)} to do, {a.jobs} workers", flush=True)
    RUNS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if a.mode == "rules":
        with ThreadPoolExecutor(a.jobs) as ex:
            for i, name in enumerate(ex.map(run_rule, todo), 1):
                print(f"{time.strftime('%H:%M:%S')} {i}/{len(todo)} {name} "
                      f"{'ok' if done(name, app_sha) else 'FAILED (kept, will retry)'}", flush=True)
    else:
        workers = [Worker(k) for k in range(a.jobs)]
        lock, it, n = threading.Lock(), iter(todo), [0]

        def loop(w):
            while True:
                with lock:
                    t = next(it, None)
                if t is None:
                    return
                name = w.run(t)
                with lock:
                    n[0] += 1
                    print(f"{time.strftime('%H:%M:%S')} {n[0]}/{len(todo)} {name} "
                          f"{'ok' if done(name, app_sha) else 'FAILED (kept, will retry)'}", flush=True)
        try:
            th = [threading.Thread(target=loop, args=(w,)) for w in workers]
            for x in th:
                x.start()
            for x in th:
                x.join()
        finally:
            for w in workers:
                w.close()
    print(f"[queue6] {a.mode} pass finished in {(time.time() - t0) / 3600:.2f} wall-hours", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
