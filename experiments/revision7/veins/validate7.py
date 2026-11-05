"""Revision 7: validate the rebuilt binary before any reported run (results/revision7/ADDENDUM.md, Validation).

    python -m experiments.revision7.veins.validate7 gate     checks 1-3, 5; writes results/revision7/veins/app_gate.json on success
    python -m experiments.revision7.veins.validate7 unit     check 4

Replay at rural d=20, seed 200. With both new parameters off the new binary must reproduce the stored event logs byte for
byte; a capped replay's logged requests must then be reproduced exactly offline. Resumable: finished gate replays are kept in
cache/veins_runs/r7_gate. The revision-6 validation files are not rewritten.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from common.config import PROJECT_ROOT
from experiments.revision1.provenance import stamp
from experiments.revision6.veins import queue6 as q6
from experiments.revision6.veins import validate_policy as vp
from experiments.revision7.veins import queue7 as q7

OUT = PROJECT_ROOT / "results" / "revision7" / "veins"
GATE_RUNS = PROJECT_ROOT / "cache" / "veins_runs" / "r7_gate"
GATE_RUNS_CAP = PROJECT_ROOT / "cache" / "veins_runs" / "r7_gate_capped"
GATE_LOGS = Path(r"D:\aiharp-r7-bridge-logs\gate")
SC, D, S = "rural_highway", 20.0, 200
GATE_SLOTTED_SHA = "707d65331f9ecee307f321b38f6c6ded9989bec47320a4a844616875ccf2fba8"


def sha_events(path: Path) -> str:
    with gzip.open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def wsl_sha() -> str:
    r = subprocess.run(["wsl", "-d", "Ubuntu", "--", "sha256sum", "/home/upasana/veins-r7/aiharp/aiharp"], capture_output=True, text=True)
    return r.stdout.split()[0]


def classical(pol: str, kn: str) -> dict:
    name = q6.name_of(SC, D, S, pol, kn)
    ev = GATE_RUNS / f"{name}.events.csv.gz"
    if not ev.exists():
        subprocess.run(q7.runner_cmd(q6.BUNDLES / f"{SC}_d{D:g}_s{S}", pol, kn, S, GATE_RUNS), capture_output=True, text=True)
    rec = json.loads((GATE_RUNS / f"{name}.run.json").read_text())
    return {"replay": name, "exit_code": rec["exit_code"], "trace_identical": rec["fcd_check"].get("identical"),
            "events_sha256": sha_events(ev)}


def policy(worker, out: Path, *flags: str) -> tuple[str, dict]:
    t = (SC, D, S, "ai_harp", "ref_s0_b1", ("ref_s0", 1.0))
    name = q6.name_of(*t[:5])
    if not (out / f"{name}.events.csv.gz").exists():
        worker.run7(t, out, *flags)
    rec = json.loads((out / f"{name}.run.json").read_text())
    return name, {"replay": name, "exit_code": rec["exit_code"], "trace_identical": rec["fcd_check"].get("identical"),
                  "events_sha256": sha_events(out / f"{name}.events.csv.gz")}


def gate() -> dict:
    GATE_RUNS.mkdir(parents=True, exist_ok=True)
    GATE_RUNS_CAP.mkdir(parents=True, exist_ok=True)
    checks: dict[str, dict] = {}
    c = classical("slotted_1p", "8")
    checks["slotted_1p_vs_revision6_gate"] = {**c, "reference_sha256": GATE_SLOTTED_SHA, "pass": c["events_sha256"] == GATE_SLOTTED_SHA}
    h3 = PROJECT_ROOT / "cache" / "veins_runs" / "h3"
    kn = {}
    for pol, k in q7.d9_settings():
        kn.setdefault(pol, k)
    for pol, k in kn.items():
        c = classical(pol, k)
        ref = sha_events(h3 / f"{c['replay']}.events.csv.gz")
        checks[f"{pol}_vs_revision5_replay"] = {**c, "reference_sha256": ref, "pass": c["events_sha256"] == ref}
    q7.Worker7.logs = GATE_LOGS
    w = q7.Worker7(0)
    try:
        name, c = policy(w, GATE_RUNS)
        ref = sha_events(q6.RUNS / f"{name}.events.csv.gz")
        checks["policy_off_vs_revision6_replay"] = {**c, "reference_sha256": ref, "pass": c["events_sha256"] == ref}
        q7.Worker7.logs = GATE_LOGS / "capped"
        name, c = policy(w, GATE_RUNS_CAP, "--max-decisions-per-epoch", "1")
    finally:
        w.close()
    with gzip.open(GATE_RUNS_CAP / f"{name}.events.csv.gz", "rt") as f:
        lines = f.read().splitlines()
    c["capped_callbacks"] = sum(l.startswith("C,") for l in lines)
    c["decisions"] = sum(l.startswith("D,") for l in lines)
    checks["capped_replay_runs"] = {**c, "pass": c["exit_code"] == 0 and bool(c["trace_identical"]) and c["capped_callbacks"] > 0}
    log = q6.BRIDGE / f"gate_{name}.jsonl"
    with gzip.open(GATE_LOGS / "capped" / f"{name}.jsonl.gz", "rb") as fi, open(log, "wb") as fo:
        shutil.copyfileobj(fi, fo)
    tr = vp.trace(log)
    log.unlink()
    (OUT / "port_validation_policy_trace.json").write_text(
        json.dumps({"provenance": stamp("revision7/veins/port_validation_policy_trace", vp.CELL, seeds=[S]), **tr}, indent=1, default=float))
    checks["capped_trace_check"] = {"requests": tr["requests"], "mismatches": tr["mismatches"], "pass": tr["pass"]}
    return checks


def main(argv=None) -> int:
    import numpy
    print(f"[env] {sys.executable} numpy {numpy.__version__}", flush=True)
    argv = argv if argv is not None else sys.argv[1:]
    OUT.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("SUMO_HOME", r"D:\sumo\sumo-1.19.0")
    if argv[0] == "unit":
        r = vp.unit()
        (OUT / "port_validation_policy_unit.json").write_text(
            json.dumps({"provenance": stamp("revision7/veins/port_validation_policy_unit", vp.CELL, seeds=[vp.CELL["seed"]]), **r},
                       indent=1, default=float))
        print(json.dumps({k: r[k] for k in ("contexts", "max_abs_diff_action_distribution", "gate_verdict_mismatches", "pass")}))
        return 0 if r["pass"] else 1
    checks = gate()
    ok = all(c["pass"] for c in checks.values())
    print(json.dumps(checks, indent=1))
    payload = {"provenance": stamp("revision7/veins/app_gate", {"replay": f"{SC} d={D:g} seed {S}"}), "pass": ok, "app_sha256": wsl_sha(),
               "revision6_app_sha256": json.loads(q6.GATE.read_text())["app_sha256"], "checks": checks}
    (OUT / ("app_gate.json" if ok else "app_gate_FAILED.json")).write_text(json.dumps(payload, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
