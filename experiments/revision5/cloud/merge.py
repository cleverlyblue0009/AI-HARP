"""Revision-5 offload: merge what pull.sh fetched from the VMs, with provenance checks (laptop side).

    python -m experiments.revision5.cloud.merge            (SUMO_HOME is set internally for the trace checks)

Nothing from a VM is used unless its ~/GATE.json says it passed (rule-7 bit-exactness and byte-identical Veins
replays). Then:

Veins replays (cache/cloud/<vm>/veins_runs -> cache/veins_runs/h3):
  * the bundle's canonical-trace sha256 must equal the laptop's own cached trace's (Windows SUMO) and its
    originator (index, step) must equal the laptop's own computation -- checked once per bundle and recorded in
    cache/cloud/verified_bundles.json;
  * the bundle must belong to that VM's slice; the replay must have exit code 0 and an identical trace, and its
    record's bundle hash must match; replays already present on the laptop are never overwritten.
Classical rows (cache/cloud/<vm>/runs/*__sliceIofN.csv -> results/revision5/runs/<name>__cloud.csv):
  * task_hash must recompute from the row; backend fallback; metrics_version and the git commit must match the
    laptop's; rows already in the laptop file or the cloud file are skipped. The laptop never appends to a file
    its own sweeps write.
Every decision is logged to cache/cloud/merge_log.jsonl.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import time
from pathlib import Path

from common.config import PROJECT_ROOT

CLOUD = PROJECT_ROOT / "cache" / "cloud"
H3 = PROJECT_ROOT / "cache" / "veins_runs" / "h3"
RUNS = PROJECT_ROOT / "results" / "revision5" / "runs"
VMS = {"aiharp-ci": 0, "aiharp-ea": 1, "aiharp-ks": 2}
N_SLICES = 3
VERIFIED = CLOUD / "verified_bundles.json"
LOG = CLOUD / "merge_log.jsonl"


def log(**kw) -> None:
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"t": time.strftime("%Y-%m-%dT%H:%M:%S"), **kw}) + "\n")


def gate_ok(vm: str) -> tuple[bool, dict]:
    p = CLOUD / vm / "GATE.json"
    if not p.exists():
        return False, {}
    g = json.loads(p.read_text())
    return bool(g.get("pass")), g


def verify_bundle(name: str, bjson: dict, cache: dict) -> bool:
    if name in cache:
        return cache[name]["ok"] and cache[name]["trace_sha256"] == bjson.get("trace_sha256")
    from experiments.revision5.veins.export import trace_sha256
    from experiments.revision5.veins.veins_metrics import context
    sc, rest = name.split("_d")
    d, s = rest.split("_s")
    res = context(sc, float(d), int(s))                       # laptop's canonical trace (cached) + flooding run
    mine = trace_sha256(res.trace)
    ok = (mine == bjson.get("trace_sha256") and bjson["origin"]["index"] == res.origin_index
          and bjson["origin"]["step"] == res.origin_step)
    cache[name] = {"ok": ok, "trace_sha256": bjson.get("trace_sha256"), "laptop_sha256": mine,
                   "origin_vm": bjson["origin"], "origin_laptop": [int(res.origin_index), int(res.origin_step)]}
    log(kind="bundle", bundle=name, ok=ok, **{k: v for k, v in cache[name].items() if k != "ok"})
    return ok


def merge_veins(vm: str, gate: dict, verified: dict) -> tuple[int, int]:
    from experiments.revision5.cloud.vm_veins import mine
    src, bsrc = CLOUD / vm / "veins_runs", CLOUD / vm / "bundles"
    added = rejected = 0
    for rj in sorted(src.glob("*.run.json")):
        name = rj.name[: -len(".run.json")]
        if (H3 / rj.name).exists():
            continue
        rec = json.loads(rj.read_text())
        bundle = rec["bundle"]
        bj = bsrc / bundle / "bundle.json"
        why = None
        if not mine(bundle, VMS[vm], N_SLICES):
            why = "bundle not in this VM's slice"
        elif rec.get("exit_code") != 0 or not rec.get("fcd_check", {}).get("identical"):
            why = "failed or non-identical replay (left for retry)"
        elif not bj.exists():
            why = "bundle.json not pulled yet"
        else:
            b = json.loads(bj.read_text())
            if not verify_bundle(bundle, b, verified):
                why = "bundle trace/origin differs from the laptop's"
            elif rec.get("bundle_trace_sha256") != b.get("trace_sha256"):
                why = "replay record's bundle hash differs"
        if why:
            rejected += 1
            log(kind="veins", vm=vm, run=name, accepted=False, why=why)
            continue
        H3.mkdir(parents=True, exist_ok=True)
        for suffix in (".events.csv.gz", ".sca.gz", ".run.json"):
            f = src / (name + suffix)
            if f.exists():
                shutil.copy2(f, H3 / f.name)
        added += 1
        log(kind="veins", vm=vm, run=name, accepted=True, host=rec.get("host"), commit=rec.get("repo_commit"),
            app_sha256=rec.get("app_sha256"))
    return added, rejected


def merge_rows(vm: str, gate: dict) -> tuple[int, int]:
    from analysis.metrics import METRICS_VERSION
    from experiments.revision1.runner import load_rows, task
    added = rejected = 0
    for f in sorted((CLOUD / vm / "runs").glob("*__slice*of*.csv")):
        base = f.name.split("__slice")[0]
        target = RUNS / f"{base}__cloud.csv"
        have = {r["task_hash"] for r in load_rows(RUNS / f"{base}.csv")} | {r["task_hash"] for r in load_rows(target)}
        new = []
        for r in load_rows(f):
            if r["task_hash"] in have:
                continue
            val = json.loads(r["value"]) if r.get("value") not in (None, "") and r["param"] else None
            t = task(r["scenario"], float(r["density"]), r["policy"], int(r["seed"]), param=r["param"] or None,
                     value=val, backend="fallback")
            why = None
            if t["task_hash"] != r["task_hash"]:
                why = "task hash does not recompute"
            elif r.get("backend") != "fallback":
                why = f"backend {r.get('backend')}"
            elif r.get("metrics_version") != str(METRICS_VERSION):
                why = "metrics_version differs"
            elif r.get("git_commit", "").replace("-dirty", "") != gate.get("repo_commit"):
                why = "git commit differs from the gated commit"
            if why:
                rejected += 1
                log(kind="row", vm=vm, file=f.name, task_hash=r["task_hash"], accepted=False, why=why)
                continue
            new.append(r)
            have.add(r["task_hash"])
        if new:
            exists = target.exists()
            with target.open("a", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=list(new[0].keys()))
                if not exists:
                    w.writeheader()
                w.writerows(new)
            added += len(new)
            log(kind="rows", vm=vm, file=f.name, target=target.name, added=len(new))
    return added, rejected


def main() -> int:
    os.environ.setdefault("SUMO_HOME", "D:/sumo/sumo-1.19.0")
    CLOUD.mkdir(parents=True, exist_ok=True)
    verified = json.loads(VERIFIED.read_text()) if VERIFIED.exists() else {}
    for vm in VMS:
        ok, gate = gate_ok(vm)
        if not ok:
            print(f"{vm}: gate {'FAILED' if gate else 'not run yet'} -- nothing merged")
            continue
        va, vr = merge_veins(vm, gate, verified)
        ra, rr = merge_rows(vm, gate)
        print(f"{vm}: veins +{va} (rejected {vr}); rows +{ra} (rejected {rr})")
    VERIFIED.write_text(json.dumps(verified, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
