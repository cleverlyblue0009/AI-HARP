"""Revision 10, step 1 (before the addendum): pipeline, bundle and checkpoint gates on a Linux runner. NO policy or
classical replay on the replication seeds R-A / R-B (results/revision10/seeds.json).

    python -m experiments.revision10.veins.gate10 pipeline --label j1     regenerate urban NLOS d=20 seed 220 bundle
    python -m experiments.revision10.veins.gate10 bundles --seeds 230 231 ...   R-A/R-B bundles, digests, mobility sanity
    python -m experiments.revision10.veins.gate10 checkpoint              STRUCT9 hashes, gate checks 1+2, one H6 replay
    python -m experiments.revision10.veins.gate10 assemble                -> results/revision10/gates.json (local)

Outputs (runner): results/revision10/gate/<mode>_<label>.json.

Bundle identity. H6 regenerated each bundle on its runner and did not commit it (READING_NOTES §3); two regenerations
differ ONLY in the XML comment headers SUMO / randomTrips write (date, absolute work path) and in bundle.json, which embeds
those files' hashes and a provenance stamp. The content digest therefore hashes every bundle file with XML comments
removed, fcd_ref.npz by array contents, and bundle.json without "provenance" and with its "files" map recomputed on the
comment-free files. The pipeline gate requires (a) the digest to be identical on two independent runners and (b) the
regenerated bundle to match every identity H6 committed for this bundle: trace_sha256 (= the replays'
bundle_trace_sha256), t0, minEnd (run_veins8.py:88), playground size and maxInterfDist (= the committed H6 command line).
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from common.config import PROJECT_ROOT, load_yaml

R9 = PROJECT_ROOT / "results" / "revision9"
R10 = PROJECT_ROOT / "results" / "revision10"
GATE_DIR = R10 / "gate"
CELLS = (("urban_nlos", 20.0), ("rural_highway", 20.0))
RUNS = [f"{arm}_s{k}" for arm in ("plain", "rand") for k in range(5)]
REF_SHARD = R9 / "veins" / "h6" / "urban_nlos_d20_s220"
REF_REPLAY = "urban_nlos_d20_s220__struct_rand_s0_d0__s220"
COMMENT = re.compile(rb"<!--.*?-->", re.S)


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def digest(bdir: Path) -> dict:
    """Content digest of an exported bundle, invariant to SUMO's comment headers and the provenance stamp."""
    files = {}
    for f in sorted(bdir.iterdir()):
        if f.name == "bundle.json" or not f.is_file():
            continue
        if f.suffix == ".npz":
            with np.load(f) as z:
                h = hashlib.sha256()
                for k in sorted(z.files):
                    a = np.ascontiguousarray(z[k])
                    h.update(k.encode() + str(a.dtype).encode() + str(a.shape).encode() + a.tobytes())
                files[f.name] = h.hexdigest()
        elif f.suffix in (".xml", ".sumocfg", ".cfg"):
            files[f.name] = sha(COMMENT.sub(b"", f.read_bytes()))
        else:
            files[f.name] = sha(f.read_bytes())
    b = json.loads((bdir / "bundle.json").read_text())
    b.pop("provenance", None)
    b["files"] = {k: files[k] for k in sorted(b.get("files", {})) if k in files}
    files["bundle.json(content)"] = sha(json.dumps(b, sort_keys=True, default=str).encode())
    return {"files": files, "content_sha256": sha(json.dumps(files, sort_keys=True).encode()),
            "trace_sha256": b["trace_sha256"], "t0_sumo_s": b["t0_sumo_s"], "n_vehicles": len(b["vehicles"]),
            "n_steps": b["n_steps"], "maxInterfDist_m": b["veins"]["maxInterfDist_m"],
            "playground_m": [b["veins"]["playgroundSizeX_m"], b["veins"]["playgroundSizeY_m"]]}


def mobility(bdir: Path) -> dict:
    """Vehicle count and achieved density vs commanded (as mobility/sumo_runner._check_density, from the bundle)."""
    from mobility.generate import load_scenario
    from mobility.sumo_runner import DENSITY_TOLERANCE, _lane_km
    b = json.loads((bdir / "bundle.json").read_text())
    n = int(b["n_steps"])
    active = np.zeros(n)
    for v in b["vehicles"]:
        for a, e in v["active"]:
            active[a:e + 1] += 1
    lk = _lane_km(load_scenario(b["scenario"]))
    ach = float(active.mean()) / lk
    rel = abs(ach - float(b["density"])) / float(b["density"])
    return {"n_vehicles": len(b["vehicles"]), "mean_active": float(active.mean()), "lane_km": lk,
            "commanded": float(b["density"]), "achieved_veh_km_lane": round(ach, 3), "rel_error": round(rel, 4),
            "within_tolerance": bool(rel <= DENSITY_TOLERANCE), "tolerance": DENSITY_TOLERANCE,
            "origin": b["origin"]}


def h6_identity() -> dict:
    """What H6 committed about the seed-220 urban bundle (from the reference replay record)."""
    rec = json.loads((REF_SHARD / "raw" / f"{REF_REPLAY}.run.json").read_text())
    arg = {c.split("=", 1)[0]: c.split("=", 1)[1] for c in rec["cmd"] if c.startswith("--*.") and "=" in c}
    return {"trace_sha256": rec["bundle_trace_sha256"], "t0_sumo_s": float(arg["--*.manager.t0"]),
            "min_end_s": float(arg["--*.manager.minEnd"]),
            "playground_m": [int(arg["--*.playgroundSizeX"].rstrip("m")), int(arg["--*.playgroundSizeY"].rstrip("m"))],
            "maxInterfDist_m": float(arg["--*.connectionManager.maxInterfDist"].rstrip("m"))}


def write(name: str, payload: dict) -> Path:
    GATE_DIR.mkdir(parents=True, exist_ok=True)
    p = GATE_DIR / f"{name}.json"
    p.write_text(json.dumps(payload, indent=1, default=str))
    print(json.dumps(payload, default=str)[:4000], flush=True)
    return p


def runner_env() -> dict:
    import platform

    import numpy
    cpu = next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), None)
    return {"python": platform.python_version(), "numpy": numpy.__version__, "cpu": cpu, "host": platform.node()}


def cmd_pipeline(label: str) -> int:
    from experiments.revision8.veins.ci_veins import BUNDLES, bundles
    bdir = BUNDLES / "urban_nlos_d20_s220"
    shutil.rmtree(bdir, ignore_errors=True)
    bundles({(("urban_nlos", 20.0), 220)})
    d, ref = digest(bdir), h6_identity()
    b = json.loads((bdir / "bundle.json").read_text())
    t0 = b["t0_sumo_s"]
    mend = max(t0 + 20.0, t0 + b["origin"]["step"] * b["dt"] + 10.0)     # as run_veins8.py:88
    checks = {"trace_sha256": d["trace_sha256"] == ref["trace_sha256"], "t0_sumo_s": d["t0_sumo_s"] == ref["t0_sumo_s"],
              "min_end_s": mend == ref["min_end_s"],
              "playground_m": d["playground_m"] == ref["playground_m"],
              "maxInterfDist_m": d["maxInterfDist_m"] == ref["maxInterfDist_m"]}
    write(f"pipeline_{label}", {"env": runner_env(), "digest": d, "h6_identity": ref, "checks": checks,
                                "min_end_s": mend, "pass": all(checks.values())})
    return 0 if all(checks.values()) else 2


def cmd_bundles(seeds: list[int], label: str) -> int:
    from experiments.revision8.veins.ci_veins import BUNDLES, bundles
    bundles({(c, s) for c in CELLS for s in seeds})
    out = {}
    for sc, d in CELLS:
        for s in seeds:
            bdir = BUNDLES / f"{sc}_d{d:g}_s{s}"
            out[bdir.name] = {"digest": digest(bdir), "mobility": mobility(bdir)}
    write(f"bundles_{label}", {"env": runner_env(), "seeds": seeds, "bundles": out,
                               "note": "bundles only: no policy or classical replay was run on these seeds"})
    return 0


def cmd_checkpoint(label: str) -> int:
    from experiments.revision8.veins.ci_veins import (APP, BUNDLES, GATE2_REF, GATE_SHA, bundles, replay,
                                                      restore_checkpoints, sha_events)
    from experiments.revision9.veins.ci9 import bridged9
    from experiments.revision9.veins.decision_server9 import trace
    out = GATE_DIR / f"checkpoint_{label}_work"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    theta = {r: sha((R9 / "train" / r / "theta.json").read_bytes()) for r in RUNS}
    gens = {r: json.loads((R9 / "train" / r / "theta.json").read_text())["generations"] for r in RUNS}
    ck = restore_checkpoints()
    bundles({(("rural_highway", 20.0), 200), (("urban_nlos", 20.0), 220)})
    app_sha = sha((APP / "aiharp").read_bytes())
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
    # the H6 reference replay: rand_s0, delta 0, urban NLOS d=20, seed 220, revision-9 checkpoints
    b220 = BUNDLES / "urban_nlos_d20_s220"
    bridged9([{"bdir": b220, "seed": 220, "policy": "struct", "knobs": "rand_s0_d0",
               "cfg": {"policy": "struct", "run": "rand_s0", "delta": 0.0, "train_dir": "results/revision9/train"},
               "raw": out / "raw", "logs": out / "logs"}], 1, out)
    ev3 = out / "raw" / f"{REF_REPLAY}.events.csv.gz"
    s3 = sha_events(ev3) if ev3.exists() else None
    ref3 = sha_events(REF_SHARD / "raw" / f"{REF_REPLAY}.events.csv.gz")
    rec = json.loads((out / "raw" / f"{REF_REPLAY}.run.json").read_text()) if (out / "raw" / f"{REF_REPLAY}.run.json").exists() else {}
    logs = sorted((out / "logs").glob("*.jsonl"))
    tc = [trace(p) for p in logs]
    res = {"env": runner_env(), "struct9_theta_sha256": theta, "struct9_generations": gens, "aiharp_checkpoints": ck,
           "app_sha256": app_sha,
           "check1": {"events_sha256": s1, "reference_sha256": GATE_SHA, "pass": s1 == GATE_SHA},
           "check2": {"events_sha256": s2, "reference_sha256": ref2, "pass": s2 == ref2},
           "h6_replay": {"replay": REF_REPLAY, "events_sha256": s3, "committed_events_sha256": ref3, "pass": s3 == ref3,
                         "exit_code": rec.get("exit_code"), "fcd_check": rec.get("fcd_check"),
                         "bundle_trace_sha256": rec.get("bundle_trace_sha256"),
                         "trace_check": tc, "trace_pass": bool(tc) and all(x["pass"] for x in tc)},
           "veins_source": veins_source(b220)}
    res["pass"] = (res["check1"]["pass"] and res["check2"]["pass"] and res["h6_replay"]["pass"]
                   and res["h6_replay"]["trace_pass"] and all(g == 80 for g in gens.values()))
    write(f"checkpoint_{label}", res)
    shutil.rmtree(out, ignore_errors=True)
    return 0 if res["pass"] else 2


def veins_source(bdir: Path) -> dict:
    """Report only: the analogue models in the built Veins 5.3 tree and the ones the project enables."""
    from experiments.revision8.veins.ci_veins import VR5
    sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "revision5" / "veins"))
    import run_veins as r5
    am = VR5 / "veins-veins-5.3" / "src" / "veins" / "modules" / "analogueModel"
    listing = sorted(p.name for p in am.iterdir()) if am.exists() else None
    hits = subprocess.run(["grep", "-rniE", "lognormal|log_normal|log-normal", str(VR5 / "veins-veins-5.3" / "src")],
                          capture_output=True, text=True).stdout.splitlines()
    b = json.loads((bdir / "bundle.json").read_text())
    ini = PROJECT_ROOT / "experiments" / "revision8" / "veins" / "aiharp" / "omnetpp.ini"
    ini_lines = [f"experiments/revision8/veins/aiharp/omnetpp.ini:{i}: {l.strip()}"
                 for i, l in enumerate(ini.read_text().splitlines(), 1)
                 if re.search(r"analogue|config\.xml|decider|obstacle", l, re.I)]
    return {"analogueModel_dir": str(am), "analogueModel_listing": listing, "lognormal_grep_src": hits,
            "omnetpp_ini_lines": ini_lines,
            "config_xml_urban_nlos_d20_s220": r5.config_xml(b["veins"], b.get("buildings") is not None),
            "config_xml_source": "experiments/revision5/veins/run_veins.py:47-58 (config_xml), written per replay"}


def cmd_assemble() -> int:
    """Local: results/revision10/gate/*.json -> results/revision10/gates.json."""
    from experiments.revision1.provenance import stamp, write_json
    seeds = json.loads((R10 / "seeds.json").read_text())
    pipe = {p.stem: json.loads(p.read_text()) for p in sorted(GATE_DIR.glob("pipeline_*.json"))}
    ck = {p.stem: json.loads(p.read_text()) for p in sorted(GATE_DIR.glob("checkpoint_*.json"))}
    bun = {}
    for p in sorted(GATE_DIR.glob("bundles_*.json")):
        bun.update(json.loads(p.read_text())["bundles"])
    digests = {k: v["digest"]["content_sha256"] for k, v in pipe.items()}
    pipeline = {"runs": {k: {"env": v["env"], "content_sha256": v["digest"]["content_sha256"], "checks": v["checks"],
                             "pass": v["pass"]} for k, v in pipe.items()},
                "h6_identity": next(iter(pipe.values()))["h6_identity"] if pipe else None,
                "independent_runs": len(pipe), "digests_identical": len(set(digests.values())) == 1}
    pipeline["pass"] = bool(len(pipe) >= 2 and pipeline["digests_identical"] and all(v["pass"] for v in pipe.values()))
    want = [f"{sc}_d{d:g}_s{s}" for sc, d in CELLS for s in seeds["R_A"] + seeds["R_B"]]
    bundles = {"expected": len(want), "present": sum(w in bun for w in want), "missing": [w for w in want if w not in bun],
               "per_bundle": {w: {"content_sha256": bun[w]["digest"]["content_sha256"],
                                  "trace_sha256": bun[w]["digest"]["trace_sha256"], **bun[w]["mobility"]}
                              for w in want if w in bun}}
    bundles["mobility_all_within_tolerance"] = all(v["within_tolerance"] for v in bundles["per_bundle"].values())
    cp = next(iter(ck.values())) if ck else {}
    payload = {"provenance": stamp("revision10/gates", {}), "seeds": {"R_A": seeds["R_A"], "R_B": seeds["R_B"]},
               "pipeline_gate": pipeline, "bundles": bundles,
               "checkpoint_gate": {k: v for k, v in cp.items() if k != "veins_source"},
               "veins_source_check_report_only": cp.get("veins_source"),
               "all_gates_pass": bool(pipeline["pass"] and cp.get("pass") and not bundles["missing"])}
    write_json(R10 / "gates.json", payload)
    print(json.dumps({"pipeline": pipeline["pass"], "checkpoint": cp.get("pass"), "bundles": bundles["present"],
                      "mobility_ok": bundles["mobility_all_within_tolerance"]}))
    return 0 if payload["all_gates_pass"] else 2


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=["pipeline", "bundles", "checkpoint", "assemble"])
    ap.add_argument("--seeds", type=int, nargs="*", default=[])
    ap.add_argument("--label", default="j1")
    a = ap.parse_args(argv)
    if a.mode == "pipeline":
        return cmd_pipeline(a.label)
    if a.mode == "bundles":
        return cmd_bundles(a.seeds, a.label)
    if a.mode == "checkpoint":
        return cmd_checkpoint(a.label)
    return cmd_assemble()


if __name__ == "__main__":
    raise SystemExit(main())
