"""H3: run one Veins replay (inside WSL; stdlib + numpy only).

    python3 run_veins.py <bundle_dir> <policy> <knob> <seed> <out_dir>

bundle_dir is an exported bundle (experiments/revision5/veins/export.py). The run happens in a fresh
directory on the Linux filesystem; afterwards the SUMO trace Veins drove (fcd_veins.xml) is compared
exactly with the bundle's reference trace, and the event log, OMNeT++ scalars and a run record
(wall time, identity verdict, exit code) are copied to out_dir. A run whose trace is not identical is
kept and flagged, never silently dropped.
"""

from __future__ import annotations

import gzip
import json
import os
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

HOME = Path.home() / "veins-r5"
APP = HOME / "aiharp"
VEINS = HOME / "veins-veins-5.3"
SUMO = HOME / "sumo-1.19.0" / "bin" / "sumo"


def config_xml(v: dict, obstacles: bool) -> str:
    p = lambda n, val, t="double": f'<parameter name="{n}" type="{t}" value="{val}"/>'
    models = [
        '<AnalogueModel type="BreakpointPathlossModel">' + "".join(
            p(k, v[k]) for k in ("alpha1", "alpha2", "L01", "L02", "breakpointDistance")) + "</AnalogueModel>",
        '<AnalogueModel type="NakagamiFading">' + p("constM", "true", "bool") + p("m", v["nakagami_m"])
        + "</AnalogueModel>",
    ]
    if obstacles:
        models.append('<AnalogueModel type="SimpleObstacleShadowing" thresholding="true"><obstacles>'
                      '<type id="building" db-per-cut="9" db-per-meter="0.4"/></obstacles></AnalogueModel>')
    return ("<root><AnalogueModels>" + "".join(models) + "</AnalogueModels>"
            f'<Decider type="Decider80211p">{p("centerFrequency", v["centerFrequency"])}</Decider></root>')


def nodes_txt(b: dict) -> str:
    out = [f"t0 {b['t0_sumo_s']!r}", f"dt {b['dt']!r}", f"n_steps {b['n_steps']}",
           f"comm_range_m {b['comm_range_m']!r}", f"max_hops {b['settings']['max_hops']}",
           f"ttl_s {b['settings']['message_ttl_s']!r}", f"origin {b['origin']['index']} {b['origin']['step']}"]
    v = b["veins"]
    out.append(f"beacon {v['beacon_hz']!r} {v['beacon_duration_s']!r} {v['cs_range_m']!r} "
               f"{int(v['dcc_enabled'])} {v['dcc_target_cbr']!r} {v['dcc_min_beacon_hz']!r}")
    out.append(f"access {v['cw_slots']} {v['slot_time_s']!r}")
    for v in b["vehicles"]:
        out.append(f"v {v['index']} {v['sumo_id']} " + " ".join(f"{a} {z}" for a, z in v["active"]))
    return "\n".join(out) + "\n"


def fcd_table(path: Path, t_from: float, t_to: float) -> dict:
    t, ids, x, y = [], [], [], []
    now = None
    for ev, el in ET.iterparse(str(path), events=("start", "end")):
        if ev == "start" and el.tag == "timestep":
            now = round(float(el.get("time")), 4)
        elif ev == "end" and el.tag == "vehicle":
            if t_from - 1e-6 <= now < t_to - 1e-6:
                t.append(now); ids.append(el.get("id")); x.append(float(el.get("x"))); y.append(float(el.get("y")))
            el.clear()
        elif ev == "end" and el.tag == "timestep":
            el.clear()
    return {"t": np.array(t), "id": np.array(ids), "x": np.array(x), "y": np.array(y)}


def fcd_compare(a: dict, b: dict) -> dict:
    if len(a["t"]) != len(b["t"]):
        return {"identical": False, "rows": [int(len(a["t"])), int(len(b["t"]))]}
    ids_eq = bool(np.array_equal(a["id"], b["id"]) and np.array_equal(a["t"], b["t"]))
    dx = float(np.max(np.abs(a["x"] - b["x"]))) if len(a["t"]) else 0.0
    dy = float(np.max(np.abs(a["y"] - b["y"]))) if len(a["t"]) else 0.0
    return {"identical": ids_eq and dx == 0.0 and dy == 0.0, "rows": int(len(a["t"])), "ids_equal": ids_eq,
            "max_abs_dx_m": dx, "max_abs_dy_m": dy}


def run(bundle: Path, policy: str, knob: float, seed: int, out: Path) -> dict:
    b = json.loads((bundle / "bundle.json").read_text())
    v = b["veins"]
    name = f"{bundle.name}__{policy}_{knob:g}__s{seed}"
    work = Path("/tmp/aiharp_runs") / name
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    for f in bundle.iterdir():
        if f.suffix in (".xml", ".sumocfg"):
            shutil.copy2(f, work / f.name)
    for f in ("omnetpp.ini", "antenna.xml"):
        shutil.copy2(APP / f, work / f)
    obstacles = b["scenario"] == "urban_nlos"
    (work / "config.xml").write_text(config_xml(v, obstacles))
    (work / "nodes.txt").write_text(nodes_txt(b))
    t0 = b["t0_sumo_s"]
    t_end = t0 + b["n_steps"] * b["dt"]
    min_end = max(t0 + 20.0, t0 + b["origin"]["step"] * b["dt"] + 10.0)   # early-stop floor (CBR window)
    cmd = [str(APP / "aiharp"), "-u", "Cmdenv", "-f", "omnetpp.ini", "-n", f"{APP}:{VEINS / 'src' / 'veins'}",
           "--result-dir=results", f"--sim-time-limit={t_end:.3f}s", f"--seed-set={seed}",
           f"--*.playgroundSizeX={v['playgroundSizeX_m']}m", f"--*.playgroundSizeY={v['playgroundSizeY_m']}m",
           f"--*.connectionManager.maxInterfDist={v['maxInterfDist_m']}m",
           f"--*.manager.t0={t0!r}", f"--*.manager.minEnd={min_end!r}",
           f"--*.manager.firstStepAt={round(t0 - 1.0, 1)}s",
           f'--*.manager.command="{SUMO}"',
           f'--*.node[*].appl.policy="{policy}"', f"--*.node[*].appl.knob={knob}",
           '--*.node[*].appl.nodesFile="nodes.txt"', '--*.node[*].appl.logFile="events.csv"']
    if obstacles:
        poly = next(work.glob("*.poly.xml"), None)
        if poly is None:
            raise SystemExit("urban_nlos bundle has no buildings (*.poly.xml)")
        cmd.append(f'--*.obstacles.obstacles=xmldoc("config.xml", '
                   '"//AnalogueModel[@type=\'SimpleObstacleShadowing\']/obstacles")')
    else:
        cmd.append('--*.obstacles.obstacles=xml("<obstacles/>")')
    env = dict(os.environ, PATH=f"{SUMO.parent}:" + ":".join(
        p for p in os.environ.get("PATH", "").split(":") if not p.startswith("/mnt/")))
    t_wall = time.time()
    proc = subprocess.run(cmd, cwd=work, env=env, capture_output=True, text=True)
    wall = time.time() - t_wall
    rec = {"min_end_s": min_end, "bundle": bundle.name, "policy": policy, "knob": knob, "seed": seed, "exit_code": proc.returncode,
           "wall_s": round(wall, 1), "cmd": cmd, "stdout_tail": proc.stdout[-3000:], "stderr_tail": proc.stderr[-3000:]}
    fcd = work / "fcd_veins.xml"
    if fcd.exists():
        ref = dict(np.load(bundle / "fcd_ref.npz"))
        got = fcd_table(fcd, b["t0_sumo_s"], t_end)
        # an early-stopped run covers a prefix of the window: compare exactly the steps it simulated
        t_last = float(got["t"].max()) if len(got["t"]) else b["t0_sumo_s"] - 1
        keep = ref["t"] <= t_last + 1e-6
        rec["fcd_check"] = fcd_compare({k: a[keep] for k, a in ref.items()}, got)
        rec["fcd_check"]["covered_until_s"] = t_last
    else:
        rec["fcd_check"] = {"identical": False, "error": "no fcd_veins.xml"}
    out.mkdir(parents=True, exist_ok=True)
    for src, dst in ((work / "events.csv", out / f"{name}.events.csv.gz"),):
        if src.exists():
            with open(src, "rb") as fi, gzip.open(dst, "wb") as fo:
                shutil.copyfileobj(fi, fo)
    sca = next((work / "results").glob("*.sca"), None)
    if sca:
        with open(sca, "rb") as fi, gzip.open(out / f"{name}.sca.gz", "wb") as fo:
            shutil.copyfileobj(fi, fo)
    (out / f"{name}.run.json").write_text(json.dumps(rec, indent=1))
    if rec["exit_code"] == 0 and rec["fcd_check"].get("identical"):
        shutil.rmtree(work, ignore_errors=True)
    return rec


if __name__ == "__main__":
    bdir, pol, kn, sd, od = sys.argv[1:6]
    r = run(Path(bdir), pol, float(kn), int(sd), Path(od))
    print(json.dumps({k: r[k] for k in ("bundle", "policy", "knob", "seed", "exit_code", "wall_s", "fcd_check")}))
