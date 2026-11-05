"""Revision 6: one Veins replay with the revision-6 app (inside WSL; stdlib + numpy only).

    python3 run_veins6.py <bundle_dir> <policy> <knobs> <seed> <out_dir> [--bridge host:port] [--app DIR]

Same as experiments/revision5/veins/run_veins.py (whose functions are reused unchanged: config.xml, nodes.txt, the
exact trace-identity check) with the revision-6 app binary, the bundle's r6.txt lines (risk field, hazard,
carriageway signs; experiments/revision6/veins/r6_bundle.py) appended to nodes.txt, and the policy's knobs:
  classical  <knobs> = the revision-5 knob (one number)
  rar        <knobs> = "a,b,c"
  scf_slotted        = "n_slots,timeout_s"
  ai_harp            = a run tag (the decision server holds checkpoint and bias); --bridge is required
--app selects the binary directory (default ~/veins-r6/aiharp); the binary-identity gate also runs the revision-5
binary through this script (--app ~/veins-r5/aiharp).
"""

from __future__ import annotations

import gzip
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "revision5" / "veins"))
import run_veins as r5  # noqa: E402  (revision-5 runner, reused unchanged)

HOME = Path.home()
APP6 = HOME / "veins-r6" / "aiharp"


def claim_replay(name: str, app: Path, out: Path) -> Path | None:
    """Claim a replay for this process; None if it has a good record already or a live process holds it."""
    p = out / f"{name}.run.json"
    if p.exists():
        r = json.loads(p.read_text())
        if r.get("exit_code") == 0 and r.get("fcd_check", {}).get("identical") and r.get("app_sha256") == r5._sha(app / "aiharp"):
            return None
    d = Path("/tmp/aiharp_runs6") / "claims"
    d.mkdir(parents=True, exist_ok=True)
    c = d / name
    for _ in range(2):
        try:
            fd = os.open(c, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                os.kill(int(c.read_text() or 0), 0)
                return None                                # held by a live process
            except (ProcessLookupError, ValueError, FileNotFoundError):
                c.unlink(missing_ok=True)                  # left by a killed process
                continue
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        return c
    return None


def run(bundle: Path, policy: str, knobs: str, seed: int, out: Path, bridge: str = "", app: Path = APP6, app_args=()) -> dict:
    b = json.loads((bundle / "bundle.json").read_text())
    v = b["veins"]
    tag = knobs.replace(",", "_").replace(" ", "")
    name = f"{bundle.name}__{policy}_{tag}__s{seed}"
    work = Path("/tmp/aiharp_runs6") / app.parent.name / name     # per binary: concurrent gate runs must not share it
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    for f in bundle.iterdir():
        if f.suffix in (".xml", ".sumocfg"):
            shutil.copy2(f, work / f.name)
    for f in ("omnetpp.ini", "antenna.xml"):
        shutil.copy2(app / f, work / f)
    obstacles = b["scenario"] == "urban_nlos"
    (work / "config.xml").write_text(r5.config_xml(v, obstacles))
    extra = (bundle / "r6.txt").read_text() if (bundle / "r6.txt").exists() else ""
    if policy in ("rar", "scf_slotted", "ai_harp") and not extra:
        raise SystemExit(f"{bundle}: no r6.txt (run experiments.revision6.veins.r6_bundle first)")
    (work / "nodes.txt").write_text(r5.nodes_txt(b) + extra)
    t0 = b["t0_sumo_s"]
    t_end = t0 + b["n_steps"] * b["dt"]
    min_end = max(t0 + 20.0, t0 + b["origin"]["step"] * b["dt"] + 10.0)
    cmd = [str(app / "aiharp"), "-u", "Cmdenv", "-f", "omnetpp.ini", "-n", f"{app}:{r5.VEINS / 'src' / 'veins'}",
           "--result-dir=results", f"--sim-time-limit={t_end:.3f}s", f"--seed-set={seed}",
           f"--*.playgroundSizeX={v['playgroundSizeX_m']}m", f"--*.playgroundSizeY={v['playgroundSizeY_m']}m",
           f"--*.connectionManager.maxInterfDist={v['maxInterfDist_m']}m",
           f"--*.manager.t0={t0!r}", f"--*.manager.minEnd={min_end!r}",
           f"--*.manager.firstStepAt={round(t0 - 1.0, 1)}s",
           f'--*.manager.command="{r5.SUMO}"',
           f'--*.node[*].appl.policy="{policy}"',
           '--*.node[*].appl.nodesFile="nodes.txt"', '--*.node[*].appl.logFile="events.csv"']
    if policy in ("rar", "scf_slotted"):
        cmd.append(f'--*.node[*].appl.knobs="{knobs}"')
    elif policy == "ai_harp":
        if not bridge:
            raise SystemExit("ai_harp needs --bridge host:port")
        cmd += [f'--*.node[*].appl.bridge="{bridge}"', f"--*.node[*].appl.replaySeed={seed}"]
    else:
        cmd.append(f"--*.node[*].appl.knob={float(knobs)}")
    if obstacles:
        if not b.get("buildings") or not (work / b["buildings"]).exists():
            raise SystemExit("urban_nlos bundle has no buildings (*.poly.xml)")
        cmd.append('--*.obstacles.obstacles=xmldoc("config.xml", '
                   '"//AnalogueModel[@type=\'SimpleObstacleShadowing\']/obstacles")')
    else:
        cmd.append('--*.obstacles.obstacles=xml("<obstacles/>")')
    cmd += list(app_args)                                  # revision 7: the new build's parameters
    env = dict(os.environ, PATH=f"{r5.SUMO.parent}:" + ":".join(
        p for p in os.environ.get("PATH", "").split(":") if not p.startswith("/mnt/")))
    t_wall = time.time()
    proc = subprocess.run(cmd, cwd=work, env=env, capture_output=True, text=True)
    wall = time.time() - t_wall
    rec = {"host": os.uname().nodename, "app_sha256": r5._sha(app / "aiharp"), "repo_commit": r5._repo_commit(),
           "bundle_trace_sha256": b.get("trace_sha256"), "min_end_s": min_end, "bundle": bundle.name,
           "policy": policy, "knobs": knobs, "seed": seed, "bridge": bridge, "exit_code": proc.returncode,
           "wall_s": round(wall, 1), "cmd": cmd, "stdout_tail": proc.stdout[-3000:], "stderr_tail": proc.stderr[-3000:]}
    fcd = work / "fcd_veins.xml"
    if fcd.exists():
        import numpy as np
        ref = dict(np.load(bundle / "fcd_ref.npz"))
        got = r5.fcd_table(fcd, b["t0_sumo_s"], t_end)
        t_last = float(got["t"].max()) if len(got["t"]) else b["t0_sumo_s"] - 1
        keep = ref["t"] <= t_last + 1e-6
        rec["fcd_check"] = r5.fcd_compare({k: a[keep] for k, a in ref.items()}, got)
        rec["fcd_check"]["covered_until_s"] = t_last
    else:
        rec["fcd_check"] = {"identical": False, "error": "no fcd_veins.xml"}
    out.mkdir(parents=True, exist_ok=True)
    if (work / "events.csv").exists():
        with open(work / "events.csv", "rb") as fi, gzip.open(out / f"{name}.events.csv.gz", "wb") as fo:
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
    args = sys.argv[1:]
    bridge, app = "", APP6
    if "--bridge" in args:
        i = args.index("--bridge"); bridge = args[i + 1]; del args[i:i + 2]
    if "--app" in args:
        i = args.index("--app"); app = Path(args[i + 1]).expanduser(); del args[i:i + 2]
    app_args = []
    if "--max-decisions-per-epoch" in args:          # revision 7 (flags, not raw arguments: wsl.exe passes through a shell)
        i = args.index("--max-decisions-per-epoch")
        app_args.append(f"--*.node[*].appl.maxDecisionsPerEpoch={int(args[i + 1])}"); del args[i:i + 2]
    if "--beacon-cam-velocity" in args:
        args.remove("--beacon-cam-velocity")
        app_args.append("--*.node[*].appl.beaconCamVelocity=true")
    bdir, pol, kn, sd, od = args[:5]
    name = f"{Path(bdir).name}__{pol}_{kn.replace(',', '_').replace(' ', '')}__s{sd}"
    claim = claim_replay(name, app, Path(od))        # two queues may share the task list (D10: freed cores)
    if claim is None:
        print(json.dumps({"skipped": name}))
        raise SystemExit(0)
    try:
        r = run(Path(bdir), pol, kn, int(sd), Path(od), bridge, app, app_args)
    finally:
        claim.unlink(missing_ok=True)
    print(json.dumps({k: r[k] for k in ("bundle", "policy", "knobs", "seed", "exit_code", "wall_s", "fcd_check")}))
