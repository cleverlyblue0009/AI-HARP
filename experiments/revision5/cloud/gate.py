"""Revision-5 offload gate (runs on the VM). Writes ~/GATE.json; rows from a VM are rejected unless it passes.

    python -m experiments.revision5.cloud.gate --ref ~/gate_ref

1. Rule 7: ref_s0, rural_highway d=20, bias -2/+1, tau 0.5, eval seeds 0-9 -> all 20 rows match
   results/revision1/agent_runs/ref_s0.csv to <= 1e-12 (experiments.revision5.bitexact).
2. Veins: the replays in --ref (event logs produced in WSL on the laptop) are re-run here and their event
   logs must be byte-identical; the bundle is exported here, so its canonical-trace sha256 is reported too
   (the laptop compares it with its own cached trace).
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

from common.config import PROJECT_ROOT

GATE_REPLAYS = [("rural_highway", 2.0, 201, "greedy_farthest", 4.0), ("rural_highway", 20.0, 200, "slotted_1p", 8.0)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ref", type=Path, required=True)
    a = ap.parse_args(argv)
    out = {"host": platform.node(), "os": platform.platform(), "python": sys.version.split()[0],
           "repo_commit": subprocess.run(["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"], capture_output=True,
                                         text=True).stdout.strip()}
    import numpy, torch
    out["numpy"], out["torch"] = numpy.__version__, torch.__version__

    bx = subprocess.run([sys.executable, "-m", "experiments.revision5.bitexact"], cwd=PROJECT_ROOT,
                        capture_output=True, text=True)
    out["bitexact"] = {"exit_code": bx.returncode, "tail": (bx.stdout + bx.stderr)[-1500:],
                       "pass": bx.returncode == 0 and "PASS" in bx.stdout}

    from experiments.revision5.veins.export import export
    from experiments.revision5.veins.queue import RUNS
    RUNS.mkdir(parents=True, exist_ok=True)
    veins = []
    for sc, d, s, pol, knob in GATE_REPLAYS:
        bdir = export(sc, d, s)
        b = json.loads((bdir / "bundle.json").read_text())
        name = f"{bdir.name}__{pol}_{knob:g}__s{s}"
        gate_dir = Path.home() / "gate_runs"
        env = dict(os.environ, AIHARP_REPO=str(PROJECT_ROOT))
        subprocess.run([sys.executable, str(Path.home() / "veins-r5" / "run_veins.py"), str(bdir), pol, repr(knob),
                        str(s), str(gate_dir)], capture_output=True, text=True, env=env)
        rec = json.loads((gate_dir / f"{name}.run.json").read_text())
        mine = gzip.open(gate_dir / f"{name}.events.csv.gz", "rb").read()
        ref = gzip.open(a.ref / f"{name}.events.csv.gz", "rb").read()
        veins.append({"replay": name, "exit_code": rec["exit_code"], "fcd_identical": rec["fcd_check"].get("identical"),
                      "events_identical": mine == ref, "events_bytes": [len(mine), len(ref)],
                      "trace_sha256": b.get("trace_sha256"), "origin": b.get("origin"), "wall_s": rec["wall_s"]})
    out["veins"] = veins
    out["pass"] = out["bitexact"]["pass"] and all(v["exit_code"] == 0 and v["fcd_identical"] and v["events_identical"]
                                                   for v in veins)
    (Path.home() / "GATE.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("host", "pass")} | {"bitexact": out["bitexact"]["pass"],
                      "veins": [(v["replay"], v["events_identical"], v["fcd_identical"]) for v in veins]}))
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
