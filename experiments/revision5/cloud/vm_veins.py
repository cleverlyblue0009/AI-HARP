"""Revision-5 offload: the H3 Veins replays of one VM's slice (runs on the VM, Linux).

    python -m experiments.revision5.cloud.vm_veins --slice 0 --n 3 --seeds 200-229 --jobs 3 [--skip done.txt]

Work unit = a bundle (headline cell x seed); a VM owns the bundles with sha256(bundle name) mod n == slice, so
the slices are disjoint and each VM exports only its own bundles. Per bundle: export (experiments.revision5.
veins.export, SUMO on this VM; the bundle records the canonical trace's sha256, which the laptop checks
against its own cached trace before accepting anything), then every classical setting (55) is replayed.
Skip-if-done: a replay is skipped when its run.json exists here with exit 0 and an identical trace, or its
name is listed in --skip (replays already done on the laptop). Network-layer seeds (200-209) first.
Outputs: ~/AI-HARP/cache/veins_runs/h3/ and ~/AI-HARP/cache/veins/<bundle>/ (pulled by the laptop).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from experiments.revision5.sweep import HEADLINE
from experiments.revision5.veins.queue import BUNDLES, RUNS, settings

RUNNER = Path.home() / "veins-r5" / "run_veins.py"


def seeds_arg(s: str) -> list[int]:
    a, _, b = s.partition("-")
    return list(range(int(a), int(b) + 1)) if b else [int(x) for x in s.split(",")]


def mine(name: str, i: int, n: int) -> bool:
    return int(hashlib.sha256(name.encode()).hexdigest()[:8], 16) % n == i


def done(name: str) -> bool:
    p = RUNS / f"{name}.run.json"
    if not p.exists():
        return False
    r = json.loads(p.read_text())
    return r.get("exit_code") == 0 and bool(r.get("fcd_check", {}).get("identical"))


def replay(args: tuple) -> str:
    bdir, pol, knob, seed = args
    env = dict(os.environ, AIHARP_REPO=str(Path.home() / "AI-HARP"))
    subprocess.run([sys.executable, str(RUNNER), str(bdir), pol, repr(knob), str(seed), str(RUNS)],
                   capture_output=True, text=True, env=env)
    name = f"{bdir.name}__{pol}_{knob:g}__s{seed}"
    return f"{name} {'ok' if done(name) else 'FAILED/non-identical (kept; retried next launch)'}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--slice", type=int, required=True)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--seeds", default="200-229")
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--skip", type=Path, default=None)
    a = ap.parse_args(argv)
    skip = set(a.skip.read_text().split()) if a.skip and a.skip.exists() else set()
    RUNS.mkdir(parents=True, exist_ok=True)
    seeds = sorted(seeds_arg(a.seeds), key=lambda s: (s > 209, s))          # network layer first
    bundles = [(sc, d, s) for s in seeds for sc, d in HEADLINE]
    my = [(sc, d, s) for sc, d, s in bundles if mine(f"{sc}_d{d:g}_s{s}", a.slice, a.n)]
    print(f"[vm_veins] slice {a.slice}/{a.n}: {len(my)} of {len(bundles)} bundles", flush=True)
    from experiments.revision5.veins.export import export
    for sc, d, s in my:
        bdir = BUNDLES / f"{sc}_d{d:g}_s{s}"
        if not (bdir / "bundle.json").exists():
            print(f"[vm_veins] export {bdir.name}", flush=True)
            export(sc, d, s)
        todo = [(bdir, pol, knob, s) for _, pol, knob in settings()
                if f"{bdir.name}__{pol}_{knob:g}__s{s}" not in skip and not done(f"{bdir.name}__{pol}_{knob:g}__s{s}")]
        print(f"[vm_veins] {bdir.name}: {len(todo)} replays to do", flush=True)
        with ThreadPoolExecutor(a.jobs) as ex:
            for msg in ex.map(replay, todo):
                print(msg, flush=True)
    print("[vm_veins] slice complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
