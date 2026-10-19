"""H3 Veins queue: export bundles and run replays, resumably (Windows side; launches WSL jobs).

    python -m experiments.revision5.veins.queue export --seeds 200 201 ...
    python -m experiments.revision5.veins.queue run    --seeds 200 201 ... --jobs 2

Cells: the 4 headline cells (protocol.json). Settings: the classical POLICY_SWEEPS settings of the
families -- all 7, 55 settings (greedy_farthest and dvcast use beacon neighbour tables in Veins;
see AiharpApp.cc). A run is done when its run.json exists with exit code 0 and an identical
trace; anything else (failure, non-identical trace) is kept on disk and retried, never dropped.
Outputs: cache/veins_runs/h3/ (logs, scalars, run records), scored later by veins_metrics.py.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from analysis.pareto import POLICY_SWEEPS
from common.config import PROJECT_ROOT
from experiments.revision5.sweep import HEADLINE

VEINS_POLICY = {"flooding": "flooding", "p_persistence_03": "p_persistence", "slotted_1p": "slotted_1p",
                "weighted_p": "weighted_p", "counter_based": "counter_based",
                "greedy_farthest": "greedy_farthest", "dvcast": "dvcast"}
NOT_IMPLEMENTED: tuple[str, ...] = ()
BUNDLES = PROJECT_ROOT / "cache" / "veins"
RUNS = PROJECT_ROOT / "cache" / "veins_runs" / "h3"
WSL_ROOT = "/mnt/c/Users/UPASANA/AI-HARP"


def settings() -> list[tuple[str, str, float]]:
    """(sweep family, Veins policy, knob) for the implemented families."""
    out = []
    for fam, (param, values) in POLICY_SWEEPS.items():
        if fam not in VEINS_POLICY:
            continue
        for v in (values or (0.0,)):
            out.append((fam, VEINS_POLICY[fam], float(v)))
    return out


def bundle_dir(sc: str, d: float, seed: int) -> Path:
    return BUNDLES / f"{sc}_d{d:g}_s{seed}"


def run_name(sc: str, d: float, seed: int, pol: str, knob: float, vseed: int) -> str:
    return f"{sc}_d{d:g}_s{seed}__{pol}_{knob:g}__s{vseed}"


def done(name: str) -> bool:
    p = RUNS / f"{name}.run.json"
    if not p.exists():
        return False
    r = json.loads(p.read_text())
    return r.get("exit_code") == 0 and bool(r.get("fcd_check", {}).get("identical"))


def wsl(path: Path) -> str:
    return WSL_ROOT + "/" + path.relative_to(PROJECT_ROOT).as_posix()


def run_one(args: tuple) -> str:
    sc, d, seed, pol, knob, vseed = args
    subprocess.run(["wsl", "-d", "Ubuntu", "--", "python3", "/home/upasana/veins-r5/run_veins.py",
                    wsl(bundle_dir(sc, d, seed)), pol, repr(knob), str(vseed), wsl(RUNS)],
                   capture_output=True, text=True)
    name = run_name(sc, d, seed, pol, knob, vseed)
    return f"{name} {'ok' if done(name) else 'FAILED/non-identical (kept, will retry)'}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=["export", "run", "status"])
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--jobs", type=int, default=2)
    a = ap.parse_args(argv)
    if a.mode == "export":
        from experiments.revision5.veins.export import export
        for sc, d in HEADLINE:
            for s in a.seeds:
                if not (bundle_dir(sc, d, s) / "bundle.json").exists():
                    print("exported", export(sc, d, s), flush=True)
        return 0
    todo = [(sc, d, s, pol, knob, s) for sc, d in HEADLINE for s in a.seeds for _, pol, knob in settings()
            if (bundle_dir(sc, d, s) / "bundle.json").exists()]
    pending = [t for t in todo if not done(run_name(*t))]
    print(f"[veins] {len(todo)} runs ({len(settings())} settings; not implemented: {', '.join(NOT_IMPLEMENTED)}), "
          f"{len(pending)} to do", flush=True)
    if a.mode == "status":
        return 0
    RUNS.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(a.jobs) as ex:
        for msg in ex.map(run_one, pending):
            print(msg, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
