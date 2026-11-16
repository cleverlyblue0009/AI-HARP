"""Revision 8: generate the training-pool traces (seeds 100-131 only) for the four headline cells, both backends.

    SUMO_HOME=... python -m experiments.revision8.traces --backend sumo --jobs 4
    python -m experiments.revision8.traces --backend fallback --jobs 4

SUMO 1.19.0 (eclipse-sumo==1.19.0 here; D:\\sumo-1.19.0 on the laptop). Only training-pool seeds are generated, never
CAL/SEL/EVAL. Writes results/revision8/traces_<backend>.json (per trace: cell, seed, n_vehicles, n_steps, file).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

from common.config import PROJECT_ROOT, load_yaml

CELLS = (("rural_highway", 2.0), ("rural_highway", 20.0), ("rural_highway", 80.0), ("urban_nlos", 20.0))
POOL = range(100, 132)


def one(args):
    sc, d, s, backend = args
    import logging
    logging.disable(logging.WARNING)
    from mobility.generate import get_trace, load_scenario
    tr = get_trace(load_scenario(sc), d, s, weather="clear", backend=backend, phy_cfg=load_yaml("phy.yaml"))
    return {"scenario": sc, "density": d, "seed": s, "n_vehicles": int(tr.n_vehicles), "n_steps": int(tr.n_steps),
            "backend": tr.meta.get("backend", backend)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=["sumo", "fallback"])
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    a = ap.parse_args(argv)
    print(f"[env] {sys.executable} SUMO_HOME={os.environ.get('SUMO_HOME')}", flush=True)
    tasks = [(sc, d, s, a.backend) for sc, d in CELLS for s in POOL]
    with ProcessPoolExecutor(a.jobs) as ex:
        out = list(ex.map(one, tasks))
    p = PROJECT_ROOT / "results" / "revision8" / f"traces_{a.backend}.json"
    p.write_text(json.dumps({"backend": a.backend, "seeds": [POOL[0], POOL[-1]], "traces": out}, indent=1))
    print(p, len(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
