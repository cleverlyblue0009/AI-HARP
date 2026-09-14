"""Pre-generate every mobility trace training can sample.

::

    python -m experiments.warm_traces --jobs 16

Scenarios x curriculum densities x training weather x training-pool seeds
(2 x 8 x 2 x 32 = 1,024 traces, ~1 GB). Traces are cached under
``cache/traces`` keyed by configuration; writes are atomic, so parallel
generation is safe. Run once on a fresh machine before training, or every
rollout worker regenerates the same traces during the first updates.
"""

from __future__ import annotations

import argparse
import logging
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any

from agents.constrained_reward import curriculum_all_densities
from common.config import load_yaml


def _one(work: tuple[str, float, int, str, dict[str, Any]]) -> str:
    from mobility.generate import get_trace, load_scenario

    logging.getLogger("aiharp").setLevel(logging.WARNING)
    scenario, density, seed, weather, phy = work
    get_trace(load_scenario(scenario), density, seed, weather=weather, phy_cfg=phy)
    return f"{scenario} d={density:g} {weather} seed {seed}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Warm the mobility trace cache for training")
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args(argv)
    cfg = load_yaml("agent.yaml")
    phy = load_yaml("phy.yaml")
    t = cfg["training"]
    pool = t.get("train_seed_pool", {"start": 100, "count": 32})
    seeds = range(int(pool["start"]), int(pool["start"]) + int(pool["count"]))
    work = [(sc, d, s, w, phy) for sc in t["train_scenarios"]
            for d in curriculum_all_densities(t) for w in t["train_weather"] for s in seeds]
    t0 = time.time()
    print(f"warming {len(work)} traces with {args.jobs} jobs", flush=True)
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        for i, label in enumerate(ex.map(_one, work), 1):
            if i % 50 == 0 or i == len(work):
                print(f"  {i}/{len(work)} ({time.time() - t0:.0f} s) last: {label}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
