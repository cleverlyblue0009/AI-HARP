"""Slot-granularity sensitivity: what the 100 ms epoch costs a deferring scheme.

::

    D:/aiharp-env/python.exe -m experiments.slot_granularity --jobs 2

The engine advances in 100 ms epochs, so a slotted scheme's wait is a whole
number of them: ``slotted_1p`` with ``slot_epochs = 1`` waits 100 ms per slot
where a real 802.11p implementation would wait tens of microseconds to a few
milliseconds. That inflates the absolute latency (TIR) of every deferring
scheme -- baselines and the agent alike -- and it is a stated limitation of
this simulator rather than a property of the schemes.

This sweeps ``slot_epochs`` over the committed comparator cells for the two
slot-based baselines and reports how TIR and cost move with it, so the paper
can say how much of the reported latency is granularity and how much is the
scheme. Rows are appended and skipped on restart; results/slot_granularity.csv.
"""

from __future__ import annotations

import argparse
import csv
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from common.config import PROJECT_ROOT, load_yaml

POLICIES = ("slotted_1p", "dvcast")
SLOT_EPOCHS = (1, 2, 5, 10)
METRICS = ("rwcr", "tir_median_s", "tir_p95_s", "tx_per_at_risk_informed",
           "actionable_deadline_miss_rate", "transmissions")
FIELDS = ("scenario", "density", "density_veh_km_lane", "weather", "hazard_type", "policy",
          "slot_epochs", "seed") + METRICS
KEY = ("scenario", "density", "weather", "hazard_type", "policy", "slot_epochs", "seed")
_CFGS: dict[str, Any] = {}


def _init() -> None:
    logging.disable(logging.INFO)
    _CFGS.update(phy=load_yaml("phy.yaml"), hazard=load_yaml("hazard.yaml"),
                 experiment=load_yaml("experiment.yaml"))


def run_one(task: tuple) -> dict[str, Any]:
    from experiments.run_sim import RunSpec, run_single

    if not _CFGS:
        _init()
    sc, d, w, h, pol, epochs, seed = task
    spec = RunSpec(scenario=sc, density_veh_km_lane=d, weather=w, policy=pol,
                   policy_params={"slot_epochs": int(epochs)}, seed=seed, hazard_type=h)
    m, _ = run_single(spec, phy_cfg=_CFGS["phy"], hz_cfg=_CFGS["hazard"],
                      exp_cfg=_CFGS["experiment"])
    return {"scenario": sc, "density": d, "density_veh_km_lane": d, "weather": w,
            "hazard_type": h, "policy": pol, "slot_epochs": int(epochs), "seed": int(seed),
            **{k: m.get(k, np.nan) for k in METRICS}}


def _key(row: dict[str, Any]) -> tuple[str, ...]:
    return tuple(f"{float(row[k]):g}" if k == "density" else str(row[k]) for k in KEY)


def tasks_for(cells: list[dict[str, Any]], seeds: int, epochs=SLOT_EPOCHS,
              policies=POLICIES) -> list[tuple]:
    return [(c["scenario"], float(c["density"]), c.get("weather", "clear"),
             c.get("hazard_type", "fog_bank"), pol, e, s)
            for c in cells for pol in policies for e in epochs for s in range(seeds)]


def run(out_csv: Path, tasks: list[tuple], jobs: int = 1) -> int:
    done: set[tuple[str, ...]] = set()
    if out_csv.exists():
        with out_csv.open(newline="", encoding="utf-8") as fh:
            done = {_key(r) for r in csv.DictReader(fh)}
    todo = [t for t in tasks if _key(dict(zip(KEY, (t[0], t[1], t[2], t[3], t[4], t[5], t[6]))))
            not in done]
    print(f"{len(tasks)} runs, {len(tasks) - len(todo)} done, {len(todo)} to go", flush=True)
    if not todo:
        return 0
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    new = not out_csv.exists()
    t0 = time.time()
    with out_csv.open("a", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        if new:
            wr.writeheader()
        if jobs > 1:
            from concurrent.futures import ProcessPoolExecutor

            ex = ProcessPoolExecutor(max_workers=jobs, initializer=_init)
            it = ex.map(run_one, todo, chunksize=4)
        else:
            _init()
            ex, it = None, map(run_one, todo)
        try:
            for i, row in enumerate(it, 1):
                wr.writerow(row)
                if i % 100 == 0 or i == len(todo):
                    fh.flush()
                    print(f"  {i}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
        finally:
            if ex is not None:
                ex.shutdown(cancel_futures=True)
    return len(todo)


def summarise(rows: list[dict[str, Any]]) -> str:
    """TIR and cost against slot length, per cell and policy."""
    import collections

    by: dict[tuple, list[dict[str, Any]]] = collections.defaultdict(list)
    for r in rows:
        by[(r["scenario"], f"{float(r['density']):g}", r["policy"], int(r["slot_epochs"]))].append(r)
    cells = sorted({(k[0], k[1]) for k in by})
    out = ["Slot granularity: one slot = slot_epochs x 100 ms (the engine's epoch).", ""]
    for sc, d in cells:
        out.append(f"{sc} d={d}")
        for pol in POLICIES:
            line = [f"   {pol:<12}"]
            base = None
            for e in SLOT_EPOCHS:
                rs = by.get((sc, d, pol, e))
                if not rs:
                    continue
                tir = float(np.nanmean([float(r["tir_median_s"]) for r in rs]))
                cost = float(np.nanmean([float(r["tx_per_at_risk_informed"]) for r in rs]))
                rwcr = float(np.nanmean([float(r["rwcr"]) for r in rs]))
                base = base if base is not None else tir
                line.append(f"{e:>2}x: TIR {tir:5.2f}s ({tir / base:4.2f}x) cost {cost:4.2f} "
                            f"RWCR {rwcr:.3f} |")
            out.append(" ".join(line))
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--out", default="results/slot_granularity.csv")
    args = ap.parse_args(argv)
    cells = load_yaml("experiment.yaml")["comparator"]["cells"]
    out_csv = PROJECT_ROOT / args.out
    run(out_csv, tasks_for(cells, args.seeds), args.jobs)
    with out_csv.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    text = summarise(rows)
    out_csv.with_suffix(".txt").write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
