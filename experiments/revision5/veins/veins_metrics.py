"""H3: score a Veins replay with the repo's own metric code.

    python -m experiments.revision5.veins.veins_metrics <runs_dir> [--out results/revision5/veins/veins_rows.csv]

For every <name>.run.json in runs_dir (written by run_veins.py) the Veins event log is turned into a
sim.engine.RunResult on the canonical trace, hazard, risk field, PHY and MAC of that (cell, seed) --
exactly the objects run_single builds -- and scored with analysis.metrics.compute_metrics. Nothing is
re-derived by hand: informed_step/hops/informed_by/tx_count come from the log, the reception
denominator (in-nominal-range active receivers, excluding same-step transmitters) from the trace.
Unknowable in Veins and recorded as NaN: the custom simulator's failure-cause counters.

Per run it also writes the H3 raw ingredients (PDR-vs-distance counts per 50 m bin, single-hop delays,
channel busy ratio while active) to <name>.h3.json beside the log.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np

from common.config import load_yaml
from experiments.run_sim import RunSpec, run_single
from sim.engine import NO_STEP, RunResult

BIN_M = 50.0
NAN = float("nan")


@lru_cache(maxsize=8)
def context(scenario: str, density: float, seed: int):
    """Canonical trace/hazard/risk/phy/mac of the cell (run_single's own objects), plus its flooding run."""
    _, res = run_single(RunSpec(scenario=scenario, density_veh_km_lane=density, seed=seed,
                                policy="flooding", backend="sumo"), return_result=True)
    return res


def parse_events(path: Path) -> dict[str, list[list[str]]]:
    ev: dict[str, list[list[str]]] = {k: [] for k in "OTRDX"}
    with gzip.open(path, "rt") as f:
        for line in f:
            p = line.rstrip("\n").split(",")
            ev.setdefault(p[0], []).append(p[1:])
    return ev


def scalars(path: Path) -> dict[str, dict[str, float]]:
    """{module: {name: value}} from an OMNeT++ .sca(.gz)."""
    out: dict[str, dict[str, float]] = {}
    if not path.exists():
        return out
    with gzip.open(path, "rt") as f:
        for line in f:
            if line.startswith("scalar "):
                _, mod, name, val = line.split(maxsplit=3)
                try:
                    out.setdefault(mod, {})[name] = float(val)
                except ValueError:
                    pass
    return out


def reconstruct(ev: dict, ctx: RunResult) -> tuple[RunResult, dict]:
    tr = ctx.trace
    N, T = tr.n_vehicles, tr.n_steps
    R = float(ctx.meta["comm_range_m"])
    informed_step = np.full(N, NO_STEP, dtype=np.int32)
    hops = np.zeros(N, dtype=np.int16)
    informed_by = np.full(N, -1, dtype=np.int32)
    tx_count = np.zeros(N, dtype=np.int32)
    origin_index = origin_step = NO_STEP
    for t, step, idx in ev["O"]:
        origin_index, origin_step = int(idx), int(step)
        informed_step[origin_index] = origin_step
    tx_steps: dict[int, list[int]] = {}
    tx_time: dict[tuple[int, int], float] = {}
    for t, step, idx, h, sched in ev["T"]:
        i, s = int(idx), int(step)
        tx_count[i] += 1
        tx_steps.setdefault(s, []).append(i)
        tx_time[(i, s)] = float(t)
    for t, step, idx, sender, dist, new, h, *_ in ev["R"]:
        if new == "1":
            r = int(idx)
            informed_step[r] = int(step)
            informed_by[r] = int(sender)
            hops[r] = int(h) + 1

    # reception accounting on the trace, as the engine does: in-nominal-range active receivers at the
    # transmission step, excluding the sender and anything else transmitting in that step
    decoded: dict[tuple[int, int], set[int]] = {}       # (sender, tx step) -> receivers that decoded it
    last_tx: dict[int, int] = {}
    tx_events = sorted((int(s), int(i)) for s, ii in tx_steps.items() for i in ii)
    rx_sorted = sorted(ev["R"], key=lambda p: float(p[0]))
    k = 0
    for t, step, idx, sender, dist, new, h, *_ in rx_sorted:
        while k < len(tx_events) and tx_time[(tx_events[k][1], tx_events[k][0])] <= float(t):
            last_tx[tx_events[k][1]] = tx_events[k][0]
            k += 1
        snd = int(sender)
        if snd in last_tx:
            decoded.setdefault((snd, last_tx[snd]), set()).add(int(idx))
    n_att = n_ok = 0
    bins_att: dict[int, int] = {}
    bins_ok: dict[int, int] = {}
    transmissions = []
    for s, senders in sorted(tx_steps.items()):
        act = tr.active[s].copy()
        act[senders] = False
        cand = np.flatnonzero(act)
        for i in senders:
            d = np.hypot(tr.x[s, cand] - tr.x[s, i], tr.y[s, cand] - tr.y[s, i])
            inr = cand[d <= R]
            dec = decoded.get((i, s), set())
            ok = np.array([r in dec for r in inr], dtype=bool)
            n_att += inr.size
            n_ok += int(ok.sum())
            for dd, o in zip(d[d <= R], ok):
                b = int(dd // BIN_M)
                bins_att[b] = bins_att.get(b, 0) + 1
                bins_ok[b] = bins_ok.get(b, 0) + int(o)
            transmissions.append({"step": s, "time_s": round(s * tr.dt, 3), "vehicle": int(i),
                                  "x": float(tr.x[s, i]), "y": float(tr.y[s, i]), "hop": int(hops[i]),
                                  "n_in_range": int(inr.size), "n_decoded": int(ok.sum())})
    action_counts: dict[str, int] = {}
    for p in ev["D"]:
        action_counts[p[4]] = action_counts.get(p[4], 0) + 1
    res = RunResult(
        informed_step=informed_step, hops=hops, informed_by=informed_by, tx_count=tx_count,
        origin_index=int(origin_index), origin_step=int(origin_step),
        n_transmissions=int(tx_count.sum()), n_rx_attempts=n_att, n_rx_success=n_ok,
        n_fail_sensitivity=NAN, n_fail_sinr=NAN, n_fail_beacon=NAN, n_backoff_ties=NAN,
        n_hidden_overlaps=NAN, n_busy_deferrals=NAN,
        n_dropped_ttl=sum(1 for p in ev["X"] if p[3] == "ttl"),
        transmissions=transmissions, action_counts=action_counts,
        trace=tr, hazard=ctx.hazard, risk=ctx.risk, phy=ctx.phy, mac=ctx.mac, meta=dict(ctx.meta),
    )
    # single-hop delay: reception of a NEW copy minus the time its sender itself became informed
    inf_time = {origin_index: None}
    for t, step, idx in ev["O"]:
        inf_time[int(idx)] = float(t)
    for t, step, idx, sender, dist, new, h, *_ in rx_sorted:
        if new == "1":
            inf_time[int(idx)] = float(t)
    delays = [float(t) - inf_time[int(sender)] for t, step, idx, sender, dist, new, h, *_ in rx_sorted
              if new == "1" and inf_time.get(int(sender)) is not None]
    raw = {"pdr_bins_m": BIN_M, "bins_attempts": bins_att, "bins_success": bins_ok,
           "single_hop_delays_s": delays}
    return res, raw


def busy_ratio(sca: dict) -> float:
    """Mean over vehicles of busy time while active / active time (Mac1609_4 busy signal)."""
    vals = [m["busyTimeWhileActive"] / m["activeTime"] for m in sca.values()
            if "activeTime" in m and m["activeTime"] > 1.0]
    return float(np.mean(vals)) if vals else NAN


def score(run_json: Path) -> dict:
    rec = json.loads(run_json.read_text())
    name = run_json.name[: -len(".run.json")]
    base = run_json.parent / name
    scen, rest = rec["bundle"].split("_d")
    dens, seed = rest.split("_s")
    row = {"bundle": rec["bundle"], "scenario": scen, "density": float(dens), "seed": int(seed),
           "policy": rec["policy"], "knob": rec["knob"], "veins_seed": rec["seed"], "exit_code": rec["exit_code"],
           "wall_s": rec["wall_s"], "fcd_identical": bool(rec["fcd_check"].get("identical")),
           "early_stop_s": NAN}
    events = Path(str(base) + ".events.csv.gz")
    if rec["exit_code"] != 0 or not events.exists():
        return row
    ctx = context(scen, float(dens), int(seed))
    res, raw = reconstruct(parse_events(events), ctx)
    from analysis.metrics import compute_metrics
    m = compute_metrics(res, load_yaml("hazard.yaml"))
    sca = scalars(Path(str(base) + ".sca.gz"))
    stop = [v["earlyStopAt"] for v in sca.values() if "earlyStopAt" in v]
    row["early_stop_s"] = stop[0] if stop else NAN
    raw["channel_busy_ratio"] = busy_ratio(sca)
    Path(str(base) + ".h3.json").write_text(json.dumps(raw))
    for k in ("rwcr", "causal_coverage", "tx_per_at_risk_informed", "actionable_deadline_miss_rate",
              "tir_median_s", "tir_p95_s", "pdr", "transmissions", "deadline_miss_rate"):
        row[k] = m.get(k, NAN)
    row["channel_busy_ratio"] = raw["channel_busy_ratio"]
    return row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs_dir", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    rows = [score(p) for p in sorted(a.runs_dir.glob("*.run.json"))]
    out = a.out or a.runs_dir / "veins_rows.csv"
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in rows[0], k)) if rows else []
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print({k: (round(v, 4) if isinstance(v, float) and not math.isnan(v) else v) for k, v in r.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
