"""Revision-5 protocol runner: every simulation track, one resumable task queue each.

::

    python -m experiments.revision5.sweep classical --seedset CAL  --cells grid24 --jobs 3
    python -m experiments.revision5.sweep rule      --seedset SEL  --cells grid24 --jobs 3
    python -m experiments.revision5.sweep agent     --seedset EVAL --cells headline --runs ref_s0 ref_s1 ... --jobs 3
    python -m experiments.revision5.sweep agent     --seedset EVAL --cells headline --runs ref_s0 --random-gate-from SEL

Options: --channel default|calibrated (R9 fitted extra loss), --backend fallback|sumo, --taus.
Seed sets, cells and grids come from results/revision5/protocol.json (pre-registered). Every row is one
experiments.run_sim.run_single call with config hash, metrics_version, git commit and seed
(experiments/revision1/runner.py); finished task hashes are skipped on restart. Output:
results/revision5/runs/<track>__<seedset>__<cells>__<channel>__<backend>[__<tag>].csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common.config import PROJECT_ROOT
from experiments.revision1.provenance import out_dir, stamp, write_json
from experiments.revision1.runner import load_rows, run_tasks, task
from experiments.revision1.sweeps import knob_settings

P = json.loads((PROJECT_ROOT / "results" / "revision5" / "protocol.json").read_text())
GRID24 = [(sc, float(d)) for sc in ("rural_highway", "urban_nlos", "urban_grid") for d in (1, 2, 3, 5, 10, 20, 40, 80)]
HEADLINE = [(sc, float(d)) for sc, d in P["cells"]["headline"]]
BIASES = (-2.0, -1.0, 0.0, 1.0, 2.0, 3.0)


def seeds(name: str) -> list[int]:
    return list(P["seed_sets"][name]["seeds"])


def cells(name: str) -> list[tuple[str, float]]:
    return HEADLINE if name == "headline" else HEADLINE + [c for c in GRID24 if c not in HEADLINE]


def phy_patch(channel: str):
    if channel == "default":
        return None
    f = json.loads((PROJECT_ROOT / "results" / "revision3" / "R9" / "r9_fit.json").read_text())
    return {"mac": {"extra_loss": {"p_max": f["p_max"], "n0": f["n0"], "n1": f["n1"]}}}


def rule_settings() -> list[tuple[str, str, object]]:
    """RAR (27) and SCF-slotted (12) grids from protocol.json, as (policy, param, value-json)."""
    out = []
    g = P["new_baselines"]["RAR"]["grid"]
    for a in g["a"]:
        for b in g["b"]:
            for c in g["c"]:
                out.append(("rar", "abc", json.dumps([a, b, c])))
    g = P["new_baselines"]["SCF_slotted"]["grid"]
    for n in g["n_slots"]:
        for t in g["carry_timeout_s"]:
            out.append(("scf_slotted", "slots_timeout", json.dumps([n, t])))
    return out


def checkpoint_for(run: str) -> Path:
    if run == "ref_s0":
        return PROJECT_ROOT / "checkpoints" / "campaign" / "ref" / "ckpt_final.pt"
    if run in ("gcn_s0", "mlp_s0", "star_s0"):
        return PROJECT_ROOT / "checkpoints" / "campaign" / run[:-3] / "ckpt_final.pt"
    if run == "norel_s0":
        return PROJECT_ROOT / "checkpoints" / "campaign" / "no_relevance" / "ckpt_final.pt"
    if "@" in run:                                   # budget checkpoint, e.g. ref_s0_u2000@1500
        name, upd = run.split("@")
        return PROJECT_ROOT / "checkpoints" / "revision1" / name / f"ckpt_{int(upd):06d}.pt"
    return PROJECT_ROOT / "checkpoints" / "revision1" / run / "ckpt_final.pt"


def gate_rates(runs: list[str], seedset: str, cellset: str, channel: str, backend: str) -> dict:
    """Learned-gate fallback rate per (run, cell, bias), measured on `seedset` (SEL by protocol)."""
    rates = {}
    for run in runs:
        f = out_dir("runs", 5) / f"agent__{seedset}__{cellset}__{channel}__{backend}.csv"
        acc = {}
        for r in load_rows(f):
            q = json.loads(r["params_json"])
            if q.get("checkpoint_sha") is None or float(q.get("tau", -1)) != 0.5 or "random_gate_rate" in q:
                continue
            if r.get("run") and r["run"] != run:
                continue
            k = (r["scenario"], float(r["density"]), float(r["value"]), q["checkpoint_sha"])
            acc.setdefault(k, []).append(float(r["gate_fallback_rate"]))
        rates.update({k: sum(v) / len(v) for k, v in acc.items()})
    return rates


def main(argv=None) -> int:
    from experiments.evaluate_agent import agent_params, checkpoint_sha

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("track", choices=["classical", "rule", "agent"])
    ap.add_argument("--seedset", required=True, choices=list(P["seed_sets"]))
    ap.add_argument("--cells", default="headline", choices=["headline", "grid24"])
    ap.add_argument("--channel", default="default", choices=["default", "calibrated"])
    ap.add_argument("--backend", default="fallback", choices=["fallback", "sumo"])
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--taus", type=float, nargs="*", default=[0.5])
    ap.add_argument("--random-gate-from", default=None, help="seed set whose learned-gate rates to match (SEL)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--jobs", type=int, default=3)
    a = ap.parse_args(argv)

    pp, sd, cl = phy_patch(a.channel), seeds(a.seedset), cells(a.cells)
    # Explicit, never "auto": "auto" picks SUMO whenever SUMO_HOME is set, which silently put the first SEL
    # sweeps on SUMO mobility (2026-10-19; quarantined in runs/invalid_sumo_backend/).
    bk = "sumo" if a.backend == "sumo" else "fallback"
    tasks = []
    if a.track == "classical":
        for sc, d in cl:
            for s in sd:
                for pol, par, val in knob_settings():
                    tasks.append(task(sc, d, pol, s, param=par, value=val, phy_patch=pp, backend=bk))
    elif a.track == "rule":
        for sc, d in cl:
            for s in sd:
                for pol, par, val in rule_settings():
                    tasks.append(task(sc, d, pol, s, param=par, value=val, phy_patch=pp, backend=bk))
    else:
        rates = (gate_rates(a.runs, a.random_gate_from, a.cells, a.channel, a.backend)
                 if a.random_gate_from else None)
        cks = {run: checkpoint_for(run) for run in a.runs}
        shas = {run: checkpoint_sha(ck) for run, ck in cks.items()}
        for sc, d in cl:                             # cell-major: headline cells (H2) finish first
            for run in a.runs:
                ck, sha = cks[run], shas[run]
                for s in sd:
                    for tau in ([0.5] if rates is not None else a.taus):
                        for b in BIASES:
                            if rates is not None:
                                r = rates.get((sc, d, b, sha))
                                if r is None:
                                    continue
                                fixed = agent_params(ck, deterministic=False, random_gate_rate=r)
                            else:
                                fixed = agent_params(ck, tau=tau, deterministic=False)
                            tasks.append(task(sc, d, "ai_harp", s, param="suppression_bias", value=b,
                                              fixed=fixed, phy_patch=pp, backend=bk))
    name = f"{a.track}__{a.seedset}__{a.cells}__{a.channel}__{a.backend}" + (f"__{a.tag}" if a.tag else "")
    d = out_dir("runs", 5)
    meta = d / f"{name}_meta.json"
    stamps = json.loads(meta.read_text()) if meta.exists() else []
    stamps.append(stamp(f"revision5/{name}", {"track": a.track, "cells": a.cells, "channel": a.channel,
                                              "backend": a.backend, "runs": a.runs, "taus": a.taus,
                                              "random_gate_from": a.random_gate_from, "phy_patch": pp},
                        seeds=sd, n_runs=len(tasks)))
    write_json(meta, stamps)
    rows = run_tasks(tasks, d / f"{name}.csv", jobs=a.jobs, log_every=500)
    print(f"[revision5] {name}: {len(rows)} / {len(tasks)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
