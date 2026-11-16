"""Revision 9, step 1: miss targets and gates A-C, on the TRAINING POOL only (seeds 100-131), before the addendum.

    python -m experiments.revision9.gates run --jobs 4      episodes -> results/revision9/gates_runs.csv (resumable)
    python -m experiments.revision9.gates analyse           -> results/revision9/miss_targets.json, gates.json

Episodes: every (policy setting, training group, seed 100-131) for the 55 classical settings (analysis/pareto.py
POLICY_SWEEPS), the registry defaults of flooding and DV-CAST (the coverage-target reference policies), always-suppress
(p_persistence p = 0: only the originator transmits) and the revision-8 STRUCT warm start (RAR (1, 1, 0.25) as STRUCT
parameters, results/revision8/warm_start.json). Training groups: the four (scenario, density) pairs revision-8 training
uses, clear weather, fog_bank, fallback mobility -- built exactly as run_single builds a run. Per episode: causal
coverage, causal at-risk count, transmissions, causal actionable miss (agents/causal_miss.py) and the ORACLE
actionable miss (analysis/metrics.py, for gate C only; training never sees it).

* b_g = min(pool-mean causal miss of flooding default, of DV-CAST default) + 0.05 per group.
* Gate A: per group, the number of the 55 classical settings whose pool means meet the coverage target
  (0.95 x the 32-seed ceiling, results/coverage_targets.json) AND causal miss <= b_g. Any group with 0 -> STOP.
* Gate B: the revision-9 fitness (agents/train_structured.fitness with the bounds) over all 4 x 32 pool episodes for
  always-suppress, flooding, slotted_1p 5 and 50, DV-CAST default, RAR warm start. Required: always-suppress infeasible;
  at least one efficient classical scheme (slotted_1p 5/50, DV-CAST) feasible and cheaper than flooding when flooding is
  feasible (or simply feasible when flooding is not).
* Gate C (descriptive): Spearman rank correlation between causal and oracle actionable miss per episode, over the
  episodes of those reference policies (pooled and per group). Nothing is tuned on it.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from common.config import PROJECT_ROOT, load_yaml

R9 = PROJECT_ROOT / "results" / "revision9"
RUNS = R9 / "gates_runs.csv"
GROUPS = (("rural_highway", 2.0), ("rural_highway", 20.0), ("rural_highway", 80.0), ("urban_nlos", 20.0))
POOL = range(100, 132)
FIELDS = ("key", "policy", "param", "value", "scenario", "density", "seed", "causal_coverage", "n_at_risk_causal",
          "transmissions", "causal_miss", "n_actionable_causal", "oracle_miss")


def settings() -> list[tuple[str, str, str, dict]]:
    """(label, policy, param, params) for every setting."""
    from experiments.revision1.sweeps import knob_settings
    out = []
    for pol, par, val in knob_settings():
        out.append((f"{pol}|{par}|{val}", pol, par, {} if not par else {par: val}))
    out.append(("flooding|default|", "flooding", "default", {}))
    out.append(("dvcast|default|", "dvcast", "default", {}))
    out.append(("always_suppress||", "p_persistence_03", "p", {"p": 0.0}))
    w = json.loads((PROJECT_ROOT / "results" / "revision8" / "warm_start.json").read_text())
    out.append(("rar_warm_start_struct||", "struct", "theta", {"theta": w["theta"]}))
    return out


def one(task):
    import logging
    logging.disable(logging.WARNING)
    from agents.causal_miss import causal_actionable_miss
    from agents.constrained_reward import causal_coverage, warned_at_risk
    from experiments.run_sim import RunSpec, run_single
    label, pol, par, params, sc, d, seed = task
    m, res = run_single(RunSpec(scenario=sc, density_veh_km_lane=d, weather="clear", policy=pol, policy_params=params,
                                seed=seed, hazard_type="fog_bank", backend="fallback"), return_result=True)
    rel = res.risk.relevance_matrix(res.trace, res.hazard)
    peak, at_risk, warned = warned_at_risk(rel, res.informed_step, res.risk.at_risk_threshold)
    cm, na = causal_actionable_miss(rel, res.informed_step, res.origin_step, res.trace, res.risk, res.hazard,
                                    load_yaml("hazard.yaml")["deadlines"])
    return {"key": f"{label}|{sc}|{d:g}|{seed}", "policy": pol, "param": par, "value": label.split("|")[2],
            "scenario": sc, "density": d, "seed": seed, "causal_coverage": causal_coverage(peak, at_risk, warned),
            "n_at_risk_causal": int(at_risk.sum()), "transmissions": int(res.n_transmissions), "causal_miss": cm,
            "n_actionable_causal": na, "oracle_miss": m["actionable_deadline_miss_rate"], "label": label}


def run(jobs: int, slice_: str | None = None) -> int:
    """All episodes, or (``slice_`` = "i/n", GitHub runners) those whose key hashes to i mod n, written to
    gates_runs_slice<i>of<n>.csv. The analysis merges every file and keeps one row per key."""
    import hashlib
    R9.mkdir(parents=True, exist_ok=True)
    out = RUNS
    si = sn = None
    if slice_:
        si, sn = (int(x) for x in slice_.split("/"))
        out = R9 / f"gates_runs_slice{si}of{sn}.csv"
    done = set()
    if out.exists():
        with open(out, newline="") as f:
            done = {r["key"] for r in csv.DictReader(f)}
    tasks = [(lab, pol, par, prm, sc, d, s) for lab, pol, par, prm in settings() for sc, d in GROUPS for s in POOL
             if f"{lab}|{sc}|{d:g}|{s}" not in done]
    if slice_:
        tasks = [t for t in tasks if int(hashlib.sha256(f"{t[0]}|{t[4]}|{t[5]:g}|{t[6]}".encode()).hexdigest(), 16)
                 % sn == si]
    print(f"[gates] {len(tasks)} episodes to run ({len(done)} done)", flush=True)
    new = not out.exists()
    with open(out, "a", newline="") as f, ProcessPoolExecutor(jobs) as ex:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        for k, fut in enumerate(as_completed([ex.submit(one, t) for t in tasks]), 1):
            w.writerow(fut.result())
            if k % 200 == 0:
                f.flush()
                print(f"[gates] {k}/{len(tasks)}", flush=True)
    return 0


def _num(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def analyse() -> int:
    from scipy.stats import spearmanr

    from agents.constrained_reward import CoverageTargets
    from agents.train_structured import fitness
    cfg = load_yaml("agent.yaml")["objective"]
    targets = CoverageTargets.load(PROJECT_ROOT / cfg["targets_path"], float(cfg["target_fraction"]),
                                   float(cfg["fallback_target"]))
    rows, seen = [], set()
    for p in [RUNS] + sorted(R9.glob("gates_runs_slice*of*.csv")):       # one row per key (first file wins)
        if p.exists():
            for r in csv.DictReader(open(p, newline="")):
                if r["key"] not in seen:
                    seen.add(r["key"])
                    rows.append(r)
    by: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        lab = r["key"].rsplit("|", 3)[0]
        by.setdefault((lab, f"{r['scenario']}|{float(r['density']):g}"), []).append(r)
    labels = [s[0] for s in settings()]
    gkeys = [f"{sc}|{d:g}" for sc, d in GROUPS]
    for lab in labels:
        for g in gkeys:
            n = len(by.get((lab, g), []))
            if n != len(POOL):
                raise SystemExit(f"{lab} {g}: {n} episodes, expected {len(POOL)}")

    def pool_mean(lab, g, col):
        v = np.array([_num(r[col]) for r in by[(lab, g)]])
        v = v[np.isfinite(v)]
        return float(v.mean()) if v.size else float("nan")

    # miss targets
    mt = {}
    for (sc, d), g in zip(GROUPS, gkeys):
        fl, dv = pool_mean("flooding|default|", g, "causal_miss"), pool_mean("dvcast|default|", g, "causal_miss")
        mt[g] = {"flooding_default_causal_miss": fl, "dvcast_default_causal_miss": dv, "b": min(fl, dv) + 0.05,
                 "coverage_target": targets.target(sc, d, "clear", "fog_bank")}
    from experiments.revision1.provenance import stamp, write_json
    write_json(R9 / "miss_targets.json", {
        "provenance": stamp("revision9/miss_targets", {"rule": "b_g = min(flooding, DV-CAST default pool-mean causal "
                                                                "actionable miss) + 0.05", "seeds": [100, 131]}),
        "groups": mt})

    # gate A
    classical = [lab for lab in labels if not lab.startswith(("flooding|default", "dvcast|default", "always_suppress",
                                                              "rar_warm"))]
    assert len(classical) == 55, len(classical)
    gate_a = {}
    for g in gkeys:
        ok = [lab for lab in classical if pool_mean(lab, g, "causal_coverage") >= mt[g]["coverage_target"]
              and pool_mean(lab, g, "causal_miss") <= mt[g]["b"]]
        cov_only = [lab for lab in classical if pool_mean(lab, g, "causal_coverage") >= mt[g]["coverage_target"]]
        gate_a[g] = {"n_meet_both": len(ok), "n_meet_coverage": len(cov_only), "settings_meeting_both": ok}
    a_pass = all(v["n_meet_both"] > 0 for v in gate_a.values())

    # gate B: the revision-9 fitness on all pool episodes
    bounds = {g: mt[g]["b"] for g in gkeys}
    refs = {"always_suppress": "always_suppress||", "flooding": "flooding|default|",
            "slotted_1p_5": "slotted_1p|n_slots|5", "slotted_1p_50": "slotted_1p|n_slots|50",
            "dvcast_default": "dvcast|default|", "rar_warm_start": "rar_warm_start_struct||"}
    gate_b = {}
    for name, lab in refs.items():
        outc, eps = [], []
        for (sc, d), g in zip(GROUPS, gkeys):
            for r in by[(lab, g)]:
                outc.append((_num(r["causal_coverage"]), int(r["n_at_risk_causal"]), int(r["transmissions"]),
                             _num(r["causal_miss"])))
                eps.append({"cell": (sc, d)})
        f = fitness(outc, eps, targets, bounds)
        gate_b[name] = {k: f[k] for k in ("f", "feasible", "cost", "shortfall", "miss", "miss_excess")}
    ranked = sorted(gate_b, key=lambda k: gate_b[k]["f"])
    eff = [k for k in ("slotted_1p_5", "slotted_1p_50", "dvcast_default") if gate_b[k]["feasible"]]
    fl = gate_b["flooding"]
    b_pass = (not gate_b["always_suppress"]["feasible"]) and bool(eff) and (
        not fl["feasible"] or any(gate_b[k]["f"] < fl["f"] for k in eff))

    # gate C (descriptive)
    gate_c = {}
    ref_rows = [r for lab in refs.values() for g in gkeys for r in by[(lab, g)]]
    def rho(rs):
        x = np.array([_num(r["causal_miss"]) for r in rs]); y = np.array([_num(r["oracle_miss"]) for r in rs])
        k = np.isfinite(x) & np.isfinite(y)
        if k.sum() < 3:
            return {"rho": None, "n": int(k.sum())}
        res = spearmanr(x[k], y[k])
        return {"rho": float(res.statistic), "p": float(res.pvalue), "n": int(k.sum()),
                "mean_causal": float(x[k].mean()), "mean_oracle": float(y[k].mean())}
    gate_c["pooled_reference_policies"] = rho(ref_rows)
    for g in gkeys:
        gate_c[g] = rho([r for lab in refs.values() for r in by[(lab, g)]])
    gate_c["pooled_all_57_settings"] = rho(rows)

    write_json(R9 / "gates.json", {
        "provenance": stamp("revision9/gates", {"seeds": [100, 131], "groups": gkeys}),
        "gate_A": {"pass": a_pass, "per_group": gate_a},
        "gate_B": {"pass": b_pass, "ranked_by_f": ranked, "per_policy": gate_b},
        "gate_C_descriptive": gate_c})
    print(json.dumps({"miss_targets": {g: round(v["b"], 4) for g, v in mt.items()},
                      "gate_A": {g: (v["n_meet_both"], v["n_meet_coverage"]) for g, v in gate_a.items()}, "A_pass": a_pass,
                      "gate_B": {k: (round(v["f"], 4), v["feasible"]) for k, v in gate_b.items()}, "B_pass": b_pass,
                      "gate_C": {k: (None if v["rho"] is None else round(v["rho"], 3)) for k, v in gate_c.items()}},
                     indent=1))
    return 0 if (a_pass and b_pass) else 1


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "run":
        raise SystemExit(run(int(a[a.index("--jobs") + 1]) if "--jobs" in a else 4,
                             a[a.index("--slice") + 1] if "--slice" in a else None))
    raise SystemExit(analyse())
