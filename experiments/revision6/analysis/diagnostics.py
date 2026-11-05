"""Revision 6: descriptive diagnostics behind veins_methods.json (results/revision6/analysis/veins_methods_diagnostics.json).

    python -m experiments.revision6.analysis.diagnostics

NOT pre-registered: written after veins_methods.json had been read, to explain it. Point estimates only, no tests; it
changes no V1-V4 number. Per cell and family: why a matched cost is or is not defined (best mean RWCR, lowest mean miss,
settings meeting the guard m* and both q* and m*, cost range); the policy per run and bias; whether the SCF timeout
changes any row (Veins and custom); RAR's best reach in both simulators; the point results without greedy_farthest and
DV-CAST, whose revision-5 Veins ports ran with zero beacon velocities (addendum D9); replay wall-hours per track.
"""

from __future__ import annotations

import json
import math
import sys

import numpy as np

from experiments.revision5.analysis.common import Table
from experiments.revision5.analysis.h3 import FAMILIES, veins_rows
from experiments.revision5.sweep import HEADLINE
from experiments.revision6.analysis.veins_methods import CK, RULES, SEEDS, V6, build, cell_eval, custom_rows, stats, thresholds, veins6_rows, write
from experiments.revision6.veins.queue6 import POLICY_CELLS

RUNS = [f"ref_s{k}" for k in range(5)]
NO_D9 = ("greedy_farthest", "dvcast")


def family(t: Table, fam, q, m) -> dict | None:
    ii = [i for i, k in enumerate(t.settings) if k[0] == fam]
    if not ii:
        return None
    rw, co, mi = t.mean("rwcr")[ii], t.mean("cost")[ii], t.mean("miss")[ii]
    ok = mi <= m
    return {"settings": len(ii), "max_rwcr": float(np.nanmax(rw)), "max_rwcr_within_guard": float(np.nanmax(rw[ok])) if ok.any() else None,
            "min_miss": float(np.nanmin(mi)), "meet_guard": int(ok.sum()), "meet_q_and_guard": int(((rw >= q) & ok).sum()),
            "cost_min": float(np.nanmin(co)), "cost_max": float(np.nanmax(co))}


def scf_timeout(t: Table) -> dict:
    """{n_slots: every timeout gives identical rows?} on all seeds, RWCR and cost."""
    by: dict[str, list[int]] = {}
    for i, k in enumerate(t.settings):
        if k[0] == "scf_slotted":
            by.setdefault(k[1].split(",")[0], []).append(i)
    return {n: bool(all(np.array_equal(t.m[c][ii[0]], t.m[c][i], equal_nan=True) for i in ii for c in ("rwcr", "cost")))
            for n, ii in by.items()}


def policy_grid(t: Table) -> dict:
    return {f"{k[1]:g}": {"rwcr": float(t.mean("rwcr")[i]), "cost": float(t.mean("cost")[i]), "miss": float(t.mean("miss")[i])}
            for i, k in enumerate(t.settings)}


def without(t: Table, fams) -> Table:
    ii = [i for i, k in enumerate(t.settings) if k[0] not in fams]
    return Table([t.settings[i] for i in ii], t.seeds, {c: a[ii] for c, a in t.m.items()})


def fin(x):
    return x if isinstance(x, float) and math.isfinite(x) else None


def wall_hours() -> dict:
    per: dict[tuple, list[float]] = {}
    for line in V6.open(encoding="utf-8"):
        r = json.loads(line)["row"]
        per.setdefault(("policy" if r["policy"] == "ai_harp" else "rules", CK((r["scenario"], r["density"]))), []).append(r["wall_s"])
    out = {}
    for (track, cell), w in sorted(per.items()):
        out.setdefault(track, {})[cell] = {"replays": len(w), "core_hours": sum(w) / 3600, "median_s": float(np.median(w)), "max_s": max(w)}
    for track, cells in out.items():
        cells["total"] = {"replays": sum(c["replays"] for c in cells.values()), "core_hours": sum(c["core_hours"] for c in cells.values())}
    return out


def main(argv=None) -> int:
    import numpy
    print(f"[env] {sys.executable} numpy {numpy.__version__}", flush=True)
    vr, _ = veins_rows()
    rr, pr = veins6_rows(RUNS)
    V = build(vr, rr, pr, RUNS)
    C = build(*custom_rows(RUNS), RUNS)
    cells, sens = {}, {}
    for c in HEADLINE:
        q, m = thresholds(V[c]["classical"])
        pol = {r: family(V[c]["policy"][r], r, q, m) for r in RUNS} if c in POLICY_CELLS else None
        grid = {r: policy_grid(V[c]["policy"][r]) for r in RUNS} if c in POLICY_CELLS else None
        cells[CK(c)] = {"q_star": q, "m_star": m, "classical": {f: family(V[c]["classical"], f, q, m) for f in FAMILIES},
                        "rules": {f: family(V[c]["rules"], f, q, m) for f in RULES}, "policy": pol, "policy_grid": grid,
                        "scf_timeout_inert": {"veins": scf_timeout(V[c]["rules"]), "custom": scf_timeout(C[c]["rules"])},
                        "rar_max_rwcr": {"veins": family(V[c]["rules"], "rar", q, m)["max_rwcr"],
                                         "custom": family(C[c]["rules"], "rar", q, m)["max_rwcr"]}}
        sens[c] = cell_eval({**V[c], "classical": without(V[c]["classical"], NO_D9)}, RUNS)
    st = stats(sens)
    payload = {
        "label": "Descriptive and post hoc (not pre-registered); point estimates on Tier A seeds 200-209, no tests",
        "cells": cells,
        "without_greedy_and_dvcast": {
            "note": "Point results with the two families whose revision-5 Veins ports had zero beacon velocities removed",
            "V1_statistic": st["V1"], "V2": st["V2"], "V3": st["V3"],
            "cells": {CK(c): {"q_star": e["q_star"], "m_star": e["m_star"], "best_classical": [e["best_classical"][0], fin(e["best_classical"][1])],
                              "best_rule": [e["best_rule"][0], fin(e["best_rule"][1])], "policy_n_pick": e["policy_n_pick"],
                              "policy_aggregate": fin(e["policy_aggregate"])} for c, e in sens.items()}},
        "wall_hours": wall_hours(),
    }
    print(write("veins_methods_diagnostics.json", payload, {"runs": RUNS}, seeds=SEEDS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
