"""R10: how the derived deadline's assumptions move Algorithm 1 (p95-TIR bound).

::

    python -m experiments.revision3.r10_deadline_assumptions

D_r = (R - v t_r - v^2 / (2 a)) / v per scenario (revision-1 B1), for driver reaction
t_r in {1.5, 2.5, 3.5} s and deceleration a in {2.5, 3.4, 6.0} m/s^2; v, R as in B1
(rural 27.78 m/s & 610.4 m LOS; urban_nlos 15.28 m/s & 101.4 m NLOS-corner). D_r < 0
(the vehicle cannot stop within one hop at all) is reported and treated as D = 0. Applied
as a bound on seed-mean p95 TIR (as R1), headline cells, q* and guard fixed; agent = full
model tau = 0.5, training seeds 0-4. Each configuration is compared with the paper's
(t_r = 2.5 s, a = 3.4 m/s^2).
"""

from __future__ import annotations

import itertools

from experiments.revision1.b1_latency import derived_deadlines
from experiments.revision1.provenance import out_dir, stamp, write_json
from experiments.revision2.r1_tail_latency import analyse

TR = (1.5, 2.5, 3.5)
ACC = (2.5, 3.4, 6.0)
BASE = (2.5, 3.4)


def main() -> int:
    geo = derived_deadlines()
    bounds, dvals = {}, {}
    for tr, a in itertools.product(TR, ACC):
        d = {}
        for sc, g in geo.items():
            v, R = g["v_mps"], g["range_m"]
            d[sc] = (R - v * tr - v * v / (2 * a)) / v
        dvals[f"tr={tr}|a={a}"] = d
        bounds[f"tr={tr}|a={a}"] = {sc: max(0.0, x) for sc, x in d.items()}
    res = analyse("headline", bounds)
    base = res["bounds"][f"tr={BASE[0]}|a={BASE[1]}"]
    table = {}
    for key, b in res["bounds"].items():
        ch = []
        if b["fixed_best"] != base["fixed_best"]:
            ch.append(f"best fixed {base['fixed_best']} -> {b['fixed_best']}")
        for ck, e in b["cells"].items():
            be = base["cells"][ck]
            short = ck.replace("/clear/fog_bank", "")
            if e["oracle_policy"] != be["oracle_policy"] or (e["oracle_cost"] or -1) != (be["oracle_cost"] or -1):
                ch.append(f"{short}: oracle {be['oracle_policy']} {be['oracle_cost']} -> {e['oracle_policy']} {e['oracle_cost']}")
            for s, a in e["agent"].items():
                if (a["cost"] is None) != (be["agent"][s]["cost"] is None):
                    ch.append(f"{short}: seed {s} {'gains' if a['cost'] is not None else 'loses'} the cell")
        table[key] = {"D_raw_s": dvals[key], "fixed_best": b["fixed_best"],
                      "agent_cells_matched": b["agent_cells_matched"],
                      "cells_no_qualifier": b["cells_no_qualifier"], "changes_vs_paper": ch}
    out = {"geometry": geo, "configs": table, "detail": res,
           "provenance": stamp("R10", {"t_r": TR, "a": ACC, "metric": "tir_p95_s", "paper": BASE},
                               seeds={"eval": list(range(10)), "train": res["seeds"]})}
    write_json(out_dir("R10", 3) / "r10_deadline_assumptions.json", out)
    for k, t in table.items():
        dr = ", ".join(f"{sc.split('_')[0][:5]}{'/' + sc.split('_')[1][:4] if '_' in sc else ''} {x:.2f}s"
                       for sc, x in t["D_raw_s"].items())
        print(f"{k:14s} D: {dr} | fixed {t['fixed_best']} | matched {t['agent_cells_matched']} | "
              f"no-qualifier {t['cells_no_qualifier']} | changes {len(t['changes_vs_paper'])}: {t['changes_vs_paper'][:4]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
