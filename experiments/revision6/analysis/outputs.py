"""Revision-6 table and figure, from results/revision6/analysis/veins_methods.json only (nothing is recomputed here).

    python -m experiments.revision6.analysis.outputs

T-veins-methods (results/revision6/analysis/tables/, plain tabular, no booktabs): per cell, thresholds, best classical,
best rule and the policy in Veins, then V1-V4. "--" marks an undefined value; "not run" marks the policy at rural d=80
(addendum D10). fig_veins_vs_custom (results/revision6/figures/; 3.5 in, fonttype 42, Liberation Serif): Veins vs
custom-on-SUMO matched cost on log-log axes, the V4 pairs.
"""

from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common.config import PROJECT_ROOT  # noqa: E402
from experiments.revision5.analysis.figures import W1, style  # noqa: E402
from experiments.revision5.analysis.tables import SHORT, esc, num, tabular  # noqa: E402

AN = PROJECT_ROOT / "results" / "revision6" / "analysis"
TAB = AN / "tables"
FIG = PROJECT_ROOT / "results" / "revision6" / "figures"
MARK = {"classical": ("o", "#1b6ca8", "Classical"), "rules": ("s", "#c0392b", "Rule (RAR, SCF)"), "policy": ("^", "#e08e0b", "Policy")}


def named(b: dict) -> str:
    return "--" if b["cost"] is None else f"{esc(b['family'])} {b['cost']:.3f}"


def p(x) -> str:
    return "--" if x is None else f"{x:.3f}"


def table(d: dict) -> str:
    body = []
    for c, e in d["cells"].items():
        run = c in d["policy_cells"]
        body.append([SHORT[c], num(e["q_star"]), num(e["m_star"]), named(e["best_classical"]), named(e["best_rule"]),
                     f"{e['policy_n_pick']}/{len(d['runs'])}" if run else "not run", num(e["policy_aggregate"]) if run else "not run"])
    a = tabular("lrrllrr", ["Cell", "$q^*$", "$m^*$", "Best classical", "Best rule", "Policy picks", "Policy cost"], body,
                "EVAL-hindsight comparison, Veins, Tier A seeds 200-209. Policy: training seeds 0-4; rural d=80 not run (addendum D10).")
    ci = lambda x: "--" if x is None else f"[{x[0]:+.3f}, {x[1]:+.3f}]"  # noqa: E731
    v1, v2, v3, v4 = d["V1"], d["V2"], d["V3"], d["V4"]
    cond = lambda v: f"{ci(v['ci95'])}, {d['B'] - v['undefined_replicates']}/{d['B']} replicates defined"  # noqa: E731
    b = tabular("lllrr", ["Test", "Statistic", "95\\% CI", "$p$", "Holm $p$"], [
        ["V1", f"{v1['statistic_cells_rule_cheaper']} cells of 4", "--", p(v1["p_fewer_than_2"]), p(v1["holm_p"])],
        ["V2", "undefined (no cell where both qualify)", cond(v2), p(v2["p_two_sided"]), p(v2["holm_p"])],
        ["V3", "undefined (no cell where both qualify)", cond(v3), p(v3["p_two_sided"]), p(v3["holm_p"])],
        ["V4", f"$\\rho$ = {v4['spearman_rho']:.3f}, {v4['n_pairs']} pairs", ci(v4["ci95"]), "--", "--"]],
        "V2 and V3 intervals are conditional on the replicates in which the statistic is defined; the point statistic is not.")
    return a + "\n" + b


def pairs(d: dict) -> list[tuple]:
    out = []
    for e in d["cells"].values():
        cu = e["custom_on_sumo"]
        for kind, ve, cs in (("classical", e["classical"], cu["classical"]), ("rules", e["rules"], cu["rules"]),
                             ("policy", {"p": e["policy_aggregate"]}, {"p": cu["policy_aggregate"]})):
            out += [(kind, cs[k], ve[k]) for k in ve if ve[k] is not None and cs[k] is not None]
    return out


def figure(d: dict):
    pts = pairs(d)
    if len(pts) != d["V4"]["n_pairs"]:
        raise SystemExit(f"{len(pts)} pairs, V4 has {d['V4']['n_pairs']}")
    style()
    fig, ax = plt.subplots(figsize=(W1, W1 * 0.8))
    lo = 0.5 * min(min(x, y) for _, x, y in pts)
    hi = 2 * max(max(x, y) for _, x, y in pts)
    ax.plot([lo, hi], [lo, hi], color="0.6", lw=0.6, ls="--", label="Equal cost")
    for kind, (m, col, lab) in MARK.items():
        got = [(x, y) for k, x, y in pts if k == kind]
        if got:
            ax.scatter(*zip(*got), marker=m, s=14, facecolor="none", edgecolor=col, linewidth=0.8, label=lab)
    ax.set(xscale="log", yscale="log", xlim=(lo, hi), ylim=(lo, hi), xlabel="Custom on SUMO, matched cost", ylabel="Veins, matched cost")
    ax.legend(frameon=False, loc="upper left")
    FIG.mkdir(parents=True, exist_ok=True)
    p_ = FIG / "fig_veins_vs_custom.pdf"
    fig.savefig(p_, metadata={"CreationDate": None, "ModDate": None, "Producer": None, "Creator": None})
    plt.close(fig)
    return p_


def main(argv=None) -> int:
    d = json.loads((AN / "veins_methods.json").read_text(encoding="utf-8"))
    TAB.mkdir(parents=True, exist_ok=True)
    t = TAB / "T-veins-methods.tex"
    t.write_text(table(d), encoding="utf-8", newline="\n")
    print(t)
    print(figure(d))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
