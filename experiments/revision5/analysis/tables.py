"""Revision-5 paper tables. Writes results/revision5/analysis/tables/*.tex (plain tabular, no booktabs).

    python -m experiments.revision5.analysis.tables

Every number is read from committed analysis outputs (h2.json, h3.json, h3_subepoch.json, secondary.json); nothing is
recomputed here. "--" marks an undefined value (no pick, no qualifying setting, or a non-finite cost).
  T-heldout   per headline cell: thresholds; adaptive classical, best rule and policy (seeds 0-9) on EVAL, with the safety
              columns (actionable miss, warned share, p95 TIR) and qualification under the absolute miss cap 0.40
  T-veins     network-layer agreement, Veins vs custom-on-SUMO, epoch mode and the frozen D6 sub-epoch mode
  T-ablation  encoder variants (seeds 0-4); T-ladder rungs A/B/C (seeds 0-4); T-gate gate study (seeds 0-9)
  T-interp    realised (SEL->EVAL) vs interpolated (Algorithm 1 on EVAL) cost per family, headline cells
  T-sens      the 18-setting sensitivity grid
"""

from __future__ import annotations

import json

import numpy as np

from experiments.revision5.analysis.common import OUT

TAB = OUT / "tables"
CELLS = ["rural_highway d=2", "rural_highway d=20", "rural_highway d=80", "urban_nlos d=20"]
SHORT = {"rural_highway d=2": "R2", "rural_highway d=20": "R20", "rural_highway d=80": "R80", "urban_nlos d=20": "U20"}


def _j(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def esc(s) -> str:
    return str(s).replace("_", r"\_").replace("%", r"\%")


def num(x, d=3) -> str:
    return "--" if x is None or (isinstance(x, float) and not np.isfinite(x)) or x == "inf" else f"{x:.{d}f}"


def pct(x) -> str:
    if x == "inf":
        return r"$+\infty$"
    return "--" if x is None else f"{100 * x:+.1f}\\%"


def yn(b) -> str:
    return "--" if b is None else ("yes" if b else "no")


def pick(p) -> str:
    if not p:
        return "--"
    f, k, v = p
    return esc(f) if not k else f"{esc(f)} ({esc(k)}={esc(v)})"


def tabular(cols: str, header: list[str], body: list[list[str]], caption_note: str) -> str:
    out = [f"% {caption_note}", f"\\begin{{tabular}}{{{cols}}}", r"\hline", " & ".join(header) + r" \\", r"\hline"]
    out += [" & ".join(r) + r" \\" for r in body]
    out += [r"\hline", r"\end{tabular}", ""]
    return "\n".join(out)


def save(name: str, text: str):
    TAB.mkdir(parents=True, exist_ok=True)
    p = TAB / f"{name}.tex"
    p.write_text(text, encoding="utf-8", newline="\n")
    return p


def policy_means(per: dict) -> dict:
    """Mean over the seeds that picked a setting (safety columns of the picks), and the count that qualify."""
    v = [x for x in per.values() if x is not None]
    m = lambda k: (float(np.mean([x[k] for x in v if x.get(k) is not None])) if v else None)  # noqa: E731
    return {"n_pick": len(v), "n_qual": sum(x["qualifies"] for x in v), "n_qual_abs": sum(x["qualifies_abs_cap_0.40"] for x in v),
            "miss": m("miss"), "warned": m("stop_deadline_warned_share_actionable"), "tir95": m("tir95")}


def t_heldout(sec, h2) -> str:
    g = sec["gate"]["cells"]
    body = []
    for c in CELLS:
        cell = g[c]
        q, m = cell["q_star"], cell["m_star"]
        for lab, s in (("Adaptive classical", cell["adaptive_classical"]), ("Best rule", cell["best_rule"])):
            body.append([SHORT[c], f"{q:.3f} / {m:.3f}", lab, pick(s and s["pick"]), num(s and s["cost"]),
                         yn(s and s["qualifies"]), yn(s and s["qualifies_abs_cap_0.40"]), num(s and s["miss"]),
                         num(s and s["stop_deadline_warned_share_actionable"]), num(s and s["tir95"], 2)])
        pol = cell["methods"]["learned_gate_tau0.5"]
        a, pm = pol["aggregate"], policy_means(pol["per_seed"])
        ci = a.get("cost_mean_ci95")
        body.append([SHORT[c], f"{q:.3f} / {m:.3f}", "Policy (10 seeds)",
                     f"{pm['n_pick']}/10 pick, {pm['n_qual']} qualify",
                     num(a["cost_mean"]) + (f" [{num(ci[0])}, {num(ci[1])}]" if ci else ""), yn(a["qualifies"]),
                     f"{pm['n_qual_abs']}/10 seeds", num(pm["miss"]), num(pm["warned"]), num(pm["tir95"], 2)])
    va, vr = h2["vs_adaptive_classical"], h2["vs_best_rule"]
    note = (f"T-heldout. EVAL seeds 200-229, selection on SEL 400-419. Policy cost: mean over seeds that pick, with "
            f"hierarchical 95% CI. H2 pooled relative cost difference vs adaptive classical "
            f"{100 * va['statistic_pooled_relative_cost_difference']:+.1f}% "
            f"[{100 * va['ci95'][0]:+.1f}%, {100 * va['ci95'][1]:+.1f}%]; vs best rule "
            f"{100 * vr['statistic_pooled_relative_cost_difference']:+.1f}% "
            f"[{100 * vr['ci95'][0]:+.1f}%, {100 * vr['ci95'][1]:+.1f}%]. Safety columns: mean actionable miss, "
            f"stop-deadline warned share (actionable), p95 time-to-inform (s); for the policy, means over picking seeds.")
    return tabular("lllllllrrr", ["Cell", "$q^*$ / $m^*$", "Method", "Pick", "EVAL cost", "Qual.", "Qual. (miss$\\le$0.40)",
                                  "Miss", "Warned", "p95 TIR"], body, note)


def t_veins(h3, h3s) -> str:
    body = []
    for c in CELLS:
        for lab, h in (("epoch", h3), ("sub-epoch (D6)", h3s)):
            n = h["network_layer"][c]
            b, dl = n["channel_busy_ratio"], n["single_hop_delay_median_s"]
            body.append([SHORT[c], lab, num(n["pdr_max_abs_diff"]), yn(n["pdr_criterion_pass"]),
                         f"{num(b['veins_median'])} / {num(b['custom_median'])}", pct(b["relative_diff"]),
                         yn(b["criterion_pass"]), f"{num(dl['veins'])} / {num(dl['custom'])}", yn(dl["criterion_pass"]),
                         f"{num(n['in_range_loss']['veins'])} / {num(n['in_range_loss']['custom'])}"])
    s, ss = h3["spearman"], h3s["spearman"]
    note = (f"T-veins. Veins vs custom-on-SUMO, network seeds 200-229, 55 settings. Criteria: PDR within 0.05 in every "
            f"50 m bin below the nominal range; busy ratio within 20%; median single-hop delay within 100 ms. Spearman rho "
            f"of matched costs: epoch {s['rho']:.3f} [{s['ci95'][0]:.3f}, {s['ci95'][1]:.3f}], p(rho<=0.6) "
            f"{s['p_rho_le_0.6']:.3f}; sub-epoch {ss['rho']:.3f} [{ss['ci95'][0]:.3f}, {ss['ci95'][1]:.3f}], p "
            f"{ss['p_rho_le_0.6']:.3f}. H3 fails in both modes.")
    return tabular("llrlllllll", ["Cell", "Custom mode", "PDR max gap", "Pass", "Busy ratio V / C", "Rel. diff.", "Pass",
                                  "Median delay V / C (s)", "Pass", "In-range loss V / C"], body, note)


def t_methods(sec_part, names: dict, note: str) -> str:
    body = []
    f = sec_part["cells"]["rural_highway d=20"]
    for key, lab in names.items():
        row = [lab, str(sec_part["cells_reached"][key])]
        for c in CELLS:
            a = sec_part["cells"][c]["methods"][key]["aggregate"]
            row.append(f"{a['n_qualify']}/{a['n_seeds']}" + (r"$^\ast$" if a["qualifies"] else ""))
        a = f["methods"][key]["aggregate"]
        ci = a.get("cost_mean_ci95")
        row += [num(a["cost_mean"]) + (r" $\pm$ " + num(a["cost_sd"]) if a["cost_sd"] is not None else ""),
                f"[{num(ci[0])}, {num(ci[1])}]" if ci else "--", pct(f["rural_d20_margin_vs_adaptive"][key])]
        body.append(row)
    return tabular("lr" + "l" * len(CELLS) + "lll", ["Method", "Cells reached"] + [f"{SHORT[c]} qual." for c in CELLS]
                   + ["R20 cost mean $\\pm$ SD", "R20 95\\% CI", "R20 margin vs adaptive"], body, note)


def t_interp(sec) -> str:
    fams = list(sec["interp"][CELLS[0]]["classical_families"])
    body = []
    for f in fams + ["rule"]:
        row = [esc(f)]
        for c in CELLS:
            v = sec["interp"][c]["classical_families"][f] if f != "rule" else sec["interp"][c]["rule"]
            row += [num(v["realised_SEL_to_EVAL"]), num(v["interpolated_EVAL_estimate"])]
        body.append(row)
    for c_seed in sorted(sec["interp"][CELLS[0]]["policy_seeds"], key=lambda s: int(s.split("_s")[1])):
        row = [f"policy {esc(c_seed)}"]
        for c in CELLS:
            v = sec["interp"][c]["policy_seeds"][c_seed]
            row += [num(v["realised_SEL_to_EVAL"]), num(v["interpolated_EVAL_estimate"])]
        body.append(row)
    hdr = ["Family / seed"] + [f"{SHORT[c]} {k}" for c in CELLS for k in ("real.", "interp.")]
    return tabular("l" + "rr" * len(CELLS), hdr, body,
                   "T-interp. Realised = SEL-selected setting scored on EVAL (primary); interpolated = Algorithm 1 "
                   "crossing on EVAL means (an estimate, not an implementable configuration).")


def t_sens(sec) -> str:
    s = sec["sensitivity"]
    body = []
    for k, v in s["settings"].items():
        q, tol, mode = k.split("_")
        row = [q[1:], tol[3:], mode]
        for c in CELLS:
            x = v[c]
            row.append(f"{num(x['adaptive_cost'])} / {len(x['policy_seeds_qualifying'])}")
        row += [pct(v["rural_highway d=20"]["margin"]), str(len(s["flags_vs_baseline"].get(k, [])))]
        body.append(row)
    return tabular("lll" + "l" * len(CELLS) + "rr", ["$q^*$ factor", "Miss tol.", "Mode"]
                   + [f"{SHORT[c]} adaptive cost / qual. seeds" for c in CELLS] + ["R20 margin", "Flags"], body,
                   f"T-sens. Baseline setting {s['baseline']}; flags = changes vs the baseline in oracle family, "
                   "adaptive pick or the set of qualifying policy seeds (listed in secondary.json).")


def main(argv=None) -> int:
    sec, h2, h3, h3s = _j("secondary.json"), _j("h2.json"), _j("h3.json"), _j("h3_subepoch.json")
    out = [
        save("T-heldout", t_heldout(sec, h2)),
        save("T-veins", t_veins(h3, h3s)),
        save("T-ablation", t_methods(sec["variants"], {"full_model": "Full model (GATv2)", "gcn": "GCN encoder",
                                                        "mlp": "MLP encoder", "norel": "No relevance feature"},
                                     "T-ablation. Encoder variants, training seeds 0-4. Qual.: seeds whose pick qualifies "
                                     "on EVAL; * = cell aggregate qualifies. R20 = rural highway d=20.")),
        save("T-ladder", t_methods(sec["ladder"], {"A_full_model": "A: full model", "B_oracle_reward": "B: oracle reward",
                                                    "C_oracle_input_nondeployable": "C: oracle input (non-deployable)"},
                                   "T-ladder. Rungs A/B/C, training seeds 0-4; columns as in T-ablation.")),
        save("T-gate", t_methods(sec["gate"], {"learned_gate_tau0.5": "Learned gate ($\\tau$=0.5)",
                                                "ungated_tau0": "Ungated ($\\tau$=0)",
                                                "random_gate_matched_rate": "Random gate (matched rate)"},
                                 "T-gate. Training seeds 0-9; columns as in T-ablation.")),
        save("T-interp", t_interp(sec)),
        save("T-sens", t_sens(sec)),
    ]
    for p in out:
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
