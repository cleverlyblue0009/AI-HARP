"""Revision-5 figures (IEEE Access style). Writes results/revision5/figures/*.pdf.

    python -m experiments.revision5.analysis.figures

Style: pdf.fonttype 42 (embedded TrueType), Liberation Serif (vendored in ./fonts), 8 pt, single column 3.5 in,
double column 7.16 in. Every figure is drawn from committed analysis outputs or committed per-run rows only.
  fig_veins_pdr       PDR vs distance (50 m bins, below the nominal range), Veins vs custom (epoch, and the frozen
                      D6 sub-epoch mode when h3_subepoch.json exists), 4 headline cells
  fig_veins_delay     single-hop delay CDF, same simulators and cells (network seeds 200-209)
  fig_compute         classical random-search tuning: EVAL qualification rate vs tuning CPU-hours, with the policy's
                      training CPU-hours at the budget checkpoints marked
  fig_training        Lagrange multiplier of the focus group over training updates, full-model seeds
  fig_heldout         SEL vs EVAL mean cost of every classical setting, headline cells
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

from common.config import PROJECT_ROOT  # noqa: E402
from experiments.revision5.analysis.common import EVAL, HEADLINE, OUT, SEL, rows, tables  # noqa: E402

FIG = PROJECT_ROOT / "results" / "revision5" / "figures"
FONTS = Path(__file__).parent / "fonts"
VEINS = PROJECT_ROOT / "results" / "revision5" / "veins"
R1 = PROJECT_ROOT / "results" / "revision1" / "checkpoints"
W1, W2 = 3.5, 7.16
NET = range(200, 210)
COL = {"veins": "#1b6ca8", "custom": "#c0392b", "sub": "#e08e0b"}


def style() -> None:
    for f in FONTS.glob("*.ttf"):
        font_manager.fontManager.addfont(str(f))
    matplotlib.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "serif", "font.serif": ["Liberation Serif"],
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
        "ytick.labelsize": 7, "axes.linewidth": 0.6, "lines.linewidth": 1.0, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02, "axes.spines.top": False, "axes.spines.right": False,
        "pdf.compression": 9, "svg.hashsalt": "aiharp"})


def _save(fig, name: str) -> Path:
    FIG.mkdir(parents=True, exist_ok=True)
    p = FIG / f"{name}.pdf"
    fig.savefig(p, metadata={"CreationDate": None, "ModDate": None, "Producer": None, "Creator": None})
    plt.close(fig)
    return p


def _cells():
    return [f"{sc} d={d:g}" for sc, d in HEADLINE]


def _label(c: str) -> str:
    sc, d = c.split(" d=")
    return f"{'Rural highway' if sc == 'rural_highway' else 'Urban NLOS'}, {d} veh/km/lane"


def fig_veins_pdr() -> Path:
    h3 = json.loads((OUT / "h3.json").read_text())["network_layer"]
    sp = OUT / "h3_subepoch.json"
    h3s = json.loads(sp.read_text())["network_layer"] if sp.exists() else None
    fig, axs = plt.subplots(1, 4, figsize=(W2, 1.7), sharey=True)
    for ax, c in zip(axs, _cells()):
        b = h3[c]["pdr_vs_distance"]
        x = [0.5 * (r["bin_m"][0] + r["bin_m"][1]) for r in b]
        ax.plot(x, [r["pdr_veins"] for r in b], "o-", ms=2, color=COL["veins"], label="Veins")
        ax.plot(x, [r["pdr_custom"] for r in b], "s--", ms=2, color=COL["custom"], label="Custom (epoch)")
        if h3s:
            bs = h3s[c]["pdr_vs_distance"]
            ax.plot([0.5 * (r["bin_m"][0] + r["bin_m"][1]) for r in bs], [r["pdr_custom"] for r in bs], "^:", ms=2,
                    color=COL["sub"], label="Custom (sub-epoch, D6)")
        ax.set_title(_label(c))
        ax.set_xlabel("Distance (m)")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.25, lw=0.4)
    axs[0].set_ylabel("Packet delivery ratio")
    axs[0].legend(frameon=False, loc="lower left")
    return _save(fig, "fig_veins_pdr")


def _delays():
    from experiments.revision5.analysis.h3 import veins_rows
    _, vraw = veins_rows()
    cust = [json.loads(l) for l in (VEINS / "custom_h3.jsonl").open(encoding="utf-8")]
    sp = VEINS / "custom_h3_subepoch.jsonl"
    sub = [json.loads(l) for l in sp.open(encoding="utf-8")] if sp.exists() else []
    out = {}
    for sc, d in HEADLINE:
        c = f"{sc} d={d:g}"
        pick = lambda recs: np.sort([x for r in recs if r["scenario"] == sc and float(r["density"]) == d  # noqa: E731
                                     and int(r["seed"]) in NET for x in r["single_hop_delays_s"]])
        out[c] = {"veins": np.sort([x for (s2, d2, sd, _k), v in vraw.items() if s2 == sc and d2 == d and sd in NET
                                    and v is not None for x in v["single_hop_delays_s"]]),
                  "custom": pick(cust), "sub": pick(sub) if sub else None}
    return out


def fig_veins_delay() -> Path:
    D = _delays()
    fig, axs = plt.subplots(1, 4, figsize=(W2, 1.7), sharey=True)
    for ax, c in zip(axs, _cells()):
        for k, lab, ls in (("veins", "Veins", "-"), ("custom", "Custom (epoch)", "--"), ("sub", "Custom (sub-epoch, D6)", ":")):
            x = D[c][k]
            if x is None or not len(x):
                continue
            ax.step(x, np.arange(1, len(x) + 1) / len(x), where="post", ls=ls, color=COL[k], label=lab)
        ax.set_xscale("log")
        ax.set_title(_label(c))
        ax.set_xlabel("Single-hop delay (s)")
        ax.grid(alpha=0.25, lw=0.4)
    axs[0].set_ylabel("CDF")
    axs[-1].legend(frameon=False, loc="lower right", handlelength=1.2, handletextpad=0.4, borderaxespad=0.1)
    return _save(fig, "fig_veins_delay")


def fig_compute() -> Path:
    cc = json.loads((OUT / "compute_curves.json").read_text())
    fig, ax = plt.subplots(figsize=(W1, 2.2))
    mk = dict(zip(_cells(), "osD^"))
    for i, (c, res) in enumerate(cc["classical_random_search"].items()):
        for j, s in enumerate(sorted({r["s_sel_seeds"] for r in res})):    # one line per SEL-seed count, along k
            rr = sorted((r for r in res if r["s_sel_seeds"] == s), key=lambda r: r["k_settings"])
            ax.plot([r["tuning_cpu_hours_mean"] for r in rr], [r["eval_qualification_rate"] for r in rr],
                    marker=mk.get(c, "o"), ms=2.5, lw=0.6, color=f"C{i}", alpha=(0.45, 0.7, 1.0)[j % 3],
                    label=_label(c) if j == 2 else None)
    u = sorted({k for v in cc["policy_budget_cpu"].values() for k in v}, key=lambda k: int(k[1:]))
    for k in u:
        h = np.mean([v[k]["training_cpu_hours_estimate"] for v in cc["policy_budget_cpu"].values()])
        ax.axvline(h, color="0.5", lw=0.5, ls=":")
    ax.text(np.mean([v[u[0]]["training_cpu_hours_estimate"] for v in cc["policy_budget_cpu"].values()]), 0.02,
            "policy training\n(1,000-2,000 updates) ", fontsize=6, color="0.35", va="bottom", ha="right")
    ax.set_xscale("log")
    ax.set_xlabel("CPU-hours\n(classical: 5 / 10 / 20 SEL seeds light to dark, k = 5-55 settings along each line)",
                  fontsize=7)
    ax.set_ylabel("EVAL qualification rate")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25, lw=0.4)
    ax.legend(frameon=False, fontsize=6, loc="center left")
    return _save(fig, "fig_compute")


def fig_training() -> Path:
    td = json.loads((OUT / "training_diagnostics.json").read_text())
    g = td["focus_group"]
    fig, ax = plt.subplots(figsize=(W1, 2.0))
    for k in range(10):
        p = R1 / f"ref_s{k}_history.jsonl"
        if not p.exists():
            continue
        xs, ys = [], []
        for line in p.open(encoding="utf-8"):
            h = json.loads(line)
            v = (h.get("lambda_by_group") or {}).get(g)
            if v is not None:
                xs.append(h["update"])
                ys.append(v)
        ax.plot(xs, ys, lw=0.6, alpha=0.8, label=f"seed {k}" if k < 10 else None)
    ax.set_yscale("log")
    ax.set_xlabel("Training update")
    ax.set_ylabel(f"Multiplier $\\lambda$ ({g.replace('|', ', d=')})")
    ax.grid(alpha=0.25, lw=0.4)
    ax.legend(frameon=False, fontsize=5.5, ncol=2)
    return _save(fig, "fig_training")


def fig_heldout() -> Path:
    sel = tables(rows("classical__SEL__grid24__default__fallback.csv"), HEADLINE, SEL)
    ev = tables(rows("classical__EVAL__grid24__default__fallback.csv"), HEADLINE, EVAL)
    fig, ax = plt.subplots(figsize=(W1, 2.4))
    lo, hi = math.inf, 0.0
    for (c, mk) in zip(HEADLINE, "osD^"):
        s, e = sel[c], ev[c]
        x = s.mean("cost")
        y = np.array([e.mean("cost")[e.settings.index(k)] for k in s.settings])
        ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
        ax.scatter(x[ok], y[ok], s=5, marker=mk, lw=0.3, label=_label(f"{c[0]} d={c[1]:g}"))
        lo, hi = min(lo, x[ok].min(), y[ok].min()), max(hi, x[ok].max(), y[ok].max())
    ax.plot([lo, hi], [lo, hi], color="0.5", lw=0.5, ls="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("SEL mean cost (tx per at-risk informed)")
    ax.set_ylabel("EVAL mean cost")
    ax.grid(alpha=0.25, lw=0.4)
    ax.legend(frameon=False, fontsize=6)
    return _save(fig, "fig_heldout")


def main() -> int:
    style()
    for f in (fig_veins_pdr, fig_veins_delay, fig_compute, fig_training, fig_heldout):
        print(f())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
