"""Phase 8: IEEE-style figures.

Every figure is emitted as both vector PDF (for the paper) and PNG at 300 dpi
(for quick viewing), in single-column (3.5 in) and double-column (7.16 in)
variants.

Colour
------
The categorical palette is Okabe-Ito, reordered so the weakest colour-vision
pair is not adjacent in assignment order, and capped at four series per
panel. Validated with the dataviz skill's checker rather than by eye:

* lightness band, chroma floor, normal-vision floor: PASS
* adjacent-pair CVD separation: PASS (worst adjacent dE 9.6, deutan)
* all-pairs CVD separation: dE 7.6 for green-vs-pink, which sits in the 6-8
  floor band and is legal **only with secondary encoding**

So secondary encoding is not decorative here, it is what makes the palette
legal: every series carries a distinct marker *and* a distinct line style. That
is also exactly what makes the figures readable in greyscale, which the paper
needs anyway -- the Okabe-Ito hues sit in a narrow lightness band (L 0.43-0.77)
and would collapse into each other when printed in black and white.

The sequential ramp for the attention heatmap is single-hue, light to dark, as
a magnitude encoding must be. Never a rainbow.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from common.config import RESULTS_DIR, ensure_dir
from common.logging_utils import get_logger

logger = get_logger("analysis.figures")

FIG_DIR = RESULTS_DIR / "figures"

#: IEEE column widths, inches.
SINGLE_COL = 3.5
DOUBLE_COL = 7.16

#: Okabe-Ito, reordered so the weakest colour-vision pair is not adjacent.
#: Validated (dataviz checker), light surface, ALL-PAIRS mode, per panel:
#:   lightness band / chroma floor / normal-vision floor : PASS
#:   CVD separation, worst all-pairs                     : dE 11.0 (deutan)
#: Only the first PANEL_SLOTS entries are ever used in one panel.
PALETTE: tuple[str, ...] = (
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#E69F00",  # orange
    "#CC79A7",  # reddish purple
    "#56B4E9",  # sky blue
)
#: Secondary encoding. Not decoration: the full 6-slot palette has an
#: all-pairs CVD minimum of dE 7.6 (green vs pink), which is legal ONLY with
#: secondary encoding. It is also what keeps the figures readable in
#: greyscale -- Okabe-Ito sits in a narrow lightness band (L 0.43-0.77), so
#: hue alone collapses when printed black and white.
MARKERS: tuple[str, ...] = ("o", "s", "^", "D", "v", "P")
LINESTYLES: tuple[Any, ...] = ("-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2)))

#: Hard cap on categorical series per panel.
#:
#: Seven simultaneous series cannot be coloured legally from any palette we
#: could validate: adding a seventh hue to Okabe-Ito put indigo against blue at
#: dE 11.7 for normal vision, below the 15 floor, which secondary encoding does
#: not excuse. Eight failed the chroma floor outright. So panels are faceted
#: instead of cycling hues -- cycling is what silently gave `flooding` and
#: `p_persistence_03` the same orange square in the first draft of this module.
PANEL_SLOTS = 4

GREY = "#4d4d4d"
GRID_GREY = "#d9d9d9"
REFERENCE_INK = "#7a7a7a"

#: Single-hue sequential ramp for magnitude (attention, RSSI).
SEQUENTIAL_HUE = "Blues"

#: Policy families, used to facet. Colour follows the ENTITY within a panel, so
#: a figure that drops a series never repaints the survivors.
FAMILIES: dict[str, tuple[str, ...]] = {
    "probabilistic": ("p_persistence_03", "weighted_p", "counter_based"),
    "topology-aware": ("ai_harp", "slotted_1p", "greedy_farthest", "dvcast"),
}

#: Drawn as a neutral reference mark in every panel rather than as a
#: categorical series. Flooding has no suppression knob -- its operating curve
#: is a single point by construction -- so it is a different kind of object
#: from the tunable schemes, and giving it a categorical slot both wastes a
#: scarce hue and implies a curve that does not exist.
REFERENCE_POLICY = "flooding"

PRETTY: dict[str, str] = {
    "ai_harp": "AI-HARP",
    "p_persistence_03": "p-persistence",
    "slotted_1p": "slotted 1-persistence",
    "weighted_p": "weighted p-persistence",
    "counter_based": "counter-based",
    "greedy_farthest": "greedy farthest",
    "dvcast": "DV-CAST",
    "flooding": "flooding (reference)",
}


def label_for(name: str) -> str:
    return PRETTY.get(name, name.replace("_", " "))


def family_of(policy: str) -> str:
    for fam, members in FAMILIES.items():
        if policy in members:
            return fam
    return "other"


def style_for(name: str, panel: Sequence[str]) -> dict[str, Any]:
    """Colour, marker and line style for a series WITHIN a panel.

    Raises if the panel holds more series than there are validated slots.
    Cycling would silently give two policies the same colour and marker, which
    is precisely the bug this replaced; failing loudly is the only safe
    behaviour.
    """
    members = [m for m in panel if m != REFERENCE_POLICY]
    if len(members) > PANEL_SLOTS:
        raise ValueError(
            f"{len(members)} categorical series in one panel exceeds the "
            f"{PANEL_SLOTS} validated slots ({members}). Facet the figure or "
            "fold the extras into 'other' -- do not cycle the palette."
        )
    i = members.index(name)
    return {"color": PALETTE[i], "marker": MARKERS[i], "linestyle": LINESTYLES[i]}


def reference_style() -> dict[str, Any]:
    """Neutral ink for the non-tunable reference mark."""
    return {"color": REFERENCE_INK, "marker": "X", "linestyle": "none",
            "markersize": 5, "markeredgewidth": 0.8}


def apply_ieee_style() -> None:
    """Matplotlib rcParams for camera-ready IEEE figures."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Nimbus Roman"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        # Recessive grid and axes; the data carries the ink.
        "axes.grid": True,
        "grid.color": GRID_GREY,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.8,
        "axes.edgecolor": GREY,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": GREY,
        "ytick.color": GREY,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "lines.linewidth": 1.2,
        "lines.markersize": 3.5,
        "legend.frameon": False,
        "errorbar.capsize": 2,
    })


def save(fig, name: str, out_dir: Path | None = None) -> list[Path]:
    """Write PDF (vector, for the paper) and PNG (for quick viewing)."""
    out_dir = ensure_dir(out_dir or FIG_DIR)
    paths = []
    for ext in ("pdf", "png"):
        p = out_dir / f"{name}.{ext}"
        fig.savefig(p)
        paths.append(p)
    logger.info("figure -> %s.{pdf,png}", name)
    return paths


def greyscale_check(path: Path) -> dict[str, float]:
    """Report the luminance spread of a rendered figure.

    A sanity check, not a proof: it confirms the figure uses a range of
    luminances rather than a set of equally-dark hues. The real guarantee comes
    from the distinct markers and line styles.
    """
    try:
        import matplotlib.image as mpimg
    except ImportError:  # pragma: no cover
        return {}
    img = mpimg.imread(path)
    if img.ndim < 3:
        return {}
    lum = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
    ink = lum[lum < 0.95]
    return {
        "ink_fraction": float(ink.size / lum.size),
        "luminance_p05": float(np.percentile(ink, 5)) if ink.size else float("nan"),
        "luminance_p95": float(np.percentile(ink, 95)) if ink.size else float("nan"),
    }


# ---------------------------------------------------------------------------
# Figure 1 (headline): overhead vs RWCR Pareto front
# ---------------------------------------------------------------------------
def fig_pareto(
    cells, cell_index: int = 1, width: float = DOUBLE_COL, name: str = "fig01_pareto"
) -> list[Path]:
    """Operating curves and the Pareto front, faceted by policy family.

    The headline figure: the question is not who scores highest but who reaches
    a given RWCR most cheaply. Faceted because seven series cannot be coloured
    legally in one panel (see PANEL_SLOTS).

    Flooding appears in both panels as a neutral reference point -- it has no
    suppression knob, so it is one point rather than a curve.
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogLocator, NullFormatter, ScalarFormatter

    from analysis.pareto import pareto_front

    apply_ieee_style()
    cell = cells[cell_index]
    fams = [f for f, members in FAMILIES.items()
            if any(m in cell.curves for m in members)]
    fig, axes = plt.subplots(1, len(fams), figsize=(width, width * 0.42),
                             sharey=True)
    axes = np.atleast_1d(axes)

    all_points = [p for c in cell.curves.values() for p in c.points]
    front = pareto_front(all_points)
    ref = cell.curves.get(REFERENCE_POLICY)

    for ax, fam in zip(axes, fams):
        panel = [m for m in FAMILIES[fam] if m in cell.curves]
        if front:
            ax.plot([p.cost for p in front], [p.quality for p in front],
                    color="black", linewidth=0.7, alpha=0.35, zorder=1,
                    label="Pareto front (all schemes)")
        for pol in panel:
            pts = sorted(cell.curves[pol].points, key=lambda p: p.cost)
            if not pts:
                continue
            ax.plot([p.cost for p in pts], [p.quality for p in pts],
                    label=label_for(pol), zorder=3, **style_for(pol, panel))
        if ref and ref.points:
            rp = ref.points[0]
            ax.plot([rp.cost], [rp.quality], label=label_for(REFERENCE_POLICY),
                    zorder=4, **reference_style())
        ax.set_xscale("log")
        # Decade labels alone leave a log axis with a single tick; label the
        # 1-2-5 subdivisions so the reader can actually read a cost off it.
        ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0), numticks=12))
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("Transmissions per at-risk vehicle informed")
        ax.set_title(fam, pad=3)
        ax.legend(loc="lower right")

    axes[0].set_ylabel("RWCR")
    fig.suptitle(f"{cell.key.scenario.replace('_', ' ')}, "
                 f"{cell.key.density:g} veh/km/lane", y=1.02, fontsize=8)
    return save(fig, name)


# ---------------------------------------------------------------------------
# Causal-vs-oracle estimation agreement (results/risk_estimation.csv)
# ---------------------------------------------------------------------------
def fig_estimation_agreement(
    df, width: float = SINGLE_COL, name: str = "fig04_estimation_agreement",
    metric: str = "risk_peak_corr",
) -> list[Path]:
    """How well the causal risk estimate recovers the oracle, against density.

    The estimation problem the learned policy is asked to overcome, and it is
    reported before any policy result: on a corridor heading determines
    destiny (correlation ~0.75), in a grid it does not (~0.17). Independent of
    the dissemination policy, so it has its own sweep.
    """
    import matplotlib.pyplot as plt

    apply_ieee_style()
    fig, ax = plt.subplots(figsize=(width, width * 0.72))
    panel = sorted(df["scenario"].unique())
    for scenario in panel:
        sub = df[df["scenario"] == scenario]
        g = sub.groupby("density")[metric]
        mu, sd = g.mean(), g.std()
        ax.errorbar(mu.index, mu.to_numpy(), yerr=sd.to_numpy(),
                    label=label_for(scenario), **style_for(scenario, panel))
    ax.set_xscale("log")
    ax.set_xticks(sorted(df["density"].unique()))
    ax.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Density (veh/km/lane)")
    ax.set_ylabel("Causal vs oracle peak-relevance correlation")
    ax.set_ylim(-0.05, 1.0)
    ax.legend(loc="best")
    return save(fig, name)


# ---------------------------------------------------------------------------
# Latency / overhead inversion (results/runs.csv)
# ---------------------------------------------------------------------------
def fig_latency_cost_inversion(
    df, scenario: str, density: float, width: float = SINGLE_COL,
    name: str = "fig09_latency_cost_inversion",
) -> list[Path]:
    """Cost against latency for every scheme in one cell.

    Flooding buys the best latency with the worst overhead and the suppression
    schemes do the reverse, which is why neither axis alone ranks them. Nine
    schemes exceed the four validated colour slots, so the points are neutral
    and labelled directly rather than cycling the palette.
    """
    import matplotlib.pyplot as plt

    apply_ieee_style()
    fig, ax = plt.subplots(figsize=(width, width * 0.78))
    sub = df[(df["scenario"] == scenario) & (df["density_veh_km_lane"] == density)]
    for pol, g in sub.groupby("policy"):
        x, y = float(g["tx_per_at_risk_informed"].mean()), float(g["tir_median_s"].mean())
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        st = reference_style() if pol == REFERENCE_POLICY else {
            "color": GREY, "marker": "o", "linestyle": "none", "markersize": 3.5}
        ax.plot([x], [y], **st)
        ax.annotate(label_for(pol).replace(" (reference)", ""), (x, y),
                    textcoords="offset points", xytext=(4, 3), fontsize=6, color=GREY)
    ax.set_xscale("log")
    ax.set_xlabel("Transmissions per at-risk vehicle informed")
    ax.set_ylabel("Median time to inform (s)")
    ax.set_title(f"{scenario.replace('_', ' ')}, {density:g} veh/km/lane", pad=3)
    return save(fig, name)


# ---------------------------------------------------------------------------
# Per-cell headroom: hindsight tuning vs one shipped baseline
# ---------------------------------------------------------------------------
def fig_headroom(
    cells, target: float = 0.95, axis: str = "cost", width: float = DOUBLE_COL,
    name: str = "fig02_headroom",
) -> list[Path]:
    """Per-cell oracle-best against the best single fixed baseline.

    The gap is what per-cell hindsight tuning buys, and the winner's name above
    each pair is the point: it changes from cell to cell, so no fixed scheme
    collects it.
    """
    import matplotlib.pyplot as plt

    from analysis.comparator import best_fixed_baseline, per_cell_oracle_best

    apply_ieee_style()
    oracle = per_cell_oracle_best(cells, target, axis)
    fixed = best_fixed_baseline(cells, target, axis)
    labels, o_costs, f_costs, winners = [], [], [], []
    for cell in cells:
        win, o_cost = oracle.per_cell[cell.key]
        labels.append(f"{cell.key.scenario.replace('_highway', '').replace('_', ' ')}\n"
                      f"d={cell.key.density:g}")
        o_costs.append(o_cost)
        f_costs.append(fixed.cost(cell.key))
        winners.append(label_for(win))

    fig, ax = plt.subplots(figsize=(width, width * 0.34))
    idx = np.arange(len(cells), dtype=float)
    ax.bar(idx - 0.19, o_costs, width=0.36, color=PALETTE[0], label="per-cell oracle-best")
    ax.bar(idx + 0.19, f_costs, width=0.36, color=PALETTE[1], hatch="//",
           edgecolor="white", linewidth=0.4, label=f"best fixed: {label_for(fixed.policy)}")
    for i, (o, w) in enumerate(zip(o_costs, winners)):
        if np.isfinite(o):
            ax.annotate(w, (i - 0.19, o), textcoords="offset points", xytext=(0, 2),
                        ha="center", fontsize=6, color=GREY)
    ax.set_xticks(idx, labels)
    ax.set_ylabel("Transmissions per at-risk vehicle informed")
    ax.set_title(f"Cost at RWCR >= {target:.0%} of each cell's ceiling", pad=3)
    ax.legend(loc="upper left")
    return save(fig, name)


# ---------------------------------------------------------------------------
# Figure 3: agent regret and margin, every cell
# ---------------------------------------------------------------------------
def fig_regret_margin(
    per_cell: dict[str, Any], width: float = DOUBLE_COL,
    name: str = "fig03_regret_margin",
) -> list[Path]:
    """Regret against the per-cell oracle-best, and margin over the best fixed
    baseline, one pair of bars per cell.

    Read straight from ``agent_evaluation.json`` (``axes.<axis>.per_cell``), the
    same artefact the tables read, so the figure cannot disagree with them.

    Cells the agent never matched at the quality target carry infinite regret.
    They are drawn as a marked gap, not clipped to a tall bar: a clipped
    infinity reads as a merely bad cell rather than as a failure to reach the
    target at all, which is the opposite of what happened.
    """
    import matplotlib.pyplot as plt

    apply_ieee_style()
    keys = list(per_cell)
    regret = np.array([float(per_cell[k].get("regret", np.nan)) for k in keys])
    margin = np.array([float(per_cell[k].get("margin", np.nan)) for k in keys])
    labels = [k.replace("_highway", "").replace("/clear", "").replace("_", " ")
              for k in keys]

    fig, axes = plt.subplots(2, 1, figsize=(width, width * 0.50), sharex=True)
    idx = np.arange(len(keys), dtype=float)

    for ax, vals, ylabel, note in (
        (axes[0], regret, "Regret vs oracle-best", "0 = matched hindsight tuning"),
        (axes[1], margin, "Margin vs best fixed", "positive = agent cheaper"),
    ):
        finite = np.isfinite(vals)
        wins = finite & (vals > 0) if ylabel.startswith("Margin") else np.zeros_like(finite)
        ax.bar(idx[finite & ~wins], vals[finite & ~wins], width=0.6, color=PALETTE[0],
               edgecolor="white", linewidth=0.4)
        if wins.any():                      # the one win, kept visible
            ax.bar(idx[wins], vals[wins], width=0.6, color=PALETTE[2], hatch="//",
                   edgecolor="white", linewidth=0.4)
        ax.axhline(0.0, color=GREY, linewidth=0.6)
        for i in np.flatnonzero(~finite):
            ax.annotate("never matched\nthe target", (i, 0.0), ha="center", va="bottom",
                        fontsize=5.5, color=GREY)
        ax.set_ylabel(ylabel)
        ax.annotate(note, xy=(0.99, 0.94), xycoords="axes fraction", ha="right",
                    va="top", fontsize=6, color=GREY)

    axes[1].set_xticks(idx, labels, rotation=18, ha="right")
    fig.align_ylabels(axes)
    return save(fig, name)


# ---------------------------------------------------------------------------
# Supporting figures (not in the paper's ten): metric vs density, error bands
# ---------------------------------------------------------------------------
def fig_metric_vs_density(
    df, metric: str, scenario: str, ylabel: str, name: str,
    width: float = SINGLE_COL, logy: bool = False,
    policies: Sequence[str] | None = None,
) -> list[Path]:
    """Mean +/- std across seeds, per policy, against density."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter, ScalarFormatter

    apply_ieee_style()
    fig, ax = plt.subplots(figsize=(width, width * 0.72))

    sub = df[df["scenario"] == scenario]
    present = [p for p in sub["policy"].unique()]
    panel = [p for p in policies if p in present] if policies else [
        p for p in FAMILIES["topology-aware"] if p in present
    ]

    for pol in panel:
        d = sub[sub["policy"] == pol]
        if d.empty or metric not in d:
            continue
        g = d.groupby("density_veh_km_lane")[metric]
        x = np.asarray(g.mean().index, dtype=float)
        mu = np.asarray(g.mean().values, dtype=float)
        sd = np.asarray(g.std().fillna(0).values, dtype=float)
        st = style_for(pol, panel)
        ax.plot(x, mu, label=label_for(pol), **st)
        # Error band rather than caps: less ink at this size, same information.
        ax.fill_between(x, mu - sd, mu + sd, color=st["color"], alpha=0.15, linewidth=0)

    ref = sub[sub["policy"] == REFERENCE_POLICY]
    if not ref.empty and metric in ref:
        g = ref.groupby("density_veh_km_lane")[metric]
        ax.plot(np.asarray(g.mean().index, dtype=float),
                np.asarray(g.mean().values, dtype=float),
                label=label_for(REFERENCE_POLICY), **{**reference_style(),
                                                      "linestyle": ":"})

    ax.set_xlabel("Density (veh/km/lane)")
    ax.set_ylabel(ylabel)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks(sorted(sub["density_veh_km_lane"].unique()))
    if logy:
        ax.set_yscale("log")
    ax.legend(loc="best", ncol=1)
    return save(fig, name)


# ---------------------------------------------------------------------------
# Figure 6: attention heatmap over a real dissemination event
# ---------------------------------------------------------------------------
def fig_attention_heatmap(
    attention: np.ndarray, neighbour_labels: Sequence[str], epoch_labels: Sequence[str],
    width: float = SINGLE_COL, name: str = "fig07_attention",
    xlabel: str = "Decision epoch", ylabel: str = "Neighbour",
) -> list[Path]:
    """Attention the holder placed on each neighbour, per decision epoch.

    Magnitude, so: single-hue sequential ramp, light to dark. The values shown
    are the same coefficients that selected the relay -- not a separate
    visualisation head.
    """
    import matplotlib.pyplot as plt

    apply_ieee_style()
    fig, ax = plt.subplots(figsize=(width, width * 0.72))
    im = ax.imshow(attention, aspect="auto", cmap=SEQUENTIAL_HUE,
                   vmin=0.0, vmax=float(np.nanmax(attention)) or 1.0)
    ax.set_xticks(range(len(epoch_labels)), epoch_labels, rotation=0)
    ax.set_yticks(range(len(neighbour_labels)), neighbour_labels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("Attention weight", rotation=90)
    cb.outline.set_linewidth(0.4)
    return save(fig, name)


# ---------------------------------------------------------------------------
# Figure 6: training curves -- shortfall and lambda per constraint group
# ---------------------------------------------------------------------------
def fig_training_constraint(
    history: Sequence[dict[str, Any]], width: float = DOUBLE_COL,
    name: str = "fig06_training", lambda_cap: float | None = None,
) -> list[Path]:
    """Per-group coverage shortfall and its Lagrange multiplier, over training.

    The constraint is enforced per (scenario, density) group, so a single
    pooled curve hides the result: the sparse groups plateau with their
    multiplier pinned at the cap while the dense groups converge. Groups are
    drawn in neutral ink with direct labels -- there are more of them than the
    validated colour slots, and cycling hues would invent distinctions between
    them that the reader would then try to interpret.
    """
    import matplotlib.pyplot as plt

    apply_ieee_style()
    rows = [r for r in history if r.get("shortfall_by_group")]
    if not rows:
        raise ValueError("history has no shortfall_by_group; nothing to plot")
    groups = sorted({g for r in rows for g in r["shortfall_by_group"]})
    x = np.array([int(r["update"]) for r in rows], dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(width, width * 0.52), sharex=True)
    for g in groups:
        sf = np.array([float(r["shortfall_by_group"].get(g, np.nan)) for r in rows])
        lam = np.array([float(r.get("lambda_by_group", {}).get(g, np.nan)) for r in rows])
        for ax, ys in ((axes[0], sf), (axes[1], lam)):
            ax.plot(x, ys, color=GREY, linewidth=0.7, alpha=0.85)
            fin = np.flatnonzero(np.isfinite(ys))
            if fin.size:
                ax.annotate(g.replace("|", " d="), (x[fin[-1]], ys[fin[-1]]),
                            textcoords="offset points", xytext=(2, 0),
                            fontsize=5.5, color=GREY, va="center")

    axes[0].axhline(0.0, color=PALETTE[1], linewidth=0.8, linestyle="--")
    axes[0].annotate("constraint met", xy=(0.01, 0.06), xycoords="axes fraction",
                     fontsize=6, color=PALETTE[1])
    axes[0].set_ylabel("Coverage shortfall")
    if lambda_cap:
        axes[1].axhline(float(lambda_cap), color=PALETTE[1], linewidth=0.8,
                        linestyle="--")
        axes[1].annotate(f"cap {float(lambda_cap):g}", xy=(0.01, 0.86),
                         xycoords="axes fraction", fontsize=6, color=PALETTE[1])
    axes[1].set_yscale("symlog")
    axes[1].set_ylabel(r"Multiplier $\lambda$")
    axes[1].set_xlabel("PPO update")
    fig.align_ylabels(axes)
    return save(fig, name)


# ---------------------------------------------------------------------------
# Supporting figure: pooled training reward, median with an inter-seed band
# ---------------------------------------------------------------------------
def fig_training_curves(
    histories: Sequence[Sequence[dict[str, Any]]], metric: str = "reward_mean",
    ylabel: str = "Mean reward", width: float = SINGLE_COL,
    name: str = "figS4_training_reward",
) -> list[Path]:
    """Median with an inter-seed band. Median, not mean: RL runs produce
    outliers and a mean curve would be dragged by one diverged seed."""
    import matplotlib.pyplot as plt

    apply_ieee_style()
    fig, ax = plt.subplots(figsize=(width, width * 0.72))

    n = min(len(h) for h in histories)
    arr = np.array([[h[i][metric] for i in range(n)] for h in histories], dtype=float)
    x = np.arange(1, n + 1)
    med = np.nanmedian(arr, axis=0)
    lo = np.nanpercentile(arr, 25, axis=0)
    hi = np.nanpercentile(arr, 75, axis=0)

    st = style_for("ai_harp", ["ai_harp"])
    ax.plot(x, med, color=st["color"], linestyle="-", marker="", label="median")
    ax.fill_between(x, lo, hi, color=st["color"], alpha=0.2, linewidth=0,
                    label="inter-quartile range")
    ax.set_xlabel("PPO update")
    ax.set_ylabel(ylabel)
    ax.legend(loc="best")
    return save(fig, name)


# ---------------------------------------------------------------------------
# Figure 8: ablation bar chart
# ---------------------------------------------------------------------------
def fig_ablation(
    labels: Sequence[str], means: Sequence[float], stds: Sequence[float],
    ylabel: str, baseline: float | None = None,
    width: float = DOUBLE_COL, name: str = "fig08_ablation",
) -> list[Path]:
    """Horizontal bars: the labels are long, and horizontal keeps them readable
    without rotation."""
    import matplotlib.pyplot as plt

    apply_ieee_style()
    fig, ax = plt.subplots(figsize=(width, width * 0.38))
    y = np.arange(len(labels))
    ax.barh(y, means, xerr=stds, height=0.62, color=PALETTE[0],
            edgecolor="white", linewidth=0.6, error_kw={"elinewidth": 0.7,
                                                        "ecolor": GREY})
    if baseline is not None:
        ax.axvline(baseline, color=GREY, linestyle="--", linewidth=0.8)
        ax.annotate("full model", (baseline, len(labels) - 0.35),
                    xytext=(3, 0), textcoords="offset points",
                    fontsize=6, color=GREY, va="center")
    ax.set_yticks(y, [s.replace("_", " ") for s in labels])
    ax.invert_yaxis()
    ax.set_xlabel(ylabel)
    ax.grid(axis="y", visible=False)
    return save(fig, name)


# ---------------------------------------------------------------------------
# Figure 5: confidence-gate trade-off
# ---------------------------------------------------------------------------
def fig_gate_tradeoff(
    taus: Sequence[float], fallback_rate: Sequence[float],
    performance: Sequence[float], perf_label: str = "RWCR",
    width: float = SINGLE_COL, name: str = "fig05_gate",
) -> list[Path]:
    """Fallback rate and performance against tau, as SMALL MULTIPLES.

    Deliberately not a dual-axis chart. Two measures on different scales share
    an x-axis and get one panel each; overlaying them on twin y-axes would let
    the reader infer a crossing point that is an artefact of the two scalings.
    """
    import matplotlib.pyplot as plt

    apply_ieee_style()
    fig, axes = plt.subplots(2, 1, figsize=(width, width * 0.95), sharex=True)

    axes[0].plot(taus, fallback_rate, color=PALETTE[0], marker=MARKERS[0],
                 linestyle=LINESTYLES[0])
    axes[0].set_ylabel("Fallback rate")
    axes[0].set_ylim(-0.02, 1.02)

    axes[1].plot(taus, performance, color=PALETTE[1], marker=MARKERS[1],
                 linestyle=LINESTYLES[1])
    axes[1].set_ylabel(perf_label)
    axes[1].set_xlabel(r"Confidence threshold $\tau$")
    fig.align_ylabels(axes)
    return save(fig, name)


# ---------------------------------------------------------------------------
# Figure 10: simulator validation against a published curve
# ---------------------------------------------------------------------------
def fig_validation(
    x: Sequence[float], ours: Sequence[float], published: Sequence[float],
    xlabel: str, ylabel: str, source: str,
    width: float = SINGLE_COL, name: str = "fig10_validation",
) -> list[Path]:
    """Ours against a published NS-3/Veins curve, with the deviation stated.

    The point of this figure is to make the deviation visible rather than to
    hide it, so the residual is annotated numerically.
    """
    import matplotlib.pyplot as plt

    apply_ieee_style()
    fig, ax = plt.subplots(figsize=(width, width * 0.72))
    ax.plot(x, published, color=GREY, marker=MARKERS[1], linestyle=LINESTYLES[1],
            label=source)
    ax.plot(x, ours, color=PALETTE[0], marker=MARKERS[0], linestyle=LINESTYLES[0],
            label="this simulator")

    resid = np.asarray(ours, dtype=float) - np.asarray(published, dtype=float)
    rmse = float(np.sqrt(np.nanmean(resid ** 2)))
    ax.annotate(f"RMSE {rmse:.3f}", xy=(0.97, 0.95), xycoords="axes fraction",
                ha="right", va="top", fontsize=6, color=GREY)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="best")
    return save(fig, name)
