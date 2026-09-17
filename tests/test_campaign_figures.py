"""Campaign figures that need no agent data: estimation agreement, latency/cost, headroom."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("matplotlib")
pd = pytest.importorskip("pandas")

from analysis import figures as F  # noqa: E402


def _risk_frame():
    rows = []
    for sc, base in (("rural_highway", 0.75), ("urban_grid", 0.17), ("urban_nlos", 0.19)):
        for d in (1.0, 2.0, 10.0, 80.0):
            for seed in range(3):
                rows.append({"scenario": sc, "density": d,
                             "risk_peak_corr": base + 0.01 * seed - (0.2 if d == 80 else 0.0)})
    return pd.DataFrame(rows)


def _runs_frame():
    rng = np.random.default_rng(0)
    pols = ["flooding", "p_persistence_03", "p_persistence_05", "p_persistence_07", "slotted_1p",
            "weighted_p", "counter_based", "greedy_farthest", "dvcast"]
    rows = []
    for i, p in enumerate(pols):
        for seed in range(3):
            rows.append({"scenario": "rural_highway", "density_veh_km_lane": 20.0, "policy": p,
                         "tx_per_at_risk_informed": (2.5 if p == "flooding" else 0.6 + 0.05 * i)
                         + 0.01 * rng.random(),
                         "tir_median_s": (0.2 if p == "flooding" else 0.4 + 0.03 * i)})
    return pd.DataFrame(rows)


def test_estimation_agreement_figure(tmp_path):
    paths = F.fig_estimation_agreement(_risk_frame(), name="t_est", out_dir=tmp_path) \
        if "out_dir" in F.fig_estimation_agreement.__code__.co_varnames else None
    if paths is None:                      # module writes to its own FIG_DIR
        import analysis.figures as mod
        mod.FIG_DIR = tmp_path
        paths = F.fig_estimation_agreement(_risk_frame(), name="t_est")
    assert {p.suffix for p in paths} == {".pdf", ".png"} and all(p.exists() for p in paths)
    png = next(p for p in paths if p.suffix == ".png")
    lum = F.greyscale_check(png)
    assert lum["luminance_p95"] - lum["luminance_p05"] > 0.1     # not a flat block of ink


def test_latency_cost_figure_handles_more_policies_than_colour_slots(tmp_path):
    import analysis.figures as mod

    mod.FIG_DIR = tmp_path
    df = _runs_frame()
    assert df["policy"].nunique() > F.PANEL_SLOTS
    paths = F.fig_latency_cost_inversion(df, "rural_highway", 20.0, name="t_inv")
    assert all(p.exists() for p in paths)


def test_headroom_figure_from_committed_cells(tmp_path):
    import analysis.figures as mod
    from analysis.comparator import load_cells

    mod.FIG_DIR = tmp_path
    cells = load_cells()
    paths = F.fig_headroom(cells, name="t_head")
    assert all(p.exists() for p in paths)
