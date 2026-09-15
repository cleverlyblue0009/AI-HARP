"""Policy-independent risk-estimation agreement sweep."""

from __future__ import annotations

import csv

import numpy as np

from experiments.risk_estimation import FIELDS, agreement_row, run


def test_one_corridor_cell_has_bounded_agreement_and_labels():
    row = agreement_row(("rural_highway", 2.0, "clear", "fog_bank", 0))
    assert row["topology_split"] == "train" and row["hazard_split"] == "train"
    for k in ("risk_est_precision", "risk_est_recall", "risk_est_f1"):
        assert np.isnan(row[k]) or 0.0 <= row[k] <= 1.0
    assert np.isnan(row["risk_peak_corr"]) or -1.0 <= row["risk_peak_corr"] <= 1.0
    assert row["n_at_risk_oracle"] >= 0 and row["density_veh_km_lane"] == 2.0


def test_run_writes_and_resumes(tmp_path):
    out = tmp_path / "risk.csv"
    tasks = [("rural_highway", 2.0, "clear", "fog_bank", s) for s in (0, 1)]
    assert run(out, tasks) == 2
    with out.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 2 and set(rows[0]) == set(FIELDS)
    assert run(out, tasks) == 0
