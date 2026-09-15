"""The runs.csv sweep: grid, split labels, pairing and resume."""

from __future__ import annotations

import csv

from common.config import load_yaml
from experiments.full_sweep import FIELDS, grid, row_key, splits, sweep


def test_grid_size_and_agent_arms():
    base = grid(["flooding", "dvcast"], [0.0, 0.5], range(10), None, "")
    assert len(base) == 3 * 8 * 4 * 5 * 2 * 10
    with_agent = grid(["flooding"], [0.0, 0.5], range(2), "ckpt.pt", "abc")
    agent = [t for t in with_agent if t["policy"] == "ai_harp"]
    assert len(agent) == 2 * len(with_agent) // 3
    assert {t["tau"] for t in agent} == {"0", "0.5"}
    assert all(t["checkpoint_sha"] == "abc" for t in agent)
    assert all(t["checkpoint_sha"] == "" for t in with_agent if t["policy"] != "ai_harp")


def test_grid_runs_cheap_densities_first():
    ds = [t["density"] for t in grid(["flooding"], [], range(1), None, "")]
    assert ds == sorted(ds)


def test_split_labels_follow_the_training_config():
    t = load_yaml("agent.yaml")["training"]
    assert splits("rural_highway", "fog_bank", "clear", t) == {
        "topology_split": "train", "hazard_split": "train", "weather_split": "train"}
    assert splits("urban_grid", "black_ice", "dense_fog", t) == {
        "topology_split": "held_out", "hazard_split": "held_out", "weather_split": "held_out"}
    assert splits("urban_nlos", "waterlogging", "moderate_rain", t)["hazard_split"] == "held_out"


def test_argmax_mode_refuses_to_write_into_runs_csv():
    import pytest

    from experiments.full_sweep import main

    with pytest.raises(SystemExit):
        main(["--policy-mode", "argmax", "--no-baselines", "--checkpoint", "x.pt"])


def test_sweep_writes_paired_rows_and_resumes(tmp_path):
    out = tmp_path / "runs.csv"
    tasks = grid(["flooding", "slotted_1p"], [], range(2), None, "",
                 scenarios=["rural_highway"], densities=[2.0], weathers=["clear"],
                 hazards=["fog_bank"])
    assert sweep(out, tasks, None, jobs=1, duration_s=20.0) == 4
    with out.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 4 and set(rows[0]) == set(FIELDS)
    assert {r["seed"] for r in rows} == {"0", "1"}
    assert all(r["topology_split"] == "train" and r["metrics_version"] for r in rows)
    # analysis/report.py and tables.py select on this column name
    assert all(float(r["density_veh_km_lane"]) == float(r["density"]) == 2.0 for r in rows)
    assert {row_key(r) for r in rows} == {row_key(t) for t in tasks}
    assert sweep(out, tasks, None, jobs=1, duration_s=20.0) == 0      # resumed: nothing to do
