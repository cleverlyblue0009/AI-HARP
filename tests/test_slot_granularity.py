"""Slot-granularity sensitivity sweep."""

from __future__ import annotations

import csv

from experiments.slot_granularity import FIELDS, SLOT_EPOCHS, run, run_one, summarise, tasks_for

CELL = [{"scenario": "rural_highway", "density": 2, "weather": "clear", "hazard_type": "fog_bank"}]


def test_task_grid_covers_policies_epochs_and_seeds():
    t = tasks_for(CELL, seeds=3)
    assert len(t) == 2 * len(SLOT_EPOCHS) * 3
    assert {x[5] for x in t} == set(SLOT_EPOCHS)


def test_longer_slots_raise_latency_for_a_slotted_scheme():
    fast = run_one(("rural_highway", 2.0, "clear", "fog_bank", "slotted_1p", 1, 0))
    slow = run_one(("rural_highway", 2.0, "clear", "fog_bank", "slotted_1p", 10, 0))
    assert fast["slot_epochs"] == 1 and slow["slot_epochs"] == 10
    assert slow["tir_median_s"] > fast["tir_median_s"]


def test_run_writes_and_resumes(tmp_path):
    out = tmp_path / "slots.csv"
    tasks = tasks_for(CELL, seeds=1, epochs=(1, 2), policies=("slotted_1p",))
    assert run(out, tasks) == 2
    with out.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 2 and set(rows[0]) == set(FIELDS)
    assert run(out, tasks) == 0
    assert "slot_epochs" in summarise(rows)
