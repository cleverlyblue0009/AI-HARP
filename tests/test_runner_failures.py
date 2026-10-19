"""A failing task is logged and skipped; the rest of the batch is still written (revision 5)."""

from __future__ import annotations

import pytest

import experiments.revision1.runner as R


@pytest.mark.parametrize("jobs", [1, 2])
def test_failed_task_is_skipped_and_others_written(tmp_path, capsys, jobs):
    ok = R.task("rural_highway", 2.0, "flooding", 400, backend="fallback")
    bad = dict(R.task("rural_highway", 2.0, "slotted_1p", 400, param="n_slots", value=5, backend="fallback"))
    bad["policy"] = "no_such_policy"
    bad["task_hash"] = "deadbeef0000"
    path = tmp_path / "rows.csv"
    R.run_tasks([bad, ok], path, jobs=jobs, log_every=10)
    rows = R.load_rows(path)
    assert [r["task_hash"] for r in rows] == [ok["task_hash"]]
    out = capsys.readouterr().out
    assert "FAILED deadbeef0000" in out and "will be retried" in out
