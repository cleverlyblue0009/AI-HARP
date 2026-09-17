"""Ablation evaluation driver: which runs it picks up, and what it skips."""

from __future__ import annotations

import json

import experiments.evaluate_ablations as ea


def _run(tmp_path, name, trained=True, ckpt=True):
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    if trained:
        (d / "run_summary.json").write_text(json.dumps({"stopped": "max_updates"}), encoding="utf-8")
    if ckpt:
        (d / "ckpt_final.pt").write_bytes(b"x")
    return d


def test_only_finished_runs_are_listed_in_queue_order(tmp_path, monkeypatch):
    from experiments.campaign_train import QUEUE

    monkeypatch.setattr(ea, "CAMPAIGN", tmp_path)
    _run(tmp_path, "gcn")
    _run(tmp_path, "ref")
    _run(tmp_path, "mlp", trained=False)          # still training
    _run(tmp_path, "star", ckpt=False)            # summary but no checkpoint
    got = ea.finished_runs()
    assert got == [n for n in QUEUE if n in ("ref", "gcn")]
    assert "mlp" not in got and "star" not in got


def test_already_scored_runs_are_skipped(tmp_path, monkeypatch):
    monkeypatch.setattr(ea, "CAMPAIGN", tmp_path)
    scored = tmp_path / "out" / "agent_gcn" / "sampled"
    scored.mkdir(parents=True)
    monkeypatch.setattr(ea, "out_dir", lambda name, mode: tmp_path / "out" / f"agent_{name}" / mode)
    assert not ea.is_scored("gcn", "sampled")
    (scored / "agent_evaluation.json").write_text("{}", encoding="utf-8")
    assert ea.is_scored("gcn", "sampled")
    assert not ea.is_scored("gcn", "argmax")

    calls = []
    monkeypatch.setattr(ea, "evaluate", lambda *a, **k: calls.append(a) or 0)
    _run(tmp_path, "gcn")
    assert ea.main(["--only", "gcn"]) == 0 and calls == []          # skipped
    assert ea.main(["--only", "gcn", "--force"]) == 0 and len(calls) == 1


def test_unfinished_run_is_not_evaluated(tmp_path, monkeypatch):
    monkeypatch.setattr(ea, "CAMPAIGN", tmp_path)
    monkeypatch.setattr(ea, "out_dir", lambda name, mode: tmp_path / "out" / f"agent_{name}" / mode)
    calls = []
    monkeypatch.setattr(ea, "evaluate", lambda *a, **k: calls.append(a) or 0)
    _run(tmp_path, "mlp", trained=False, ckpt=False)
    assert ea.main(["--only", "mlp"]) == 0 and calls == []
