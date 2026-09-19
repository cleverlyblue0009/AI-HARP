"""The attention capture behind figure 7, and the two figures added for the
campaign (regret/margin, per-group training curves).

The capture is opt-in on purpose: training's batched inference runs it on every
decision, so the default path must not pay for a figure. These tests pin that,
and pin the two ways the figures were wrong before they were fixed -- a
single-column heatmap, and infinite regret drawn as a finite bar.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("matplotlib")

from analysis import figures as F           # noqa: E402
from analysis.report import _attention_matrix   # noqa: E402


def _decision(step: int, holder: int, attention: list[float],
              used_fallback: bool = False) -> dict:
    return {
        "step": step, "holder": holder,
        "neighbours": list(range(100, 100 + len(attention))),
        "attention": attention, "action": "broadcast_now",
        "used_fallback": used_fallback, "confidence": 0.6,
    }


# ===================================================== opt-in relay scores ===
def test_act_returns_relay_scores_only_when_asked():
    """Attention leaves the network only for the figure.

    ``act`` copies a tensor out per decision when asked; training never asks,
    and the ranking it does use is computed inside the network anyway.
    """
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from agents.gat_drl import ActorCritic, EncoderConfig
    from agents.graph import build_decision_graph
    from tests.test_policies import make_ctx

    net = ActorCritic(EncoderConfig())
    graph = build_decision_graph(make_ctx()).to_pyg()

    assert "relay_scores" not in net.act(graph)
    out = net.act(graph, return_scores=True)
    assert "relay_scores" in out
    assert np.asarray(out["relay_scores"]).shape[0] == graph.num_nodes
    # The ranking must not change just because the scores were also returned.
    assert net.act(graph, deterministic=True)["relay_order"] == \
        net.act(graph, deterministic=True, return_scores=True)["relay_order"]


def test_act_batch_returns_one_score_vector_per_graph():
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from torch_geometric.data import Batch

    from agents.gat_drl import ActorCritic, EncoderConfig
    from agents.graph import build_decision_graph
    from tests.test_policies import make_ctx

    net = ActorCritic(EncoderConfig())
    graphs = [build_decision_graph(make_ctx()).to_pyg() for _ in range(3)]
    batch = Batch.from_data_list(graphs)

    rows = net.act_batch(batch, return_scores=True)
    assert len(rows) == 3
    for row, graph in zip(rows, graphs):
        assert np.asarray(row["relay_scores"]).shape[0] == graph.num_nodes
    assert all("relay_scores" not in r for r in net.act_batch(batch))


# ========================================================= figure 7 matrix ===
def test_attention_matrix_excludes_gate_fallbacks():
    """On a fallback the analytic policy chose and attention selected nothing.

    Plotting those columns would credit the network with decisions it did not
    make.
    """
    payload = {"decisions": [
        _decision(1, 10, [0.5, 0.3, 0.2]),
        _decision(2, 11, [0.9, 0.1, 0.0], used_fallback=True),
        _decision(3, 12, [0.4, 0.4, 0.2]),
    ]}
    matrix, rows, cols, n = _attention_matrix(payload)
    assert n == 2 and matrix.shape[1] == 2
    assert cols == ["t1", "t3"]


def test_one_decision_per_holder_still_gives_a_readable_heatmap():
    """The first version keyed columns on ONE holder's decision epochs.

    In a real event each holder decides about once, so that produced a
    12x1 heatmap -- a single stripe. Columns are decisions across the event.
    """
    payload = {"decisions": [_decision(s, 100 + s, [0.4, 0.3, 0.2, 0.1])
                             for s in range(1, 9)]}
    matrix, rows, cols, _ = _attention_matrix(payload)
    assert matrix.shape[1] == 8, "columns must be the event's decisions"
    assert matrix.shape[0] == 4 and rows[0] == "rank 1"


def test_attention_matrix_ranks_within_each_decision():
    """Rows are ranks, so every column is sorted descending.

    Vehicle id cannot be the row key: each decision has a different neighbour
    set, so such a row would be empty almost everywhere.
    """
    payload = {"decisions": [_decision(1, 10, [0.1, 0.7, 0.2]),
                             _decision(2, 11, [0.6, 0.1, 0.3])]}
    matrix, _, _, _ = _attention_matrix(payload)
    for column in matrix.T:
        finite = column[np.isfinite(column)]
        assert np.all(np.diff(finite) <= 0)
    assert matrix[0, 0] == pytest.approx(0.7)


def test_attention_matrix_is_none_without_network_decisions():
    assert _attention_matrix({"decisions": []}) is None
    assert _attention_matrix(
        {"decisions": [_decision(1, 10, [0.5], used_fallback=True)]}) is None


# =============================================== figures 3 and 6 (campaign) ==
def _capture(monkeypatch, tmp_path):
    """Run a figure and hand back the Figure object it saved."""
    saved = {}
    original = F.save

    def spy(fig, name, out_dir=None):
        saved["fig"] = fig
        return original(fig, name, out_dir=tmp_path)

    monkeypatch.setattr(F, "save", spy)
    return saved


def test_regret_margin_draws_a_gap_for_cells_that_never_matched(monkeypatch, tmp_path):
    """Infinite regret must not become a tall bar.

    Clipped, it reads as a merely bad cell instead of a failure to reach the
    quality target at all -- the opposite of what happened.
    """
    saved = _capture(monkeypatch, tmp_path)
    per_cell = {
        "rural/d=2": {"regret": float("inf"), "margin": float("-inf")},
        "rural/d=20": {"regret": 0.018, "margin": 0.019},
        "urban/d=20": {"regret": float("inf"), "margin": float("-inf")},
    }
    paths = F.fig_regret_margin(per_cell, name="t_regret")
    assert all(p.exists() for p in paths)

    regret_axis = saved["fig"].axes[0]
    drawn = [b for b in regret_axis.patches if np.isfinite(b.get_height())]
    assert len(drawn) == 1, "only the one matched cell gets a bar"
    texts = " ".join(t.get_text() for t in regret_axis.texts)
    assert "never matched" in texts


def test_training_figure_plots_every_constraint_group(monkeypatch, tmp_path):
    saved = _capture(monkeypatch, tmp_path)
    history = [{
        "update": u,
        "shortfall_by_group": {"rural_highway|20": -0.05 + 0.001 * u,
                               "urban_nlos|2": 0.30},
        "lambda_by_group": {"rural_highway|20": 2.0, "urban_nlos|2": min(500.0, u * 10)},
    } for u in range(1, 21)]

    paths = F.fig_training_constraint(history, name="t_train", lambda_cap=500.0)
    assert all(p.exists() for p in paths)
    shortfall_axis, lambda_axis = saved["fig"].axes[:2]
    # one line per group, plus the reference line on each panel
    assert len(shortfall_axis.lines) == 3 and len(lambda_axis.lines) == 3
    assert "cap 500" in " ".join(t.get_text() for t in lambda_axis.texts)


def test_training_figure_refuses_history_without_groups():
    """A pooled history cannot answer the question this figure asks."""
    with pytest.raises(ValueError, match="shortfall_by_group"):
        F.fig_training_constraint([{"update": 1, "reward_mean": 0.1}], name="t_none")
