"""Graph switches used by the architecture ablations.

The star-graph ablation checks the README's claim that without
neighbour-to-neighbour edges GATv2 degenerates to attention pooling; the
dropped-feature ablation asks whether the policy uses causal relevance at all.
"""

from __future__ import annotations

import numpy as np
import pytest

from agents.graph import NODE_FEATURES, GraphConfig, build_decision_graph
from common.config import load_yaml
from tests.test_policies import make_ctx


def _ctx():
    return make_ctx()


def test_star_graph_has_only_holder_edges():
    ctx = _ctx()
    full = build_decision_graph(ctx, GraphConfig())
    star = build_decision_graph(ctx, GraphConfig(include_neighbour_edges=False))
    k = star.meta["k"]
    assert star.n_edges == 2 * k
    assert np.all((star.edge_index[0] == 0) | (star.edge_index[1] == 0))
    assert full.n_edges >= star.n_edges
    assert np.array_equal(full.x, star.x)


def test_dropped_feature_is_zero_for_every_node_and_nothing_else_changes():
    ctx = _ctx()
    col = NODE_FEATURES.index("relevance_causal")
    full = build_decision_graph(ctx, GraphConfig())
    dropped = build_decision_graph(ctx, GraphConfig(drop_node_features=("relevance_causal",)))
    assert np.all(dropped.x[:, col] == 0.0)
    others = [i for i in range(len(NODE_FEATURES)) if i != col]
    assert np.array_equal(full.x[:, others], dropped.x[:, others])
    assert np.array_equal(full.edge_index, dropped.edge_index)
    assert np.array_equal(full.edge_attr, dropped.edge_attr)


def test_unknown_dropped_feature_is_refused():
    with pytest.raises(ValueError):
        GraphConfig(drop_node_features=("relevance_oracle",))


def test_graph_config_reads_ablation_switches():
    cfg = load_yaml("agent.yaml")
    default = GraphConfig.from_config(cfg)
    assert default.include_neighbour_edges and default.drop_node_features == ()
    cfg["graph"].update(include_neighbour_edges=False, drop_node_features=["relevance_causal"],
                        neighbour_cap_k=4)
    g = GraphConfig.from_config(cfg)
    assert not g.include_neighbour_edges
    assert g.drop_node_features == ("relevance_causal",)
    assert g.neighbour_cap_k == 4
