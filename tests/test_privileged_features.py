"""Revision-3 R8: the generic extra-node-feature hook and the privileged reward.

The hook lives in decision-time code but carries no oracle knowledge; what it must
guarantee is that (1) it is inert unless configured, (2) a configured column that is
not supplied is an error, never a silent zero, and (3) normalisation leaves the extra
column alone.
"""

from __future__ import annotations

import numpy as np
import pytest

from agents.graph import N_NODE_FEATURES, FeatureNormaliser, GraphConfig, build_decision_graph
from agents.base import DecisionContext, Trigger


def _ctx(extras):
    nb = np.array([1, 2])
    return DecisionContext(
        step=5, time_s=0.5, dt=0.1, index=0, trigger=Trigger.RECEIVE, hop_count=1,
        duplicate_count=0, own_tx_count=0, sender_index=1, sender_distance_m=100.0,
        max_sender_distance_m=100.0, age_s=0.5, x=0.0, y=0.0, speed_ms=20.0, heading=0.0,
        direction=0, relevance=0.4, eta_s=10.0, vclass="car", neighbours=nb,
        neighbour_distances=np.array([100.0, 200.0]), neighbour_dx=np.array([100.0, -200.0]),
        neighbour_dy=np.zeros(2), neighbour_vx=np.array([20.0, 25.0]), neighbour_vy=np.zeros(2),
        neighbour_relevance=np.array([0.5, 0.1]), neighbour_informed=np.array([True, False]),
        comm_range_m=600.0, rng=np.random.default_rng(0), extras=extras)


def test_extra_features_are_inert_unless_configured():
    g = build_decision_graph(_ctx({"extra_relevance_oracle": np.array([0.9, 0.8, 0.7])}), GraphConfig())
    assert g.x.shape[1] == N_NODE_FEATURES


def test_configured_extra_feature_is_appended_holder_first():
    cfg = GraphConfig(extra_node_features=("extra_relevance_oracle",))
    g = build_decision_graph(_ctx({"extra_relevance_oracle": np.array([0.9, 0.8, 0.7])}), cfg)
    assert g.x.shape[1] == N_NODE_FEATURES + 1
    assert g.x[0, -1] == pytest.approx(0.9)          # holder (vehicle 0)
    assert sorted(g.x[1:, -1].tolist()) == pytest.approx([0.7, 0.8])


def test_missing_extra_feature_is_an_error_not_a_zero():
    cfg = GraphConfig(extra_node_features=("extra_relevance_oracle",))
    with pytest.raises(KeyError):
        build_decision_graph(_ctx({}), cfg)


def test_extra_feature_names_must_be_marked():
    with pytest.raises(ValueError):
        GraphConfig(extra_node_features=("relevance_oracle",))


def test_normaliser_leaves_extra_columns_untouched():
    cfg = GraphConfig(extra_node_features=("extra_relevance_oracle",))
    g = build_decision_graph(_ctx({"extra_relevance_oracle": np.array([0.9, 0.8, 0.7])}), cfg)
    n = N_NODE_FEATURES
    norm = FeatureNormaliser(node_mean=np.ones(n), node_std=np.full(n, 2.0),
                             edge_mean=np.zeros(g.edge_attr.shape[1]), edge_std=np.ones(g.edge_attr.shape[1]),
                             n_samples=1)
    out = norm.apply(g)
    assert np.allclose(out.x[:, :n], (g.x[:, :n] - 1.0) / 2.0)
    assert np.allclose(out.x[:, n:], g.x[:, n:])
