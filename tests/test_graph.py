"""Phase 5.1 tests: decision-graph construction (NumPy, no torch required)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from agents.graph import (
    EDGE_FEATURES,
    N_EDGE_FEATURES,
    N_NODE_FEATURES,
    NODE_FEATURES,
    FeatureNormaliser,
    GraphConfig,
    build_decision_graph,
)
from tests.test_policies import empty_ctx, make_ctx


def test_holder_is_node_zero_and_is_self_relative():
    g = build_decision_graph(make_ctx())
    assert g.holder_id == 3
    assert np.allclose(g.x[0, :4], 0.0), "holder must be at the origin of its own frame"


def test_node_count_is_holder_plus_neighbours():
    g = build_decision_graph(make_ctx())
    assert g.n_nodes == 1 + 3


def test_neighbour_cap_is_respected():
    n = 40
    ctx = make_ctx(
        neighbours=np.arange(100, 100 + n),
        neighbour_distances=np.linspace(10, 500, n),
        neighbour_dx=np.linspace(10, 500, n), neighbour_dy=np.zeros(n),
        neighbour_vx=np.full(n, 20.0), neighbour_vy=np.zeros(n),
        neighbour_relevance=np.linspace(0, 1, n),
        neighbour_informed=np.zeros(n, dtype=bool),
    )
    g = build_decision_graph(ctx, GraphConfig(neighbour_cap_k=12))
    assert g.n_nodes == 13
    assert g.neighbour_ids.size == 12


def test_cap_keeps_the_nearest_neighbours():
    n = 20
    dist = np.linspace(500, 10, n)          # deliberately descending
    ctx = make_ctx(
        neighbours=np.arange(n), neighbour_distances=dist,
        neighbour_dx=dist, neighbour_dy=np.zeros(n),
        neighbour_vx=np.full(n, 20.0), neighbour_vy=np.zeros(n),
        neighbour_relevance=np.zeros(n), neighbour_informed=np.zeros(n, dtype=bool),
    )
    g = build_decision_graph(ctx, GraphConfig(neighbour_cap_k=5))
    assert set(g.neighbour_ids.tolist()) == set(np.argsort(dist)[:5].tolist())


def test_empty_neighbourhood_yields_a_single_node():
    g = build_decision_graph(empty_ctx())
    assert g.n_nodes == 1
    assert g.n_edges == 0
    assert g.relay_target(0) is None


def test_feature_widths_match_the_declared_schema():
    g = build_decision_graph(make_ctx())
    assert g.x.shape[1] == N_NODE_FEATURES == len(NODE_FEATURES)
    assert g.edge_attr.shape[1] == N_EDGE_FEATURES == len(EDGE_FEATURES)


def test_edges_are_bidirectional_between_holder_and_neighbours():
    g = build_decision_graph(make_ctx(), GraphConfig(include_neighbour_edges=False))
    pairs = set(zip(g.edge_index[0].tolist(), g.edge_index[1].tolist()))
    for j in range(1, g.n_nodes):
        assert (j, 0) in pairs and (0, j) in pairs


def test_neighbour_edges_give_the_graph_real_structure():
    """Without neighbour-to-neighbour edges the graph is a star and a GAT
    degenerates to attention pooling, which would make the GAT-vs-MLP ablation
    vacuous: there would be no structure for the GAT to exploit."""
    star = build_decision_graph(make_ctx(), GraphConfig(include_neighbour_edges=False))
    full = build_decision_graph(make_ctx(), GraphConfig(include_neighbour_edges=True))
    assert full.n_edges > star.n_edges


def test_edge_attributes_are_finite_and_distance_is_non_negative():
    g = build_decision_graph(make_ctx())
    assert np.all(np.isfinite(g.edge_attr))
    assert np.all(g.edge_attr[:, 0] >= 0)


def test_link_lifetime_is_bounded():
    """A pair with zero closing rate has infinite lifetime, which is useless as
    a feature and would poison normalisation."""
    lifetime = build_decision_graph(make_ctx()).edge_attr[:, 2]
    assert np.all(np.isfinite(lifetime))
    assert np.all((lifetime >= 0) & (lifetime <= 60.0))


def test_predicted_rssi_decreases_with_distance():
    g = build_decision_graph(make_ctx())
    order = np.argsort(g.edge_attr[:, 0])
    assert g.edge_attr[order, 3][0] >= g.edge_attr[order, 3][-1]


def test_relay_target_maps_back_to_a_real_vehicle():
    g = build_decision_graph(make_ctx())
    # Neighbours are ordered nearest-first: 1 (100 m), 2 (200 m), 5 (450 m).
    assert g.relay_target(0) == 1
    assert g.relay_target(2) == 5
    assert g.relay_target(99) is None


def test_graph_uses_only_causal_relevance():
    """The holder's relevance feature must be the context's causal estimate."""
    g = build_decision_graph(make_ctx(relevance=0.42))
    assert g.x[0, NODE_FEATURES.index("relevance_causal")] == pytest.approx(0.42)


def test_informed_flag_reflects_only_what_was_overheard():
    g = build_decision_graph(make_ctx())
    col = NODE_FEATURES.index("is_informed")
    assert g.x[0, col] == 1.0                        # the holder holds the message
    assert g.x[1:, col].tolist() == [0.0, 1.0, 0.0]  # matches neighbour_informed order


def test_ttl_remaining_shrinks_with_message_age():
    col = NODE_FEATURES.index("ttl_remaining_s")
    young = build_decision_graph(make_ctx(age_s=1.0), GraphConfig(ttl_s=60.0))
    old = build_decision_graph(make_ctx(age_s=50.0), GraphConfig(ttl_s=60.0))
    assert young.x[0, col] > old.x[0, col]
    assert old.x[0, col] == pytest.approx(10.0)


def test_ttl_remaining_never_goes_negative():
    g = build_decision_graph(make_ctx(age_s=120.0), GraphConfig(ttl_s=60.0))
    assert g.x[0, NODE_FEATURES.index("ttl_remaining_s")] == 0.0


def test_graph_config_reads_from_yaml():
    from common.config import load_yaml

    assert GraphConfig.from_config(load_yaml("agent.yaml")).neighbour_cap_k == 12


def test_declared_features_match_the_config_file():
    """The config lists the feature names for the paper; they must not drift
    from the array layout the model actually reads."""
    from common.config import load_yaml

    g = load_yaml("agent.yaml")["graph"]
    assert tuple(g["node_features"]) == NODE_FEATURES
    assert tuple(g["edge_features"]) == EDGE_FEATURES


# ------------------------------------------------------------ normalisation --
def test_normaliser_is_frozen_not_per_batch():
    graphs = [build_decision_graph(make_ctx(relevance=float(r)))
              for r in np.linspace(0, 1, 20)]
    norm = FeatureNormaliser.fit(graphs)
    assert np.allclose(norm.apply(graphs[0]).x, norm.apply(graphs[0]).x)
    before = norm.node_mean.copy()
    for g in [build_decision_graph(make_ctx(relevance=0.9)) for _ in range(5)]:
        norm.apply(g)
    assert np.allclose(norm.node_mean, before), "applying must never refit"


def test_normalised_features_are_roughly_standardised():
    graphs = [build_decision_graph(make_ctx(relevance=float(r), age_s=float(a)))
              for r in np.linspace(0, 1, 10) for a in np.linspace(0, 20, 10)]
    norm = FeatureNormaliser.fit(graphs)
    xs = np.concatenate([norm.apply(g).x for g in graphs], axis=0)
    varying = np.asarray(norm.node_std) > 1e-3
    assert np.all(np.abs(xs[:, varying].mean(axis=0)) < 0.2)


def test_normaliser_roundtrips_through_json(tmp_path):
    norm = FeatureNormaliser.fit([build_decision_graph(make_ctx()) for _ in range(4)])
    back = FeatureNormaliser.load(norm.save(tmp_path / "stats.json"))
    assert np.allclose(back.node_mean, norm.node_mean)
    assert np.allclose(back.edge_std, norm.edge_std)


def test_normaliser_rejects_a_changed_feature_schema(tmp_path):
    """Silently reading the wrong columns is the worst possible failure."""
    p = FeatureNormaliser.fit(
        [build_decision_graph(make_ctx()) for _ in range(3)]
    ).save(tmp_path / "stats.json")
    d = json.loads(p.read_text())
    d["node_features"] = ["something", "else"]
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="different node-feature set"):
        FeatureNormaliser.load(p)


def test_normaliser_never_divides_by_zero():
    """Constant features must survive normalisation rather than become inf."""
    graphs = [build_decision_graph(make_ctx()) for _ in range(5)]
    out = FeatureNormaliser.fit(graphs).apply(graphs[0])
    assert np.all(np.isfinite(out.x))
    assert np.all(np.isfinite(out.edge_attr))
