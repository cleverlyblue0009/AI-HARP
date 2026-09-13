"""Phase 5.1: decision-graph construction.

One graph per message-holder per 100 ms decision epoch. Node 0 is always the
holder; nodes 1..K are its k nearest reachable neighbours (and any RSU in
range), capped at ``neighbour_cap_k`` so batches have bounded size.

Deliberately NumPy-only
-----------------------
Nothing here imports torch. Graph construction is where feature leakage and
normalisation bugs actually happen, so it is built and tested independently of
the learning stack -- and it stays testable on a machine with no torch
installed. :meth:`DecisionGraph.to_pyg` converts on demand.

Causal features only
--------------------
Every feature comes from the :class:`~agents.base.DecisionContext`, which
exposes only what a vehicle can observe at time *t*: its own state, what it has
overheard, and neighbour position/velocity/heading from CAM/BSM beacons. The
relevance and ETA fields are the **causal** estimates from
``hazard/risk_field.py``. The oracle field is ground truth computed from
realised trajectories and must never appear here;
``tests/test_oracle_isolation.py`` enforces that by parsing this module's
imports.

Frozen normalisation
--------------------
Feature statistics are computed once on held-out traces and frozen. Per-batch
normalisation would make a vehicle's features depend on which other vehicles
happened to be in its minibatch, which cannot be reproduced when the policy
runs per-vehicle on a real OBU. That would make the trained policy
undeployable in exactly the way the paper claims it is not.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from agents.base import DecisionContext
from common.logging_utils import get_logger

logger = get_logger("agents.graph")

#: Node feature order. Must match ``configs/agent.yaml -> graph.node_features``.
NODE_FEATURES: tuple[str, ...] = (
    "rel_x", "rel_y", "rel_vx", "rel_vy",
    "heading_sin", "heading_cos",
    "eta_to_hazard_s", "relevance_causal", "is_informed",
    "neighbour_count", "hop_count", "message_age_s", "ttl_remaining_s", "is_rsu",
)

#: Edge feature order. Must match ``configs/agent.yaml -> graph.edge_features``.
EDGE_FEATURES: tuple[str, ...] = (
    "distance_m", "relative_speed_ms", "link_lifetime_s",
    "predicted_rssi_dbm", "heading_difference_rad",
)

N_NODE_FEATURES = len(NODE_FEATURES)
N_EDGE_FEATURES = len(EDGE_FEATURES)

#: Cap on estimated link lifetime, seconds. A pair with zero closing rate has
#: infinite lifetime, which is useless as a feature and poisons normalisation.
LINK_LIFETIME_CAP_S = 60.0


@dataclass
class GraphConfig:
    """Resolved from ``configs/agent.yaml -> graph``."""

    neighbour_cap_k: int = 12
    include_rsus: bool = True
    include_neighbour_edges: bool = True
    ttl_s: float = 60.0

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "GraphConfig":
        g = cfg.get("graph", {})
        return cls(
            neighbour_cap_k=int(g.get("neighbour_cap_k", 12)),
            include_rsus=bool(g.get("include_rsus", True)),
            include_neighbour_edges=bool(g.get("include_neighbour_edges", True)),
            ttl_s=float(cfg.get("simulation", {}).get("message_ttl_s", 60.0)),
        )


@dataclass
class DecisionGraph:
    """One decision epoch's graph, in plain NumPy.

    Attributes
    ----------
    x : ``[N, F]``
        Node features. Row 0 is the holder.
    edge_index : ``[2, E]``
        Source/target node indices.
    edge_attr : ``[E, D]``
        Edge features.
    neighbour_ids : ``[N-1]``
        Simulator vehicle indices of nodes 1..N-1, so an action that designates
        "the top-attended neighbour" can be mapped back to a real vehicle.
    """

    x: np.ndarray
    edge_index: np.ndarray
    edge_attr: np.ndarray
    neighbour_ids: np.ndarray
    holder_id: int
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_nodes(self) -> int:
        return int(self.x.shape[0])

    @property
    def n_edges(self) -> int:
        return int(self.edge_index.shape[1])

    def relay_target(self, rank: int) -> int | None:
        """Vehicle id of the ``rank``-th neighbour (0-based), or None.

        Used to turn a ``relay_top_k`` action into a concrete designation.
        """
        if rank < 0 or rank >= self.neighbour_ids.size:
            return None
        return int(self.neighbour_ids[rank])

    def to_pyg(self):  # pragma: no cover - requires torch
        """Convert to a ``torch_geometric.data.Data``. Imported lazily."""
        import torch
        from torch_geometric.data import Data

        return Data(
            x=torch.as_tensor(self.x, dtype=torch.float32),
            edge_index=torch.as_tensor(self.edge_index, dtype=torch.long),
            edge_attr=torch.as_tensor(self.edge_attr, dtype=torch.float32),
        )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------
def _select_neighbours(ctx: DecisionContext, k: int) -> np.ndarray:
    """Indices *into the context's neighbour arrays* for the k nearest."""
    n = ctx.n_neighbours
    if n <= k:
        return np.argsort(ctx.neighbour_distances, kind="stable")
    return np.argsort(ctx.neighbour_distances, kind="stable")[:k]


def build_decision_graph(
    ctx: DecisionContext,
    cfg: GraphConfig | None = None,
    phy: Any | None = None,
) -> DecisionGraph:
    """Build the graph for one holder at one decision epoch.

    ``phy`` is optional and used only to predict RSSI from the median link
    budget; without it that feature is filled with the free-space-equivalent
    ordering (a monotone function of distance), which keeps the feature
    meaningful for tests that do not construct a PHY.
    """
    cfg = cfg or GraphConfig()
    sel = _select_neighbours(ctx, cfg.neighbour_cap_k)
    k = int(sel.size)

    dist = ctx.neighbour_distances[sel]
    dx, dy = ctx.neighbour_dx[sel], ctx.neighbour_dy[sel]
    nvx, nvy = ctx.neighbour_vx[sel], ctx.neighbour_vy[sel]
    nrel = ctx.neighbour_relevance[sel]
    ninf = ctx.neighbour_informed[sel].astype(float)
    nb_ids = ctx.neighbours[sel]

    eta_all = ctx.extras.get("eta_all")
    n_eta = (np.asarray(eta_all)[nb_ids] if eta_all is not None
             else np.full(k, ctx.eta_s, dtype=float))
    n_eta = np.where(np.isfinite(n_eta), n_eta, LINK_LIFETIME_CAP_S * 10)

    own_vx = ctx.speed_ms * np.cos(ctx.heading)
    own_vy = ctx.speed_ms * np.sin(ctx.heading)
    ttl_remaining = max(cfg.ttl_s - ctx.age_s, 0.0)

    # --- nodes: row 0 is the holder ------------------------------------------
    x = np.zeros((k + 1, N_NODE_FEATURES), dtype=np.float64)
    x[0] = (
        0.0, 0.0, 0.0, 0.0,
        np.sin(ctx.heading), np.cos(ctx.heading),
        min(ctx.eta_s, LINK_LIFETIME_CAP_S * 10), ctx.relevance, 1.0,
        float(ctx.n_neighbours), float(ctx.hop_count), ctx.age_s, ttl_remaining, 0.0,
    )
    if k:
        n_head = np.arctan2(nvy, nvx)
        x[1:, 0] = dx
        x[1:, 1] = dy
        x[1:, 2] = nvx - own_vx
        x[1:, 3] = nvy - own_vy
        x[1:, 4] = np.sin(n_head)
        x[1:, 5] = np.cos(n_head)
        x[1:, 6] = np.minimum(n_eta, LINK_LIFETIME_CAP_S * 10)
        x[1:, 7] = nrel
        x[1:, 8] = ninf
        x[1:, 9] = float(ctx.n_neighbours)
        x[1:, 10] = float(ctx.hop_count)
        x[1:, 11] = ctx.age_s
        x[1:, 12] = ttl_remaining
        x[1:, 13] = 0.0  # RSUs are flagged here once the engine models them

    # --- edges ---------------------------------------------------------------
    src: list[int] = []
    dst: list[int] = []
    for j in range(1, k + 1):
        src += [j, 0]          # neighbour -> holder, and holder -> neighbour
        dst += [0, j]

    if cfg.include_neighbour_edges and k > 1:
        # Neighbour-to-neighbour edges where the two are themselves within
        # range. Without these the graph is a star and a GAT degenerates to
        # attention pooling, which would make the "GAT vs MLP" ablation
        # vacuous: there would be no structure for the GAT to exploit.
        pos = np.stack([dx, dy], axis=1)
        d_nn = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)
        ii, jj = np.nonzero((d_nn <= ctx.comm_range_m) & (d_nn > 0))
        src += (ii + 1).tolist()
        dst += (jj + 1).tolist()

    edge_index = np.asarray([src, dst], dtype=np.int64) if src else np.zeros((2, 0), np.int64)

    # --- edge features -------------------------------------------------------
    e = edge_index.shape[1]
    edge_attr = np.zeros((e, N_EDGE_FEATURES), dtype=np.float64)
    if e:
        node_pos = np.zeros((k + 1, 2))
        node_vel = np.zeros((k + 1, 2))
        node_pos[1:, 0], node_pos[1:, 1] = dx, dy
        node_vel[0] = (own_vx, own_vy)
        if k:
            node_vel[1:, 0], node_vel[1:, 1] = nvx, nvy

        a, b = edge_index[0], edge_index[1]
        delta = node_pos[b] - node_pos[a]
        d = np.linalg.norm(delta, axis=1)
        dv = node_vel[b] - node_vel[a]
        rel_speed = np.linalg.norm(dv, axis=1)

        # Link lifetime: time until the pair separates beyond comm range at the
        # current closing rate. Positive closing rate -> they are approaching,
        # so the link is not the binding constraint; cap it.
        with np.errstate(divide="ignore", invalid="ignore"):
            radial = np.einsum("ij,ij->i", delta, dv) / np.maximum(d, 1e-9)
            lifetime = np.where(radial > 1e-6, (ctx.comm_range_m - d) / radial,
                                LINK_LIFETIME_CAP_S)
        lifetime = np.clip(np.nan_to_num(lifetime, nan=LINK_LIFETIME_CAP_S),
                           0.0, LINK_LIFETIME_CAP_S)

        if phy is not None:
            rssi = np.asarray(phy.median_rx_power_dbm(np.maximum(d, 1.0)), dtype=float)
        else:
            rssi = -40.0 - 20.0 * np.log10(np.maximum(d, 1.0))

        head = np.arctan2(node_vel[:, 1], node_vel[:, 0])
        dhead = np.abs((head[b] - head[a] + np.pi) % (2 * np.pi) - np.pi)

        edge_attr[:, 0] = d
        edge_attr[:, 1] = rel_speed
        edge_attr[:, 2] = lifetime
        edge_attr[:, 3] = rssi
        edge_attr[:, 4] = dhead

    return DecisionGraph(
        x=x, edge_index=edge_index, edge_attr=edge_attr,
        neighbour_ids=np.asarray(nb_ids, dtype=np.int64), holder_id=int(ctx.index),
        meta={"step": ctx.step, "k": k, "trigger": ctx.trigger.value,
              "n_neighbours_total": ctx.n_neighbours},
    )


# ---------------------------------------------------------------------------
# Frozen feature normalisation
# ---------------------------------------------------------------------------
@dataclass
class FeatureNormaliser:
    """Z-score normalisation with statistics frozen at fit time.

    Never re-fit on evaluation data, and never fit per batch: a deployed OBU
    normalises its own features with constants baked into the model, and the
    training distribution must match that.
    """

    node_mean: np.ndarray
    node_std: np.ndarray
    edge_mean: np.ndarray
    edge_std: np.ndarray
    n_samples: int = 0
    frozen: bool = True
    #: Features whose fit-set std was below ``min_std``; they are centred but
    #: not scaled. Recorded so a degenerate fit is visible rather than silent.
    degenerate: tuple[str, ...] = ()
    #: What the statistics were fitted on (scenarios, densities, seeds, mode).
    provenance: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def fit(
        cls,
        graphs: Sequence[DecisionGraph],
        min_std: float = 1e-3,
        provenance: dict[str, Any] | None = None,
    ) -> "FeatureNormaliser":
        """Fit z-score statistics; degenerate features get std 1.0, not a floor.

        The first version floored std at 1e-6. A feature that happens to be
        constant in the fit set then gets multiplied by a million wherever it
        is not constant -- and the fit set was entirely an east-west rural
        corridor, where ``rel_vy`` and ``heading_sin`` are exactly zero. The
        first urban vehicle driving north at 15 m/s became a ~1.5e7 input.

        Centring a degenerate feature without scaling it keeps inputs on the
        feature's own physical scale, which is bounded. It is a safety net, not
        the fix: the fix is fitting on a representative sample
        (``agents.train.fit_normaliser``), and ``degenerate`` records when that
        did not happen.
        """
        xs = np.concatenate([g.x for g in graphs if g.n_nodes], axis=0)
        es = [g.edge_attr for g in graphs if g.n_edges]
        ea = np.concatenate(es, axis=0) if es else np.zeros((1, N_EDGE_FEATURES))

        node_std, edge_std = xs.std(axis=0), ea.std(axis=0)
        degenerate = tuple(
            [NODE_FEATURES[i] for i in np.flatnonzero(node_std < min_std)]
            + [EDGE_FEATURES[i] for i in np.flatnonzero(edge_std < min_std)]
        )
        if degenerate:
            logger.warning(
                "Feature statistics are degenerate for %s (std < %g in the fit set); "
                "they are centred but not scaled. A representative fit should not "
                "produce this except for features that are genuinely constant "
                "(e.g. is_rsu with no RSUs modelled).", list(degenerate), min_std,
            )
        return cls(
            node_mean=xs.mean(axis=0), node_std=np.where(node_std < min_std, 1.0, node_std),
            edge_mean=ea.mean(axis=0), edge_std=np.where(edge_std < min_std, 1.0, edge_std),
            n_samples=len(graphs), degenerate=degenerate,
            provenance=dict(provenance or {}),
        )

    def apply(self, graph: DecisionGraph) -> DecisionGraph:
        return DecisionGraph(
            x=(graph.x - self.node_mean) / self.node_std,
            edge_index=graph.edge_index,
            edge_attr=((graph.edge_attr - self.edge_mean) / self.edge_std
                       if graph.n_edges else graph.edge_attr),
            neighbour_ids=graph.neighbour_ids, holder_id=graph.holder_id,
            meta={**graph.meta, "normalised": True},
        )

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "node_features": list(NODE_FEATURES),
            "edge_features": list(EDGE_FEATURES),
            "node_mean": self.node_mean.tolist(), "node_std": self.node_std.tolist(),
            "edge_mean": self.edge_mean.tolist(), "edge_std": self.edge_std.tolist(),
            "n_samples": self.n_samples,
            "degenerate": list(self.degenerate),
            "provenance": self.provenance,
        }, indent=1), encoding="utf-8")
        logger.info("Froze feature stats from %d graphs -> %s", self.n_samples, path)
        return path

    @classmethod
    def load(cls, path: Path) -> "FeatureNormaliser":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        if tuple(d["node_features"]) != NODE_FEATURES:
            raise ValueError(
                "Frozen stats were fitted with a different node-feature set; "
                "refit them or the model will read the wrong columns."
            )
        return cls(
            node_mean=np.asarray(d["node_mean"]), node_std=np.asarray(d["node_std"]),
            edge_mean=np.asarray(d["edge_mean"]), edge_std=np.asarray(d["edge_std"]),
            n_samples=int(d.get("n_samples", 0)),
            degenerate=tuple(d.get("degenerate", ())),
            provenance=dict(d.get("provenance", {})),
        )

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FeatureNormaliser":
        """Rebuild from the dict ``save`` writes -- used for stats embedded in a
        checkpoint, so a checkpoint is always evaluated with its own statistics."""
        if tuple(d["node_features"]) != NODE_FEATURES:
            raise ValueError(
                "Embedded stats were fitted with a different node-feature set; "
                "this checkpoint cannot be evaluated with the current graph schema."
            )
        return cls(
            node_mean=np.asarray(d["node_mean"]), node_std=np.asarray(d["node_std"]),
            edge_mean=np.asarray(d["edge_mean"]), edge_std=np.asarray(d["edge_std"]),
            n_samples=int(d.get("n_samples", 0)),
            degenerate=tuple(d.get("degenerate", ())),
            provenance=dict(d.get("provenance", {})),
        )

    def matches(self, mode: str, scenarios: Sequence[str]) -> bool:
        """Were these statistics fitted for this purpose?

        A smoke-mode fit used to be written to the canonical path and then
        loaded, without complaint, by a later full run -- which is how 120
        graphs from the opening moments of one rural run came to normalise
        urban training. Statistics with no recorded provenance (everything
        fitted before provenance was recorded) never match.
        """
        return (self.provenance.get("mode") == mode
                and set(self.provenance.get("scenarios", [])) == set(scenarios))
