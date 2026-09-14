"""Phase 5.2: GATv2 encoder, PPO actor-critic and Dueling DQN.

Requires torch + torch-geometric (see ENVIRONMENT.md -- they live in the D:
environment). Nothing in Phases 1-4 imports this module, so the rest of the
project runs on an interpreter with no torch installed.

Attention *is* the relay selection
----------------------------------
The attention coefficients the final GATv2 layer places on the edges
``neighbour -> holder`` are used directly as the relay-selection ranking. They
are not a visualisation read out of a model that decides some other way: the
``relay_top_k`` actions designate the k-th most attended neighbour, so the
heatmap figure in the paper shows the quantity that actually drove the
decision. If that ranking were computed by a separate head, the interpretability
claim would be decorative.

Why edge features matter here
-----------------------------
``GATv2Conv`` with ``edge_dim`` set lets the attention logit depend on distance,
predicted RSSI and estimated link lifetime, not just on the two endpoint
embeddings. That is the whole reason to use GATv2 over GAT: attention over a
vehicular graph should be able to say "this neighbour is far but the link will
survive long enough", which is a statement about the edge.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GATv2Conv, GCNConv
from torch_geometric.utils import softmax as pyg_softmax

from agents.graph import N_EDGE_FEATURES, N_NODE_FEATURES

#: Discrete action space. Order is fixed and mirrors
#: ``configs/agent.yaml -> action_space.actions``.
ACTIONS: tuple[str, ...] = (
    "suppress",
    "broadcast_now",
    "defer_1",
    "defer_2",
    "defer_3",
    "relay_top_1",
    "relay_top_2",
    "relay_top_3",
    "carry_and_forward",
)
N_ACTIONS = len(ACTIONS)
RELAY_ACTIONS = {"relay_top_1": 0, "relay_top_2": 1, "relay_top_3": 2}


@dataclass
class EncoderConfig:
    kind: str = "gatv2"          # gatv2 | gcn | mlp  (the ablation axis)
    layers: int = 3
    heads: int = 4
    hidden_dim: int = 64
    dropout: float = 0.1
    residual: bool = True
    layer_norm: bool = True
    node_dim: int = N_NODE_FEATURES
    edge_dim: int = N_EDGE_FEATURES

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "EncoderConfig":
        e = cfg.get("encoder", {})
        return cls(
            kind=str(e.get("type", "gatv2")),
            layers=int(e.get("layers", 3)),
            heads=int(e.get("heads", 4)),
            hidden_dim=int(e.get("hidden_dim", 64)),
            dropout=float(e.get("dropout", 0.1)),
            residual=bool(e.get("residual", True)),
            layer_norm=bool(e.get("layer_norm", True)),
        )


class GraphEncoder(nn.Module):
    """GATv2 encoder, with GCN and MLP variants for the ablation.

    All three expose the same interface so the ablation swaps one config field
    rather than a code path. The MLP variant ignores ``edge_index`` entirely --
    that is the point of it: it measures what the graph structure is worth.
    """

    def __init__(self, cfg: EncoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        h = cfg.hidden_dim
        if cfg.kind == "gatv2" and h % cfg.heads:
            raise ValueError(f"hidden_dim {h} must be divisible by heads {cfg.heads}")

        self.input_proj = nn.Linear(cfg.node_dim, h)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(cfg.layers):
            if cfg.kind == "gatv2":
                self.convs.append(
                    GATv2Conv(h, h // cfg.heads, heads=cfg.heads, concat=True,
                              edge_dim=cfg.edge_dim, add_self_loops=False)
                )
            elif cfg.kind == "gcn":
                self.convs.append(GCNConv(h, h, add_self_loops=False))
            elif cfg.kind == "mlp":
                self.convs.append(nn.Linear(h, h))
            else:
                raise ValueError(f"unknown encoder type {cfg.kind!r}")
            self.norms.append(nn.LayerNorm(h) if cfg.layer_norm else nn.Identity())
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Return ``(node_embeddings, final_alpha, final_edge_index)``.

        ``final_alpha`` is the last layer's per-edge attention, averaged over
        heads; it is None for the non-attention variants.
        """
        h = self.input_proj(x)
        alpha = a_index = None

        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            last = i == len(self.convs) - 1
            if self.cfg.kind == "gatv2":
                if last:
                    out, (a_index, alpha) = conv(
                        h, edge_index, edge_attr=edge_attr,
                        return_attention_weights=True,
                    )
                    alpha = alpha.mean(dim=1)          # average the heads
                else:
                    out = conv(h, edge_index, edge_attr=edge_attr)
            elif self.cfg.kind == "gcn":
                out = conv(h, edge_index)
            else:
                out = conv(h)

            out = norm(out)
            out = F.elu(out)
            out = self.dropout(out)
            h = h + out if self.cfg.residual else out
        return h, alpha, a_index


def relay_ranking(
    alpha: torch.Tensor | None,
    edge_index: torch.Tensor | None,
    holder_nodes: torch.Tensor,
    n_nodes: int,
) -> torch.Tensor:
    """Per-node relay score from attention onto the holder.

    Scores every node by the attention weight on its edge into its graph's
    holder. Nodes with no such edge score ``-inf`` so they can never be
    designated. With no attention available (GCN/MLP ablations) the ranking
    falls back to zeros, which makes ``relay_top_k`` pick by node order -- a
    deliberate handicap that is part of what those ablations measure.
    """
    scores = torch.full((n_nodes,), float("-inf"))
    if alpha is None or edge_index is None:
        return torch.zeros(n_nodes)
    src, dst = edge_index[0], edge_index[1]
    into_holder = torch.isin(dst, holder_nodes)
    if into_holder.any():
        scores[src[into_holder]] = alpha[into_holder].to(scores.dtype)
    return scores


class ActorCritic(nn.Module):
    """Shared-policy PPO actor-critic over the decision graph.

    Centralised training, decentralised execution: the network sees one
    vehicle's local graph, so the trained policy runs unchanged on a single OBU.
    Nothing global enters the forward pass.
    """

    def __init__(self, cfg: EncoderConfig, n_actions: int = N_ACTIONS) -> None:
        super().__init__()
        self.encoder = GraphEncoder(cfg)
        h = cfg.hidden_dim
        self.actor = nn.Sequential(
            nn.Linear(h, h), nn.ReLU(), nn.Linear(h, n_actions)
        )
        self.critic = nn.Sequential(
            nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1)
        )
        self.n_actions = n_actions

    def forward(self, data: Data | Batch) -> dict[str, torch.Tensor]:
        emb, alpha, a_index = self.encoder(data.x, data.edge_index, data.edge_attr)

        # Node 0 of every graph in the batch is its holder.
        batch = getattr(data, "batch", None)
        if batch is None:
            holder_nodes = torch.zeros(1, dtype=torch.long)
        else:
            counts = torch.bincount(batch, minlength=int(batch.max()) + 1)
            holder_nodes = torch.cat([
                torch.zeros(1, dtype=torch.long),
                torch.cumsum(counts, 0)[:-1],
            ])

        holder_emb = emb[holder_nodes]
        logits = self.actor(holder_emb)
        value = self.critic(holder_emb).squeeze(-1)
        ranking = relay_ranking(alpha, a_index, holder_nodes, emb.shape[0])
        return {
            "logits": logits, "value": value, "relay_scores": ranking,
            "holder_nodes": holder_nodes, "attention": alpha, "attn_edges": a_index,
        }

    @torch.no_grad()
    def act(
        self, data: Data, deterministic: bool = False, action_mask: Any | None = None,
        uniform: float | None = None,
    ) -> dict[str, Any]:
        """Sample (or argmax) one action. ``action_mask`` (True = available) is
        applied before sampling, so log-prob and entropy describe the
        distribution the action was actually drawn from.

        Sampling is inverse-CDF on one uniform draw (``uniform``, or
        ``torch.rand(())`` from the global generator), the same rule
        :meth:`act_batch` applies per graph -- so a batch of decisions picks
        exactly the actions the same decisions would pick one at a time.
        """
        out = self(data)
        logits = out["logits"][0]
        if action_mask is not None:
            logits = mask_logits(logits, torch.as_tensor(action_mask, dtype=torch.bool))
        dist = torch.distributions.Categorical(logits=logits)
        probs = dist.probs
        if deterministic:
            action = int(torch.argmax(logits))
        else:
            u = torch.rand(()) if uniform is None else torch.as_tensor(float(uniform))
            action = int(sample_inverse_cdf(probs[None, :], u[None])[0])
        return {
            "action": action,
            "log_prob": float(dist.log_prob(torch.tensor(action))),
            "value": float(out["value"][0]),
            "entropy": float(dist.entropy()),
            "probs": probs.numpy(),
            "relay_order": _relay_order(out["relay_scores"]),
        }

    @torch.no_grad()
    def act_batch(
        self, batch: Batch, deterministic: bool = False,
        action_masks: torch.Tensor | None = None, uniforms: torch.Tensor | None = None,
    ) -> list[dict[str, Any]]:
        """One forward pass for a whole batch of decision graphs.

        Returns one dict per graph with the same keys as :meth:`act`. Masks
        are applied per graph before sampling; ``uniforms`` (one per graph,
        default ``torch.rand(B)``, which equals B successive ``torch.rand(())``
        draws) drive the same inverse-CDF rule as :meth:`act`.
        """
        out = self(batch)
        logits = out["logits"]
        n = logits.shape[0]
        if action_masks is not None:
            logits = mask_logits(logits, action_masks.to(torch.bool))
        dist = torch.distributions.Categorical(logits=logits)
        probs = dist.probs
        if deterministic:
            actions = torch.argmax(logits, dim=-1)
        else:
            u = torch.rand(n) if uniforms is None else uniforms.to(probs.dtype)
            actions = sample_inverse_cdf(probs, u)
        log_probs = dist.log_prob(actions)
        entropies = dist.entropy()
        ptr = batch.ptr.tolist()
        scores = out["relay_scores"]
        return [{
            "action": int(actions[j]),
            "log_prob": float(log_probs[j]),
            "value": float(out["value"][j]),
            "entropy": float(entropies[j]),
            "probs": probs[j].numpy(),
            "relay_order": _relay_order(scores[ptr[j]:ptr[j + 1]]),
        } for j in range(n)]

    def evaluate_actions(
        self, data: Batch, actions: torch.Tensor, action_masks: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Log-probs, values and entropies for PPO, under each decision's mask."""
        out = self(data)
        logits = out["logits"]
        if action_masks is not None:
            logits = mask_logits(logits, action_masks.to(torch.bool))
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(actions), out["value"], dist.entropy()


#: Logit given to unavailable actions. Finite, so entropy stays 0 * finite
#: rather than 0 * -inf = NaN; its softmax probability underflows to exactly 0.
MASKED_LOGIT = -1e9


def sample_inverse_cdf(probs: torch.Tensor, uniforms: torch.Tensor) -> torch.Tensor:
    """Row-wise categorical sample from ``[B, A]`` probabilities and ``[B]`` uniforms.

    Picks the first action whose cumulative probability exceeds ``u * total``.
    A zero-probability (masked) action never has a CDF step, so it is never
    chosen; float round-off at ``u -> 1`` is clamped to the last available
    action rather than past it.
    """
    cdf = torch.cumsum(probs, dim=-1)
    target = (uniforms * cdf[:, -1]).unsqueeze(-1)
    idx = torch.searchsorted(cdf, target, right=True).squeeze(-1)
    available = probs > 0
    last = probs.shape[-1] - 1 - torch.argmax(available.flip(-1).to(torch.int8), dim=-1)
    return torch.minimum(idx, last)


def mask_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Set logits of unavailable actions (``mask`` False) to :data:`MASKED_LOGIT`."""
    if not bool(mask.any(dim=-1).all()):
        raise ValueError("action mask leaves no available action")
    return logits.masked_fill(~mask, MASKED_LOGIT)


def _relay_order(scores: torch.Tensor) -> list[int]:
    """Neighbour node indices ranked by attention, best first.

    Node 0 is the holder and is excluded; returned indices are 1-based node
    indices, which the policy maps to vehicle ids via
    ``DecisionGraph.neighbour_ids``.
    """
    s = scores.clone()
    if s.numel() <= 1:
        return []
    s[0] = float("-inf")
    finite = torch.isfinite(s)
    if not finite.any():
        return list(range(1, s.numel()))
    order = torch.argsort(torch.where(finite, s, torch.full_like(s, -1e30)),
                          descending=True)
    return [int(i) for i in order if finite[i]]


class DuelingQNetwork(nn.Module):
    """Dueling DQN head over the same encoder (secondary result).

    Q(s,a) = V(s) + A(s,a) - mean_a A(s,a). The mean-subtraction is what makes
    the decomposition identifiable; without it V and A can drift by an
    arbitrary constant and the learned V is meaningless.
    """

    def __init__(self, cfg: EncoderConfig, n_actions: int = N_ACTIONS) -> None:
        super().__init__()
        self.encoder = GraphEncoder(cfg)
        h = cfg.hidden_dim
        self.value = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))
        self.advantage = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, n_actions))
        self.n_actions = n_actions

    def forward(self, data: Data | Batch) -> dict[str, torch.Tensor]:
        emb, alpha, a_index = self.encoder(data.x, data.edge_index, data.edge_attr)
        batch = getattr(data, "batch", None)
        if batch is None:
            holder_nodes = torch.zeros(1, dtype=torch.long)
        else:
            counts = torch.bincount(batch, minlength=int(batch.max()) + 1)
            holder_nodes = torch.cat([
                torch.zeros(1, dtype=torch.long), torch.cumsum(counts, 0)[:-1]
            ])
        holder_emb = emb[holder_nodes]
        v = self.value(holder_emb)
        a = self.advantage(holder_emb)
        q = v + a - a.mean(dim=-1, keepdim=True)
        return {
            "q": q, "value": v.squeeze(-1),
            "relay_scores": relay_ranking(alpha, a_index, holder_nodes, emb.shape[0]),
            "attention": alpha, "attn_edges": a_index,
        }

    @torch.no_grad()
    def act(
        self, data: Data, epsilon: float = 0.0, action_mask: Any | None = None,
        deterministic: bool = True,
    ) -> dict[str, Any]:
        out = self(data)
        q = out["q"][0]
        allowed = torch.arange(self.n_actions)
        if action_mask is not None:
            m = torch.as_tensor(action_mask, dtype=torch.bool)
            q = mask_logits(q, m)
            allowed = allowed[m]
        if epsilon > 0 and float(torch.rand(1)) < epsilon:
            action = int(allowed[torch.randint(allowed.numel(), (1,))])
        else:
            action = int(torch.argmax(q))
        # Softmax over Q is not a calibrated posterior, but its entropy is a
        # usable dispersion signal for the confidence gate; documented as such
        # in agents/confidence.py.
        probs = torch.softmax(q, dim=-1)
        return {
            "action": action, "q": q.numpy(), "probs": probs.numpy(),
            "entropy": float(-(probs * torch.log(probs + 1e-12)).sum()),
            "relay_order": _relay_order(out["relay_scores"]),
        }


def build_network(cfg: dict[str, Any]) -> nn.Module:
    """Construct the network named by ``configs/agent.yaml -> algorithm.name``."""
    enc = EncoderConfig.from_config(cfg)
    name = str(cfg.get("algorithm", {}).get("name", "ppo")).lower()
    if name == "ppo":
        return ActorCritic(enc)
    if name in ("dueling_dqn", "dqn"):
        return DuelingQNetwork(enc)
    raise ValueError(f"unknown algorithm {name!r}")


def graphs_to_batch(graphs: Sequence[Any]) -> Batch:
    """Collate ``DecisionGraph`` objects into a PyG batch."""
    return Batch.from_data_list([g.to_pyg() for g in graphs])


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
