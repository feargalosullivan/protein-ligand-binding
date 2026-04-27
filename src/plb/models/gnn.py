"""Ligand-GNN + ESM-pocket + MLP affinity predictor.

The model is intentionally small: a 3-layer GINE encoder over the ligand
graph, a 2-layer MLP over the cached ESM pocket / whole-protein embeddings,
and a linear head fusing the two. Total parameter count is in the few
hundred thousand range so it fits comfortably on a 6 GB GPU and trains in a
few minutes.

Architecture (default hyperparameters):

  Ligand graph (node feats 36, edge feats 8)
      -> Linear(36 -> H) ; Linear(8 -> H)
      -> GINEConv x N (each MLP: H -> H -> H, with BN + ReLU + dropout)
      -> Global mean-pool ++ max-pool          -> 2H
      -> Linear(2H -> H)                       -> H
                                                       \\
  ESM pocket (480) ++ ESM whole (480) -> 960            \\
      -> Linear(960 -> 2H) -> ReLU -> Dropout            >--> [H | H] -> 2H
      -> Linear(2H -> H)                       -> H     /
                                                       /
                                  Linear(2H -> H) -> ReLU -> Dropout
                                  Linear(H -> 1) -> pK
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch_geometric.nn import GINEConv, global_max_pool, global_mean_pool


@dataclass(frozen=True)
class GNNConfig:
    """Hyperparameters for ``AffinityModel``.

    Defaults are the "config A" we sweep first; the notebook overrides
    ``hidden_dim`` / ``n_gnn_layers`` / ``dropout`` for the other configs.
    """

    node_dim: int = 36
    edge_dim: int = 8
    esm_pocket_dim: int = 480
    esm_whole_dim: int = 480
    hidden_dim: int = 128
    n_gnn_layers: int = 3
    dropout: float = 0.1


def _gine_block(hidden: int, dropout: float) -> GINEConv:
    """One GINE message-passing layer with BN + ReLU + dropout in its MLP.

    GINEConv learns ``h_v' = MLP((1 + eps) * h_v + sum_{u in N(v)} ReLU(h_u + e_uv))``,
    so its update MLP needs to map ``hidden -> hidden``.
    """
    mlp = nn.Sequential(
        nn.Linear(hidden, hidden),
        nn.BatchNorm1d(hidden),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, hidden),
    )
    return GINEConv(mlp, train_eps=True)


class GINEncoder(nn.Module):
    """Stack of GINE convolutions followed by global mean+max pooling.

    Returns a fixed-size graph embedding of shape ``(batch_size, 2 * hidden_dim)``.
    """

    def __init__(
        self, node_dim: int, edge_dim: int, hidden_dim: int, n_layers: int, dropout: float
    ) -> None:
        super().__init__()
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1; got {n_layers}")
        self.hidden_dim = hidden_dim
        self.node_encoder = nn.Linear(node_dim, hidden_dim)
        self.edge_encoder = nn.Linear(edge_dim, hidden_dim)
        self.convs = nn.ModuleList(_gine_block(hidden_dim, dropout) for _ in range(n_layers))
        self.bns = nn.ModuleList(nn.BatchNorm1d(hidden_dim) for _ in range(n_layers))
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        h = self.node_encoder(x)
        e = self.edge_encoder(edge_attr)
        for conv, bn in zip(self.convs, self.bns, strict=True):
            h = conv(h, edge_index, e)
            h = bn(h)
            h = torch.relu(h)
            h = self.dropout(h)
        return torch.cat([global_mean_pool(h, batch), global_max_pool(h, batch)], dim=1)


class _MLP(nn.Module):
    """``Linear -> ReLU -> Dropout -> Linear`` block."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AffinityModel(nn.Module):
    """Full ligand-GNN + ESM-pool affinity predictor."""

    def __init__(self, cfg: GNNConfig) -> None:
        super().__init__()
        self.cfg = cfg
        h = cfg.hidden_dim

        self.ligand_encoder = GINEncoder(
            node_dim=cfg.node_dim,
            edge_dim=cfg.edge_dim,
            hidden_dim=h,
            n_layers=cfg.n_gnn_layers,
            dropout=cfg.dropout,
        )
        self.ligand_proj = nn.Linear(2 * h, h)

        self.protein_mlp = _MLP(
            in_dim=cfg.esm_pocket_dim + cfg.esm_whole_dim,
            hidden_dim=2 * h,
            out_dim=h,
            dropout=cfg.dropout,
        )

        self.head = nn.Sequential(
            nn.Linear(2 * h, h),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(h, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
        esm_pocket: torch.Tensor,
        esm_whole: torch.Tensor,
    ) -> torch.Tensor:
        """Predict pK for a batch.

        Parameters
        ----------
        x, edge_index, edge_attr, batch
            Standard PyG mini-batch graph attributes.
        esm_pocket
            ``(B, esm_pocket_dim)`` cached pocket-pool embedding.
        esm_whole
            ``(B, esm_whole_dim)`` cached whole-protein-pool embedding.

        Returns
        -------
        ``(B,)`` predicted pK values.
        """
        ligand = self.ligand_proj(self.ligand_encoder(x, edge_index, edge_attr, batch))
        protein = self.protein_mlp(torch.cat([esm_pocket, esm_whole], dim=1))
        return self.head(torch.cat([ligand, protein], dim=1)).squeeze(-1)

    @torch.no_grad()
    def n_parameters(self) -> int:
        """Convenience: count of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


__all__ = ["AffinityModel", "GINEncoder", "GNNConfig"]
