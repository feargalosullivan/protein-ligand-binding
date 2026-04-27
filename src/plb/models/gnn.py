"""Ligand-GNN + ESM-pocket + MLP affinity predictor."""

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch_geometric.nn import GINEConv, global_max_pool, global_mean_pool


@dataclass(frozen=True)
class GNNConfig:

    node_dim: int = 36
    edge_dim: int = 8
    esm_pocket_dim: int = 480
    esm_whole_dim: int = 480
    hidden_dim: int = 128
    n_gnn_layers: int = 3
    dropout: float = 0.1


def _gine_block(hidden: int, dropout: float) -> GINEConv:
    mlp = nn.Sequential(
        nn.Linear(hidden, hidden),
        nn.BatchNorm1d(hidden),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, hidden),
    )
    return GINEConv(mlp, train_eps=True)


class GINEncoder(nn.Module):
    """GINE convolution stack -> global mean+max pool -> (batch, 2*hidden_dim)."""

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
        # TODO: try attention pooling (set2set or similar) instead of mean+max
        return torch.cat([global_mean_pool(h, batch), global_max_pool(h, batch)], dim=1)


class _MLP(nn.Module):

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
        ligand = self.ligand_proj(self.ligand_encoder(x, edge_index, edge_attr, batch))
        protein = self.protein_mlp(torch.cat([esm_pocket, esm_whole], dim=1))
        return self.head(torch.cat([ligand, protein], dim=1)).squeeze(-1)

    @torch.no_grad()
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
