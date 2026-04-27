"""Unit tests for the Phase 4 GNN model.

We use synthetic graphs (random node/edge features) so the test runs in
milliseconds without needing RDKit or PyG datasets on disk.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pyg_data = pytest.importorskip("torch_geometric.data")

from plb.models.gnn import AffinityModel, GINEncoder, GNNConfig  # noqa: E402


def _random_batch(
    node_dim: int, edge_dim: int, esm_pocket_dim: int, esm_whole_dim: int, n_graphs: int = 3
):
    """Build a PyG ``Batch`` of ``n_graphs`` random graphs."""
    from torch_geometric.data import Batch, Data

    torch.manual_seed(0)
    graphs = []
    sizes = [4, 7, 10][:n_graphs] + [5] * max(0, n_graphs - 3)
    for n_atoms in sizes:
        n_edges = max(2, n_atoms)
        graphs.append(
            Data(
                x=torch.randn(n_atoms, node_dim),
                edge_index=torch.randint(0, n_atoms, (2, n_edges)),
                edge_attr=torch.randn(n_edges, edge_dim),
                esm_pocket=torch.randn(1, esm_pocket_dim),
                esm_whole=torch.randn(1, esm_whole_dim),
                y=torch.randn(1),
                pdb_id=f"fake{len(graphs)}",
            )
        )
    return Batch.from_data_list(graphs), n_graphs


def test_gine_encoder_output_shape() -> None:
    encoder = GINEncoder(node_dim=36, edge_dim=8, hidden_dim=32, n_layers=3, dropout=0.0)
    encoder.eval()
    batch, n_graphs = _random_batch(node_dim=36, edge_dim=8, esm_pocket_dim=4, esm_whole_dim=4)
    out = encoder(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
    assert out.shape == (n_graphs, 2 * 32), "global mean + max pool concat -> 2*hidden"
    assert out.dtype == torch.float32


def test_gine_encoder_invalid_layers() -> None:
    with pytest.raises(ValueError, match="n_layers"):
        GINEncoder(node_dim=4, edge_dim=2, hidden_dim=8, n_layers=0, dropout=0.0)


def test_affinity_model_forward_shape() -> None:
    cfg = GNNConfig(
        node_dim=36,
        edge_dim=8,
        esm_pocket_dim=16,
        esm_whole_dim=16,
        hidden_dim=32,
        n_gnn_layers=2,
        dropout=0.0,
    )
    model = AffinityModel(cfg)
    model.eval()
    batch, n_graphs = _random_batch(node_dim=36, edge_dim=8, esm_pocket_dim=16, esm_whole_dim=16)
    pred = model(
        batch.x,
        batch.edge_index,
        batch.edge_attr,
        batch.batch,
        batch.esm_pocket,
        batch.esm_whole,
    )
    assert pred.shape == (n_graphs,), "one scalar pK prediction per graph"
    assert torch.isfinite(pred).all()


def test_affinity_model_n_parameters_positive() -> None:
    model = AffinityModel(GNNConfig(hidden_dim=8, n_gnn_layers=2))
    n = model.n_parameters()
    assert n > 0
    # Sanity check we don't accidentally have a multi-million-param model
    # for tiny hidden dims (catches Linear-on-wrong-dim regressions).
    assert n < 200_000


def test_affinity_model_backward_runs() -> None:
    cfg = GNNConfig(
        node_dim=4,
        edge_dim=2,
        esm_pocket_dim=4,
        esm_whole_dim=4,
        hidden_dim=8,
        n_gnn_layers=2,
        dropout=0.0,
    )
    model = AffinityModel(cfg)
    batch, _ = _random_batch(node_dim=4, edge_dim=2, esm_pocket_dim=4, esm_whole_dim=4)
    pred = model(
        batch.x,
        batch.edge_index,
        batch.edge_attr,
        batch.batch,
        batch.esm_pocket,
        batch.esm_whole,
    )
    loss = (pred - batch.y).pow(2).mean()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() for g in grads)
