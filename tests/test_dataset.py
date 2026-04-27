"""Tests for plb.data.dataset - the PyG dataset that fuses ligand graphs + ESM features.

We mock the ESM cache with synthetic ``.npz`` files and the ligand-graph cache
with a hand-built dict, so these tests run without RDKit, PDBbind, or any real
ESM model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pyg_data = pytest.importorskip("torch_geometric.data")

from plb.data.cache import esm_cache_dir  # noqa: E402
from plb.data.dataset import (  # noqa: E402
    PDBbindGraphDataset,
    build_data_object,
    ligand_cache_path,
    load_ligand_graph_cache,
    make_dataloader,
)


def _make_fake_ligand(n_atoms: int = 5, node_dim: int = 36, edge_dim: int = 8) -> dict:
    n_edges = max(2, n_atoms)
    return {
        "x": torch.randn(n_atoms, node_dim),
        "edge_index": torch.randint(0, n_atoms, (2, n_edges)),
        "edge_attr": torch.randn(n_edges, edge_dim),
        "smiles": "CCO",
    }


def _write_fake_esm(cache_dir: Path, pdb_id: str, dim: int = 8, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_dir / f"{pdb_id}.npz",
        pocket=rng.standard_normal(dim).astype(np.float32),
        whole=rng.standard_normal(dim).astype(np.float32),
        n_pocket=np.int32(10),
        n_total=np.int32(100),
        cutoff=np.float32(6.0),
    )


def test_build_data_object_shapes() -> None:
    ligand = _make_fake_ligand(n_atoms=4)
    pocket = np.zeros(8, dtype=np.float32)
    whole = np.ones(8, dtype=np.float32)
    data = build_data_object("1abc", ligand, pocket, whole, pK=7.5)

    assert data.x.shape == (4, 36)
    assert data.esm_pocket.shape == (1, 8)
    assert data.esm_whole.shape == (1, 8)
    assert data.y.shape == (1,)
    assert data.pdb_id == "1abc"
    assert float(data.y) == pytest.approx(7.5)


def test_dataset_basic(tmp_path: Path) -> None:
    cache = esm_cache_dir(tmp_path)
    for i, pid in enumerate(["1abc", "2def"]):
        _write_fake_esm(cache, pid, dim=8, seed=i)
    ligand_cache = {"1abc": _make_fake_ligand(4), "2def": _make_fake_ligand(7)}

    ds = PDBbindGraphDataset(
        pdb_ids=["1abc", "2def"],
        labels={"1abc": 5.0, "2def": 9.0},
        data_root=tmp_path,
        refined_dir=tmp_path / "raw" / "refined",  # not actually read
        ligand_cache=ligand_cache,
    )
    assert len(ds) == 2
    assert ds[0].pdb_id == "1abc"
    assert float(ds[0].y) == pytest.approx(5.0)
    assert ds[0].x.shape == (4, 36)
    assert ds[1].x.shape == (7, 36)
    assert not ds.skipped


def test_dataset_skips_missing(tmp_path: Path) -> None:
    cache = esm_cache_dir(tmp_path)
    _write_fake_esm(cache, "1abc", dim=8, seed=0)
    ligand_cache = {"1abc": _make_fake_ligand()}

    ds = PDBbindGraphDataset(
        pdb_ids=["1abc", "missing_esm", "missing_label"],
        labels={"1abc": 5.0, "missing_esm": 6.0},
        data_root=tmp_path,
        refined_dir=tmp_path / "raw" / "refined",
        ligand_cache=ligand_cache,
    )
    assert len(ds) == 1
    assert ds[0].pdb_id == "1abc"
    assert len(ds.skipped) == 2
    reasons = {r for _, r in ds.skipped}
    assert "missing esm cache" in reasons
    assert "missing label" in reasons


def test_dataloader_batches_esm_correctly(tmp_path: Path) -> None:
    """The dataloader must turn 3 ``(1, D)`` ESM tensors into a single ``(3, D)``."""
    cache = esm_cache_dir(tmp_path)
    pids = ["1abc", "2def", "3ghi"]
    for i, pid in enumerate(pids):
        _write_fake_esm(cache, pid, dim=8, seed=i)
    ligand_cache = {pid: _make_fake_ligand(n_atoms=4 + i) for i, pid in enumerate(pids)}

    ds = PDBbindGraphDataset(
        pdb_ids=pids,
        labels={pid: float(i) for i, pid in enumerate(pids)},
        data_root=tmp_path,
        refined_dir=tmp_path / "raw" / "refined",
        ligand_cache=ligand_cache,
    )
    loader = make_dataloader(ds, batch_size=3, shuffle=False, num_workers=0)
    batch = next(iter(loader))

    assert batch.esm_pocket.shape == (3, 8)
    assert batch.esm_whole.shape == (3, 8)
    assert batch.y.shape == (3,)
    # batch.x stacks per-atom features across all 3 graphs (4 + 5 + 6 = 15 nodes)
    assert batch.x.shape == (15, 36)
    assert batch.batch.shape == (15,)
    assert set(batch.batch.tolist()) == {0, 1, 2}


def test_load_ligand_graph_cache_missing_returns_empty(tmp_path: Path) -> None:
    blob = load_ligand_graph_cache(tmp_path)
    assert blob == {}


def test_load_ligand_graph_cache_roundtrip(tmp_path: Path) -> None:
    payload = {"1abc": _make_fake_ligand()}
    path = ligand_cache_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)

    loaded = load_ligand_graph_cache(tmp_path)
    assert "1abc" in loaded
    torch.testing.assert_close(loaded["1abc"]["x"], payload["1abc"]["x"])
