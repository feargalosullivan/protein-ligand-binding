"""Tests for plb.data.cache - the ESM embedding cache loader."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from plb.data.cache import (
    CachedEmbedding,
    esm_cache_dir,
    load_esm_embedding,
    load_esm_embedding_matrix,
)


def _write_fake(cache_dir: Path, pdb_id: str, dim: int = 8, seed: int = 0) -> np.ndarray:
    """Write a fake .npz with deterministic content. Returns the pocket vector."""
    rng = np.random.default_rng(seed)
    pocket = rng.standard_normal(dim).astype(np.float32)
    whole = rng.standard_normal(dim).astype(np.float32)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_dir / f"{pdb_id}.npz",
        pocket=pocket,
        whole=whole,
        n_pocket=np.int32(17),
        n_total=np.int32(123),
        cutoff=np.float32(6.0),
    )
    return pocket


def test_esm_cache_dir_layout(tmp_path: Path) -> None:
    p = esm_cache_dir(tmp_path, "esm2_t12_35M_UR50D")
    # Check that the relative segment matches the layout produced by
    # scripts/precompute_esm.py - no surprises if either side moves.
    assert p == tmp_path / "cache" / "esm" / "esm2_t12_35M_UR50D"


def test_load_single(tmp_path: Path) -> None:
    cache = esm_cache_dir(tmp_path)
    pocket = _write_fake(cache, "1abc", dim=480)
    emb = load_esm_embedding("1abc", data_root=tmp_path)

    assert isinstance(emb, CachedEmbedding)
    assert emb.pdb_id == "1abc"
    assert emb.embed_dim == 480
    assert emb.pocket.dtype == np.float32
    assert emb.whole.dtype == np.float32
    assert emb.n_pocket == 17
    assert emb.n_total == 123
    assert emb.cutoff == pytest.approx(6.0)
    np.testing.assert_array_equal(emb.pocket, pocket)


def test_load_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no cached ESM embedding"):
        load_esm_embedding("9zzz", data_root=tmp_path)


def test_matrix_shape_and_order(tmp_path: Path) -> None:
    cache = esm_cache_dir(tmp_path)
    for i, pid in enumerate(["1abc", "2def", "3ghi"]):
        _write_fake(cache, pid, dim=4, seed=i)

    X, kept = load_esm_embedding_matrix(["1abc", "2def", "3ghi"], data_root=tmp_path)
    assert X.shape == (3, 8)  # pocket(4) + whole(4)
    assert X.dtype == np.float32
    assert kept == ["1abc", "2def", "3ghi"]

    # Order must match input order (not alphabetical from the cache directory).
    X_rev, kept_rev = load_esm_embedding_matrix(["3ghi", "1abc"], data_root=tmp_path)
    assert kept_rev == ["3ghi", "1abc"]
    np.testing.assert_array_equal(X_rev[0], X[2])
    np.testing.assert_array_equal(X_rev[1], X[0])


def test_matrix_pocket_only(tmp_path: Path) -> None:
    cache = esm_cache_dir(tmp_path)
    _write_fake(cache, "1abc", dim=4, seed=0)
    X, _ = load_esm_embedding_matrix(["1abc"], data_root=tmp_path, columns=("pocket",))
    assert X.shape == (1, 4)


def test_matrix_unknown_column_raises(tmp_path: Path) -> None:
    cache = esm_cache_dir(tmp_path)
    _write_fake(cache, "1abc", dim=4, seed=0)
    with pytest.raises(ValueError, match="unknown columns"):
        load_esm_embedding_matrix(["1abc"], data_root=tmp_path, columns=("bogus",))


def test_matrix_skip_missing(tmp_path: Path) -> None:
    cache = esm_cache_dir(tmp_path)
    _write_fake(cache, "1abc", dim=4, seed=0)
    _write_fake(cache, "2def", dim=4, seed=1)

    X, kept = load_esm_embedding_matrix(
        ["1abc", "9zzz", "2def"], data_root=tmp_path, skip_missing=True
    )
    assert X.shape == (2, 8)
    assert kept == ["1abc", "2def"]


def test_matrix_strict_missing_raises(tmp_path: Path) -> None:
    cache = esm_cache_dir(tmp_path)
    _write_fake(cache, "1abc", dim=4, seed=0)

    with pytest.raises(FileNotFoundError):
        load_esm_embedding_matrix(["1abc", "9zzz"], data_root=tmp_path)


def test_matrix_empty_columns_raises(tmp_path: Path) -> None:
    cache = esm_cache_dir(tmp_path)
    _write_fake(cache, "1abc", dim=4, seed=0)
    with pytest.raises(ValueError, match="columns must be non-empty"):
        load_esm_embedding_matrix(["1abc"], data_root=tmp_path, columns=())


def test_matrix_all_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no embeddings loaded"):
        load_esm_embedding_matrix(["1abc", "2def"], data_root=tmp_path, skip_missing=True)
