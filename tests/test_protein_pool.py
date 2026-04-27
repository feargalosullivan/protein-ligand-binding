"""Tests for the ESM pooling helper.

We don't load the actual ESM model here (that takes 150 MB + a download); the
``ESMEmbedder`` class is exercised via Phase 2's sanity notebook against a
real model. ``pool_pocket_embedding`` is pure numpy, so we test it directly.
"""

from __future__ import annotations

import numpy as np
import pytest

from plb.data.protein import (
    ESM_EMBED_DIMS,
    embed_dim_for,
    pool_pocket_embedding,
)

D = 480


def test_pool_basic_two_chains() -> None:
    chain_a = np.arange(10 * D, dtype=np.float32).reshape(10, D)
    chain_b = np.arange(5 * D, dtype=np.float32).reshape(5, D) + 100
    embeddings = {"A": chain_a, "B": chain_b}
    pocket_pos = {"A": [0, 1, 2], "B": []}

    pocket, whole = pool_pocket_embedding(embeddings, pocket_pos, embed_dim=D)

    expected_pocket = chain_a[:3].mean(axis=0)
    expected_whole = np.concatenate([chain_a, chain_b], axis=0).mean(axis=0)
    np.testing.assert_allclose(pocket, expected_pocket, rtol=1e-5)
    np.testing.assert_allclose(whole, expected_whole, rtol=1e-5)


def test_pool_falls_back_to_whole_when_no_pocket() -> None:
    """If no chain has any pocket residue, pocket pool == whole pool.

    This is the safety fallback - we never want a downstream model to receive
    an all-zero protein vector just because pocket detection failed.
    """
    chain_a = np.arange(8 * D, dtype=np.float32).reshape(8, D)
    embeddings = {"A": chain_a}
    pocket_pos = {"A": []}

    pocket, whole = pool_pocket_embedding(embeddings, pocket_pos, embed_dim=D)

    np.testing.assert_allclose(pocket, whole)


def test_pool_zero_when_all_inputs_empty() -> None:
    embeddings: dict[str, np.ndarray] = {}
    pocket_pos: dict[str, list[int]] = {}

    pocket, whole = pool_pocket_embedding(embeddings, pocket_pos, embed_dim=D)

    assert pocket.shape == (D,)
    assert whole.shape == (D,)
    assert np.all(pocket == 0.0)
    assert np.all(whole == 0.0)


def test_pool_dtype_is_float32() -> None:
    chain_a = np.arange(4 * D, dtype=np.float64).reshape(4, D)  # note float64 input
    embeddings = {"A": chain_a}
    pocket_pos = {"A": [0, 1]}

    pocket, whole = pool_pocket_embedding(embeddings, pocket_pos, embed_dim=D)

    assert pocket.dtype == np.float32
    assert whole.dtype == np.float32


def test_pool_pocket_spans_multiple_chains() -> None:
    chain_a = np.ones((3, D), dtype=np.float32)
    chain_b = np.ones((3, D), dtype=np.float32) * 5.0
    embeddings = {"A": chain_a, "B": chain_b}
    pocket_pos = {"A": [0], "B": [0, 1]}

    pocket, _ = pool_pocket_embedding(embeddings, pocket_pos, embed_dim=D)

    # Pool over [1, 5, 5] -> mean = 11/3
    np.testing.assert_allclose(pocket, np.full(D, 11.0 / 3.0, dtype=np.float32))


def test_embed_dim_for_known_models() -> None:
    for name, dim in ESM_EMBED_DIMS.items():
        assert embed_dim_for(name) == dim


def test_embed_dim_for_unknown_raises() -> None:
    with pytest.raises(KeyError):
        embed_dim_for("not-a-real-model")
