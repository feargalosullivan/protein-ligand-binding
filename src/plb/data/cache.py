"""Loader for the ESM embedding npz cache produced by scripts/precompute_esm.py.

The cache lives at::

    data/cache/esm/{model_name}/{pdb_id}.npz

and each file holds the keys ``pocket``, ``whole``, ``n_pocket``, ``n_total``,
``cutoff`` (see ``scripts/precompute_esm.py`` for the schema). This module just
provides convenient typed accessors and a vectorised loader that stacks many
complexes into a single matrix (handy for building XGBoost feature tables).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_ESM_MODEL = "esm2_t12_35M_UR50D"


@dataclass(frozen=True)
class CachedEmbedding:
    """Single complex's pocket + whole-protein ESM embeddings."""

    pdb_id: str
    pocket: np.ndarray  # (D,) float32
    whole: np.ndarray  # (D,) float32
    n_pocket: int
    n_total: int
    cutoff: float

    @property
    def embed_dim(self) -> int:
        return int(self.pocket.shape[0])


def esm_cache_dir(data_root: Path | str, model_name: str = DEFAULT_ESM_MODEL) -> Path:
    """Canonical path of the ESM cache directory for a given model."""
    return Path(data_root) / "cache" / "esm" / model_name


def _load_one(cache_dir: Path, pdb_id: str) -> CachedEmbedding:
    path = cache_dir / f"{pdb_id}.npz"
    if not path.is_file():
        raise FileNotFoundError(
            f"no cached ESM embedding for {pdb_id} at {path}. "
            "Run `python scripts/precompute_esm.py` first."
        )
    with np.load(path) as data:
        return CachedEmbedding(
            pdb_id=pdb_id,
            pocket=np.asarray(data["pocket"], dtype=np.float32),
            whole=np.asarray(data["whole"], dtype=np.float32),
            n_pocket=int(data["n_pocket"]),
            n_total=int(data["n_total"]),
            cutoff=float(data["cutoff"]),
        )


def load_esm_embedding(
    pdb_id: str,
    data_root: Path | str = "data",
    model_name: str = DEFAULT_ESM_MODEL,
) -> CachedEmbedding:
    """Load the cached pocket + whole-protein embedding for a single complex."""
    return _load_one(esm_cache_dir(data_root, model_name), pdb_id)


def load_esm_embedding_matrix(
    pdb_ids: Iterable[str],
    data_root: Path | str = "data",
    model_name: str = DEFAULT_ESM_MODEL,
    columns: Sequence[str] = ("pocket", "whole"),
    skip_missing: bool = False,
) -> tuple[np.ndarray, list[str]]:
    """Stack many cached embeddings into a single ``(N, D_total)`` feature matrix.

    Parameters
    ----------
    pdb_ids
        PDB IDs to load (in input order).
    data_root, model_name
        Used to find the cache directory; see ``esm_cache_dir``.
    columns
        Subset of ``("pocket", "whole")`` to concatenate per row. Order is
        preserved in the output. Default is both, giving a ``(N, 2D)`` matrix.
    skip_missing
        If ``True``, silently drop PDB IDs without a cache file. If ``False``
        (default), raise ``FileNotFoundError`` on the first miss.

    Returns
    -------
    X
        ``(N, D_total)`` float32 array. ``D_total`` is ``D * len(columns)``
        where ``D`` is the model's per-residue embedding dim.
    kept_ids
        The subset of ``pdb_ids`` that were actually loaded (equal to
        ``pdb_ids`` unless ``skip_missing=True`` and some were absent).
    """
    if not columns:
        raise ValueError("columns must be non-empty")
    valid = {"pocket", "whole"}
    bad = set(columns) - valid
    if bad:
        raise ValueError(f"unknown columns {sorted(bad)}; expected subset of {sorted(valid)}")

    cache_dir = esm_cache_dir(data_root, model_name)
    rows: list[np.ndarray] = []
    kept: list[str] = []

    for pdb_id in pdb_ids:
        path = cache_dir / f"{pdb_id}.npz"
        if not path.is_file():
            if skip_missing:
                continue
            raise FileNotFoundError(
                f"no cached ESM embedding for {pdb_id} at {path}. "
                "Run `python scripts/precompute_esm.py` first."
            )
        emb = _load_one(cache_dir, pdb_id)
        parts = [getattr(emb, col) for col in columns]
        rows.append(np.concatenate(parts, axis=0))
        kept.append(pdb_id)

    if not rows:
        raise ValueError("no embeddings loaded (empty input or all missing)")

    return np.stack(rows, axis=0).astype(np.float32, copy=False), kept


__all__ = [
    "DEFAULT_ESM_MODEL",
    "CachedEmbedding",
    "esm_cache_dir",
    "load_esm_embedding",
    "load_esm_embedding_matrix",
]
