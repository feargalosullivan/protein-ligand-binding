"""Pre-compute ligand graphs for every PDBbind complex into a single ``.pt`` file.

The Phase 4 GNN re-parses the same ~5000 SDF files every training run unless
we cache them. This script reads each ``{pdb_id}_ligand.sdf`` once with
RDKit, converts to a torch-friendly dict, and saves the result to
``data/cache/ligand_graphs.pt`` (a flat ``{pdb_id: dict}`` mapping).

Usage::

    python scripts/precompute_ligand_graphs.py
    python scripts/precompute_ligand_graphs.py --limit 50  # smoke test
    python scripts/precompute_ligand_graphs.py --overwrite
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
from tqdm.auto import tqdm

from plb.data.dataset import _ligand_dict_from_sdf, _refined_sdf_path, ligand_cache_path
from plb.data.pdbbind import find_default_paths

try:
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.warning")
except ImportError:
    pass

log = logging.getLogger("precompute_ligand_graphs")


def main(data_root: Path, limit: int | None, overwrite: bool) -> int:
    paths = find_default_paths(data_root)
    refined_root: Path = paths["refined_root"]
    if not refined_root.is_dir():
        log.error("refined-set directory not found at %s", refined_root)
        log.error("Run `python scripts/download_pdbbind.py` first.")
        return 1

    cache_path = ligand_cache_path(data_root)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    cache: dict[str, dict] = {}
    if cache_path.is_file() and not overwrite:
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        log.info("loaded existing cache with %d entries from %s", len(cache), cache_path)

    pdb_ids = sorted(p.name for p in refined_root.iterdir() if p.is_dir())
    if limit is not None:
        pdb_ids = pdb_ids[:limit]
    log.info("found %d complexes under %s", len(pdb_ids), refined_root)

    n_new, n_skipped, n_errors = 0, 0, 0
    errors: list[tuple[str, str]] = []
    for pdb_id in tqdm(pdb_ids, desc="ligand graphs", unit="cmplx"):
        if pdb_id in cache and not overwrite:
            n_skipped += 1
            continue
        sdf = _refined_sdf_path(refined_root, pdb_id)
        if not sdf.is_file():
            errors.append((pdb_id, "missing sdf"))
            n_errors += 1
            continue
        try:
            cache[pdb_id] = _ligand_dict_from_sdf(sdf)
            n_new += 1
        except Exception as exc:  # noqa: BLE001 - keep going
            errors.append((pdb_id, repr(exc)))
            n_errors += 1

    torch.save(cache, cache_path)
    log.info(
        "done: %d new, %d cached, %d errors; total %d -> %s",
        n_new,
        n_skipped,
        n_errors,
        len(cache),
        cache_path,
    )
    if errors:
        log.warning("first 5 errors: %s", errors[:5])
    return 0 if n_errors < max(10, len(pdb_ids) // 100) else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    sys.exit(main(data_root=args.data_root, limit=args.limit, overwrite=args.overwrite))
