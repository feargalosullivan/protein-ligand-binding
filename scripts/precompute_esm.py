"""Pre-compute ESM-2 pocket-pooled embeddings for every PDBbind complex.

For each ``{pdb_id}_protein.pdb`` + ``{pdb_id}_ligand.sdf`` pair under
``data/raw/PDBbind_v2020_refined/refined-set/``:

1. Identify pocket residues within ``--cutoff`` angstroms of the ligand.
2. Run ESM-2 on each chain and assemble per-residue embeddings.
3. Mean-pool over (a) pocket residues across chains and (b) the whole protein.
4. Save ``data/cache/esm/{pdb_id}.npz`` with keys::

       pocket : (D,) float32 - mean over pocket residues
       whole  : (D,) float32 - mean over the whole protein
       n_pocket : int        - number of pocket residues found
       n_total  : int        - total number of standard residues
       cutoff   : float

Existing cache files are skipped, so the script is safely resumable.

Usage::

    python scripts/precompute_esm.py
    python scripts/precompute_esm.py --device cuda --model esm2_t12_35M_UR50D
    python scripts/precompute_esm.py --limit 50  # quick smoke test
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from plb.data.pdbbind import find_default_paths
from plb.data.pocket import pocket_from_pdb_files
from plb.data.protein import DEFAULT_MODEL, ESMEmbedder, embed_dim_for, pool_pocket_embedding

# PDBbind ligand SDFs frequently set the "2D" flag in their header but ship 3D
# conformers, which makes RDKit emit a warning per file. We trust PDBbind's
# coordinates either way, so suppress to avoid 5316 lines of noise.
try:
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.warning")
except ImportError:
    pass

log = logging.getLogger("precompute_esm")


def _complex_paths(refined_root: Path, pdb_id: str) -> tuple[Path, Path] | None:
    """Locate the ``_protein.pdb`` and ``_ligand.sdf`` for a complex.

    Returns ``None`` if either file is missing.
    """
    folder = refined_root / pdb_id
    protein = folder / f"{pdb_id}_protein.pdb"
    ligand = folder / f"{pdb_id}_ligand.sdf"
    if not protein.is_file() or not ligand.is_file():
        return None
    return protein, ligand


def _process_one(
    pdb_id: str,
    protein_pdb: Path,
    ligand_sdf: Path,
    embedder: ESMEmbedder,
    cutoff: float,
) -> dict[str, object]:
    pocket = pocket_from_pdb_files(protein_pdb, ligand_sdf, cutoff_angstrom=cutoff)
    chain_embeddings = {cid: embedder.embed_chain(seq) for cid, seq in pocket.chains.items()}
    pocket_vec, whole_vec = pool_pocket_embedding(
        chain_embeddings, pocket.pocket_residues, embedder.embed_dim
    )
    return {
        "pocket": pocket_vec,
        "whole": whole_vec,
        "n_pocket": np.int32(pocket.n_pocket_residues),
        "n_total": np.int32(pocket.n_total_residues),
        "cutoff": np.float32(cutoff),
    }


def main(
    data_root: Path,
    model_name: str,
    device: str,
    cutoff: float,
    limit: int | None,
    overwrite: bool,
) -> int:
    paths = find_default_paths(data_root)
    refined_root: Path = paths["refined_root"]
    if not refined_root.is_dir():
        log.error("refined-set directory not found at %s", refined_root)
        log.error("Run `python scripts/download_pdbbind.py` first.")
        return 1

    cache_dir = data_root / "cache" / "esm" / model_name
    cache_dir.mkdir(parents=True, exist_ok=True)
    errors_path = cache_dir / "_errors.jsonl"

    pdb_ids = sorted(p.name for p in refined_root.iterdir() if p.is_dir())
    if limit is not None:
        pdb_ids = pdb_ids[:limit]
    log.info("found %d complexes under %s", len(pdb_ids), refined_root)

    embedder = ESMEmbedder(model_name=model_name, device=device)
    embed_dim_for(model_name)  # validate model name early

    n_done, n_skipped, n_errors = 0, 0, 0
    with errors_path.open("a", encoding="utf-8") as err_log:
        for pdb_id in tqdm(pdb_ids, desc=f"ESM[{model_name}]", unit="cmplx"):
            cache_path = cache_dir / f"{pdb_id}.npz"
            if cache_path.exists() and not overwrite:
                n_skipped += 1
                continue

            located = _complex_paths(refined_root, pdb_id)
            if located is None:
                err_log.write(
                    json.dumps({"pdb_id": pdb_id, "error": "missing protein/ligand files"}) + "\n"
                )
                n_errors += 1
                continue
            protein_pdb, ligand_sdf = located

            try:
                result = _process_one(pdb_id, protein_pdb, ligand_sdf, embedder, cutoff)
                np.savez_compressed(cache_path, **result)
                n_done += 1
            except Exception as exc:  # noqa: BLE001 - intentional, want to keep going
                err_log.write(json.dumps({"pdb_id": pdb_id, "error": repr(exc)}) + "\n")
                n_errors += 1

    log.info(
        "done: %d new, %d cached, %d errors -> %s",
        n_done,
        n_skipped,
        n_errors,
        cache_dir,
    )
    return 0 if n_errors < max(10, len(pdb_ids) // 100) else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--model", default=DEFAULT_MODEL, help="fair-esm model name")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--cutoff", type=float, default=6.0, help="pocket cutoff (angstroms)")
    parser.add_argument(
        "--limit", type=int, default=None, help="process only the first N complexes"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="re-compute even if a cache file already exists",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    sys.exit(
        main(
            data_root=args.data_root,
            model_name=args.model,
            device=args.device,
            cutoff=args.cutoff,
            limit=args.limit,
            overwrite=args.overwrite,
        )
    )
