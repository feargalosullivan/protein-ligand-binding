"""Featurisation: ligand graphs, ECFP fingerprints, protein pocket embeddings.

This module is a stub in Phase 1 - the actual implementations land in Phase 2
once we have RDKit and ESM-2 in the environment. We define the *interfaces*
here so the rest of the codebase can import without circular weirdness later.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class LigandFeaturiser(Protocol):
    """Anything that turns an SDF / SMILES into a representation."""

    def from_sdf(self, sdf_path: Path) -> object: ...

    def from_smiles(self, smiles: str) -> object: ...


class ProteinFeaturiser(Protocol):
    """Anything that turns a protein (sequence or pocket) into a vector."""

    def embed_pocket(self, pdb_path: Path, ligand_sdf: Path) -> object: ...
