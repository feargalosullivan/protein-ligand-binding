"""Unit tests for the pocket-residue identification.

The geometry core (``_pocket_residue_mask``) is pure numpy and tested without
any structural dependencies. The Biopython integration is exercised against a
small in-memory PDB string written to a temp file, so the test suite stays
self-contained.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from plb.data.pocket import _pocket_residue_mask


def test_pocket_mask_simple() -> None:
    """3 residues, ligand sits next to residue 1 only."""
    # Residue 0: 2 atoms far away. Residue 1: 1 atom near. Residue 2: 1 atom far.
    protein_atom_coords = np.asarray(
        [
            [10.0, 10.0, 10.0],  # residue 0
            [11.0, 10.0, 10.0],  # residue 0
            [0.5, 0.0, 0.0],  # residue 1, very near origin
            [20.0, 0.0, 0.0],  # residue 2
        ],
        dtype=np.float32,
    )
    protein_atom_residue_idx = np.asarray([0, 0, 1, 2], dtype=np.int64)
    ligand_coords = np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32)

    mask = _pocket_residue_mask(
        protein_atom_coords, protein_atom_residue_idx, ligand_coords, cutoff=5.0
    )

    assert mask.tolist() == [False, True, False]


def test_pocket_mask_inclusive_at_cutoff() -> None:
    """A residue exactly at the cutoff is included (<=, not <)."""
    protein_atom_coords = np.asarray([[6.0, 0.0, 0.0]], dtype=np.float32)
    protein_atom_residue_idx = np.asarray([0], dtype=np.int64)
    ligand_coords = np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32)

    mask = _pocket_residue_mask(
        protein_atom_coords, protein_atom_residue_idx, ligand_coords, cutoff=6.0
    )

    assert mask.tolist() == [True]


def test_pocket_mask_just_outside_cutoff() -> None:
    """A residue just past the cutoff is excluded."""
    protein_atom_coords = np.asarray([[6.001, 0.0, 0.0]], dtype=np.float32)
    protein_atom_residue_idx = np.asarray([0], dtype=np.int64)
    ligand_coords = np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32)

    mask = _pocket_residue_mask(
        protein_atom_coords, protein_atom_residue_idx, ligand_coords, cutoff=6.0
    )

    assert mask.tolist() == [False]


def test_pocket_mask_uses_minimum_atom_distance() -> None:
    """A residue must have ANY heavy atom in range, not all of them.

    The pocket definition is "any heavy atom within cutoff", not "centroid
    within cutoff". This test would fail if we used centroids.
    """
    # Residue 0 has atoms scattered: one close, others far.
    protein_atom_coords = np.asarray(
        [
            [0.5, 0.0, 0.0],  # close
            [50.0, 0.0, 0.0],  # very far
            [60.0, 0.0, 0.0],  # very far
        ],
        dtype=np.float32,
    )
    protein_atom_residue_idx = np.asarray([0, 0, 0], dtype=np.int64)
    ligand_coords = np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32)

    mask = _pocket_residue_mask(
        protein_atom_coords, protein_atom_residue_idx, ligand_coords, cutoff=2.0
    )

    assert mask.tolist() == [True]


def test_pocket_mask_empty_inputs() -> None:
    empty = np.empty((0, 3), dtype=np.float32)
    empty_idx = np.empty((0,), dtype=np.int64)
    mask = _pocket_residue_mask(empty, empty_idx, empty, cutoff=5.0)
    assert mask.shape == (0,)


def test_pocket_mask_rejects_negative_cutoff() -> None:
    with pytest.raises(ValueError):
        _pocket_residue_mask(
            np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
            np.asarray([0], dtype=np.int64),
            np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
            cutoff=-1.0,
        )


# --- Integration test against a tiny synthetic PDB + SDF ---------------------


SYNTHETIC_PDB = """\
HEADER    UNIT TEST
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 20.00           N
ATOM      2  CA  ALA A   1       1.500   0.000   0.000  1.00 20.00           C
ATOM      3  C   ALA A   1       2.000   1.500   0.000  1.00 20.00           C
ATOM      4  O   ALA A   1       3.000   2.000   0.000  1.00 20.00           O
ATOM      5  N   GLY A   2       1.500   2.500   0.000  1.00 20.00           N
ATOM      6  CA  GLY A   2       2.000   4.000   0.000  1.00 20.00           C
ATOM      7  C   GLY A   2       3.500   4.500   0.000  1.00 20.00           C
ATOM      8  O   GLY A   2       4.000   5.500   0.000  1.00 20.00           O
ATOM      9  N   VAL A   3      20.000  20.000  20.000  1.00 20.00           N
ATOM     10  CA  VAL A   3      21.500  20.000  20.000  1.00 20.00           C
ATOM     11  C   VAL A   3      22.000  21.500  20.000  1.00 20.00           C
ATOM     12  O   VAL A   3      23.000  22.000  20.000  1.00 20.00           O
END
"""

# A trivial single-atom "ligand" placed right next to residue ALA A 1 / GLY A 2
# but far from VAL A 3.
SYNTHETIC_SDF = """\
ligand
     RDKit          3D

  1  0  0  0  0  0  0  0  0  0999 V2000
    1.0000    1.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
M  END
$$$$
"""


def test_pocket_from_pdb_files_integration(tmp_path: Path) -> None:
    """End-to-end: tiny in-memory protein + ligand -> correct pocket residues."""
    pytest.importorskip("Bio")
    pytest.importorskip("rdkit")

    from plb.data.pocket import pocket_from_pdb_files

    pdb_path = tmp_path / "tiny.pdb"
    sdf_path = tmp_path / "tiny.sdf"
    pdb_path.write_text(SYNTHETIC_PDB, encoding="utf-8")
    sdf_path.write_text(SYNTHETIC_SDF, encoding="utf-8")

    pocket = pocket_from_pdb_files(pdb_path, sdf_path, cutoff_angstrom=6.0)

    assert pocket.chain_ids == ["A"]
    assert pocket.chains["A"] == "AGV"  # ALA, GLY, VAL
    # ALA (residue 0) and GLY (residue 1) are within 6A of (1, 1, 0); VAL (2)
    # is not. Order of positions follows residue order in the chain.
    assert pocket.pocket_residues["A"] == [0, 1]
    assert pocket.n_total_residues == 3
    assert pocket.cutoff_angstrom == 6.0
