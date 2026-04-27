"""Pocket residue identification: which residues are within R angstrom of the ligand.

The expensive heavy lifting (parsing PDB, building atom arrays) goes through
Biopython. The actual distance calculation is a pure-numpy function so we can
unit-test it without any structural files.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Three-letter -> one-letter residue mapping. Includes a few common modified
# residues that appear in PDBbind cleaned structures (selenomethionine, etc.).
THREE_TO_ONE: dict[str, str] = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    # Common modified / protonated forms folded back to canonical residues.
    "MSE": "M",
    "SEC": "U",
    "PYL": "O",
    "HIP": "H",
    "HID": "H",
    "HIE": "H",
    "CYX": "C",
    "CYM": "C",
}


@dataclass(frozen=True)
class Pocket:
    """A protein binding pocket extracted from a PDBbind complex.

    Attributes
    ----------
    chains
        ``chain_id -> one-letter amino-acid sequence`` for every chain that
        contains at least one standard residue.
    pocket_residues
        ``chain_id -> list of 0-indexed positions in the chain's sequence``
        identifying which residues are within ``cutoff_angstrom`` of any heavy
        atom of the ligand.
    pdb_ids
        ``chain_id -> [(resseq, icode), ...]`` parallel to ``chains``, for
        traceability back to the original PDB numbering.
    cutoff_angstrom
        The cutoff used to define the pocket.
    """

    chains: dict[str, str]
    pocket_residues: dict[str, list[int]]
    pdb_ids: dict[str, list[tuple[int, str]]]
    cutoff_angstrom: float

    @property
    def n_pocket_residues(self) -> int:
        return sum(len(v) for v in self.pocket_residues.values())

    @property
    def n_total_residues(self) -> int:
        return sum(len(s) for s in self.chains.values())

    @property
    def chain_ids(self) -> list[str]:
        return sorted(self.chains.keys())


def _pocket_residue_mask(
    protein_atom_coords: np.ndarray,
    protein_atom_residue_idx: np.ndarray,
    ligand_coords: np.ndarray,
    cutoff: float,
) -> np.ndarray:
    """Return a boolean mask over residues marking those near the ligand.

    Pure-numpy core function so the geometry can be tested without any PDB I/O.

    Parameters
    ----------
    protein_atom_coords
        ``(n_atoms, 3)`` float array of heavy-atom xyz coordinates.
    protein_atom_residue_idx
        ``(n_atoms,)`` integer array assigning each atom to a residue index in
        ``[0, n_residues)``. Indices must be contiguous starting at 0.
    ligand_coords
        ``(n_lig_atoms, 3)`` float array of ligand heavy-atom coordinates.
    cutoff
        Distance cutoff in angstroms.

    Returns
    -------
    Boolean mask of shape ``(n_residues,)`` where ``True`` means the residue
    has at least one heavy atom within ``cutoff`` of any ligand heavy atom.
    """
    if protein_atom_coords.size == 0 or ligand_coords.size == 0:
        return np.zeros(0, dtype=bool)
    if cutoff <= 0:
        raise ValueError(f"cutoff must be positive; got {cutoff}")

    # Pairwise distances. For typical sizes (~3000 protein atoms x ~50 ligand
    # atoms) this is ~150k floats; fine. For huge systems consider scipy KDTree.
    diffs = protein_atom_coords[:, None, :] - ligand_coords[None, :, :]
    min_dists_sq = (diffs * diffs).sum(axis=-1).min(axis=1)
    near_atoms = min_dists_sq <= cutoff * cutoff

    n_residues = int(protein_atom_residue_idx.max()) + 1
    mask = np.zeros(n_residues, dtype=bool)
    mask[protein_atom_residue_idx[near_atoms]] = True
    return mask


def pocket_from_pdb_files(
    protein_pdb: Path | str,
    ligand_sdf: Path | str,
    cutoff_angstrom: float = 6.0,
) -> Pocket:
    """Identify pocket residues using Biopython for the protein + RDKit for the ligand.

    Heavy atoms only (hydrogens are ignored). Non-standard residues, waters,
    and HETATM lines are excluded from the protein sequence.
    """
    from Bio.PDB import PDBParser
    from rdkit import Chem

    protein_pdb = Path(protein_pdb)
    ligand_sdf = Path(ligand_sdf)

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", str(protein_pdb))

    # We only consider model 0 (PDBbind cleaned files have a single model).
    model = next(iter(structure))

    chain_residues: dict[str, list[tuple[int, str, str, np.ndarray]]] = {}
    for chain in model:
        cid = chain.id if chain.id.strip() else "A"
        for residue in chain:
            hetflag, resseq, icode = residue.get_id()
            if hetflag.strip():  # waters, ligands, etc.
                continue
            resname = residue.get_resname()
            if resname not in THREE_TO_ONE:
                continue
            heavy_coords = np.asarray(
                [a.get_coord() for a in residue if a.element != "H"],
                dtype=np.float32,
            )
            if len(heavy_coords) == 0:
                continue
            chain_residues.setdefault(cid, []).append((resseq, icode, resname, heavy_coords))

    if not chain_residues:
        raise ValueError(f"no standard residues found in {protein_pdb}")

    # Parse the ligand and pull out its heavy-atom coordinates.
    suppl = Chem.SDMolSupplier(str(ligand_sdf), sanitize=False, removeHs=True)
    ligand_mol = next((m for m in suppl if m is not None), None)
    if ligand_mol is None:
        raise ValueError(f"could not load ligand from {ligand_sdf}")
    if ligand_mol.GetNumConformers() == 0:
        raise ValueError(f"ligand SDF has no 3D conformer: {ligand_sdf}")
    conf = ligand_mol.GetConformer()
    ligand_coords = np.asarray(
        [list(conf.GetAtomPosition(i)) for i in range(ligand_mol.GetNumAtoms())],
        dtype=np.float32,
    )

    chains_seq: dict[str, str] = {}
    pocket_pos: dict[str, list[int]] = {}
    pdb_ids: dict[str, list[tuple[int, str]]] = {}

    for cid, residues in chain_residues.items():
        coord_blocks = [r[3] for r in residues]
        residue_indices = np.concatenate(
            [np.full(len(c), i, dtype=np.int64) for i, c in enumerate(coord_blocks)]
        )
        protein_atom_coords = np.concatenate(coord_blocks).astype(np.float32)

        mask = _pocket_residue_mask(
            protein_atom_coords=protein_atom_coords,
            protein_atom_residue_idx=residue_indices,
            ligand_coords=ligand_coords,
            cutoff=cutoff_angstrom,
        )

        chains_seq[cid] = "".join(THREE_TO_ONE[r[2]] for r in residues)
        pocket_pos[cid] = np.where(mask)[0].tolist()
        pdb_ids[cid] = [(int(r[0]), str(r[1])) for r in residues]

    return Pocket(
        chains=chains_seq,
        pocket_residues=pocket_pos,
        pdb_ids=pdb_ids,
        cutoff_angstrom=cutoff_angstrom,
    )
