"""Ligand featurisation: 2D molecular graphs and ECFP fingerprints via RDKit."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

ATOM_TYPES: list[str] = [
    "C",
    "N",
    "O",
    "F",
    "P",
    "S",
    "Cl",
    "Br",
    "I",
    "B",
    "Si",
    "Se",
    "OTHER",
]
DEGREES: list[int] = [0, 1, 2, 3, 4, 5]
NUM_HS: list[int] = [0, 1, 2, 3, 4]
HYBRIDISATIONS: list[str] = ["SP", "SP2", "SP3", "SP3D", "SP3D2", "OTHER"]
BOND_TYPES: list[str] = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]
CHIRALITIES: list[str] = ["R", "S", "NONE"]


@dataclass(frozen=True)
class LigandGraph:
    """Undirected 2D molecular graph (numpy arrays, framework-agnostic)."""

    node_feats: np.ndarray
    edge_index: np.ndarray
    edge_feats: np.ndarray
    smiles: str

    @property
    def num_atoms(self) -> int:
        return int(self.node_feats.shape[0])

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1])


def feature_dims() -> dict[str, int]:
    """Fixed node/edge feature dimensions (for building the model without data)."""
    n_atom = (
        len(ATOM_TYPES)
        + len(DEGREES)
        + 1  # formal charge (scalar)
        + len(NUM_HS)
        + len(HYBRIDISATIONS)
        + 1  # aromatic
        + 1  # in ring
        + len(CHIRALITIES)
    )
    n_bond = len(BOND_TYPES) + 4  # bond type + conjugated + in-ring + E + Z
    return {"node": n_atom, "edge": n_bond}


def _onehot(value: object, vocab: list) -> list[float]:
    if value not in vocab:
        value = vocab[-1]
    return [1.0 if v == value else 0.0 for v in vocab]


def _atom_features(atom) -> np.ndarray:
    from rdkit.Chem import ChiralType, HybridizationType

    hyb_map = {
        HybridizationType.SP: "SP",
        HybridizationType.SP2: "SP2",
        HybridizationType.SP3: "SP3",
        HybridizationType.SP3D: "SP3D",
        HybridizationType.SP3D2: "SP3D2",
    }
    chir_map = {
        ChiralType.CHI_TETRAHEDRAL_CW: "R",
        ChiralType.CHI_TETRAHEDRAL_CCW: "S",
    }

    feats: list[float] = []
    feats += _onehot(atom.GetSymbol(), ATOM_TYPES)
    feats += _onehot(min(atom.GetDegree(), DEGREES[-1]), DEGREES)
    feats.append(float(atom.GetFormalCharge()))
    feats += _onehot(min(atom.GetTotalNumHs(), NUM_HS[-1]), NUM_HS)
    feats += _onehot(hyb_map.get(atom.GetHybridization(), "OTHER"), HYBRIDISATIONS)
    feats.append(1.0 if atom.GetIsAromatic() else 0.0)
    feats.append(1.0 if atom.IsInRing() else 0.0)
    feats += _onehot(chir_map.get(atom.GetChiralTag(), "NONE"), CHIRALITIES)
    return np.asarray(feats, dtype=np.float32)


def _bond_features(bond) -> np.ndarray:
    from rdkit.Chem import BondType

    bt_map = {
        BondType.SINGLE: "SINGLE",
        BondType.DOUBLE: "DOUBLE",
        BondType.TRIPLE: "TRIPLE",
        BondType.AROMATIC: "AROMATIC",
    }
    feats: list[float] = []
    feats += _onehot(bt_map.get(bond.GetBondType(), "SINGLE"), BOND_TYPES)
    feats.append(1.0 if bond.GetIsConjugated() else 0.0)
    feats.append(1.0 if bond.IsInRing() else 0.0)
    stereo = str(bond.GetStereo())
    feats.append(1.0 if "STEREOE" in stereo else 0.0)
    feats.append(1.0 if "STEREOZ" in stereo else 0.0)
    return np.asarray(feats, dtype=np.float32)


def ligand_graph_from_mol(mol) -> LigandGraph:
    from rdkit import Chem

    if mol is None:
        raise ValueError("mol is None")
    if mol.GetNumAtoms() == 0:
        raise ValueError("mol has zero atoms")

    node_feats = np.stack([_atom_features(a) for a in mol.GetAtoms()])

    src: list[int] = []
    dst: list[int] = []
    edge_feats_list: list[np.ndarray] = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bf = _bond_features(bond)
        src.append(i)
        dst.append(j)
        edge_feats_list.append(bf)
        src.append(j)
        dst.append(i)
        edge_feats_list.append(bf)

    if edge_feats_list:
        edge_index = np.asarray([src, dst], dtype=np.int64)
        edge_feats = np.stack(edge_feats_list)
    else:
        n_bond_feats = feature_dims()["edge"]
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_feats = np.empty((0, n_bond_feats), dtype=np.float32)

    return LigandGraph(
        node_feats=node_feats,
        edge_index=edge_index,
        edge_feats=edge_feats,
        smiles=Chem.MolToSmiles(mol),
    )


def ligand_graph_from_smiles(smiles: str) -> LigandGraph:
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"could not parse SMILES: {smiles!r}")
    return ligand_graph_from_mol(mol)


def ligand_graph_from_sdf(sdf_path: Path | str, sanitize: bool = True) -> LigandGraph:
    from rdkit import Chem

    sdf_path = Path(sdf_path)
    if not sdf_path.is_file():
        raise FileNotFoundError(sdf_path)
    suppl = Chem.SDMolSupplier(str(sdf_path), sanitize=sanitize, removeHs=True)
    mol = next((m for m in suppl if m is not None), None)
    if mol is None:
        raise ValueError(f"no valid molecule in {sdf_path}")
    return ligand_graph_from_mol(mol)


def ecfp_fingerprint(
    smiles_or_mol: str | object,
    radius: int = 2,
    n_bits: int = 2048,
) -> np.ndarray:
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator
    from rdkit.DataStructs import ConvertToNumpyArray

    if isinstance(smiles_or_mol, str):
        mol = Chem.MolFromSmiles(smiles_or_mol)
        if mol is None:
            raise ValueError(f"could not parse SMILES: {smiles_or_mol!r}")
    else:
        mol = smiles_or_mol
        if mol is None:
            raise ValueError("mol is None")

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fp = gen.GetFingerprint(mol)
    arr = np.zeros((n_bits,), dtype=np.uint8)
    ConvertToNumpyArray(fp, arr)
    return arr


def ecfp_from_sdf(
    sdf_path: Path | str,
    radius: int = 2,
    n_bits: int = 2048,
) -> np.ndarray:
    from rdkit import Chem

    sdf_path = Path(sdf_path)
    if not sdf_path.is_file():
        raise FileNotFoundError(sdf_path)

    for sanitize in (True, False):
        suppl = Chem.SDMolSupplier(str(sdf_path), sanitize=sanitize, removeHs=True)
        mol = next((m for m in suppl if m is not None), None)
        if mol is not None:
            return ecfp_fingerprint(mol, radius=radius, n_bits=n_bits)

    raise ValueError(f"no valid molecule in {sdf_path}")
