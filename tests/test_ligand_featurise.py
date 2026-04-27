"""Unit tests for ligand featurisation against known small molecules.

These tests need RDKit. We skip the whole module gracefully if RDKit isn't
installed, so the test suite stays runnable in a minimal dev environment.
"""

from __future__ import annotations

import numpy as np
import pytest

rdkit = pytest.importorskip("rdkit")  # noqa: F841 - just gating the module

from plb.data.ligand import (  # noqa: E402 - after importorskip
    LigandGraph,
    ecfp_fingerprint,
    feature_dims,
    ligand_graph_from_mol,
    ligand_graph_from_smiles,
)

ASPIRIN_SMILES = "CC(=O)Oc1ccccc1C(=O)O"
BENZENE_SMILES = "c1ccccc1"
METHANE_SMILES = "C"


def test_feature_dims_consistent_with_real_atoms() -> None:
    """The advertised dim must equal what the featuriser actually emits."""
    dims = feature_dims()
    g = ligand_graph_from_smiles(ASPIRIN_SMILES)
    assert g.node_feats.shape[1] == dims["node"]
    assert g.edge_feats.shape[1] == dims["edge"]


def test_aspirin_atom_count() -> None:
    """Aspirin (C9H8O4) has 13 heavy atoms (9 C + 4 O); H is implicit."""
    g = ligand_graph_from_smiles(ASPIRIN_SMILES)
    assert g.num_atoms == 13


def test_benzene_is_fully_aromatic_and_in_ring() -> None:
    g = ligand_graph_from_smiles(BENZENE_SMILES)
    dims = feature_dims()

    assert g.num_atoms == 6
    # Benzene has 6 aromatic bonds, doubled for undirected => 12 edges.
    assert g.num_edges == 12

    # The aromatic flag is a fixed offset inside the node feature vector. We
    # don't hardcode the offset here (would couple the test to feature order);
    # instead, we sum across all atoms for that flag and check it equals 6.
    aromatic_per_atom = g.node_feats.sum(axis=0)
    assert np.any(aromatic_per_atom == 6), (
        "expected at least one node-feature column to be all-1 for benzene "
        "(aromatic + in-ring would both qualify)"
    )

    # Edge features: every benzene bond is aromatic and in-ring.
    assert g.edge_feats.shape == (12, dims["edge"])
    # No bond is single/double/triple - they're all aromatic. The first 4
    # columns of edge feats are the bond-type one-hot.
    bond_type_block = g.edge_feats[:, :4]
    assert np.allclose(bond_type_block.sum(axis=0), [0, 0, 0, 12])


def test_methane_has_no_edges() -> None:
    g = ligand_graph_from_smiles(METHANE_SMILES)
    assert g.num_atoms == 1
    assert g.num_edges == 0
    assert g.edge_index.shape == (2, 0)
    assert g.edge_feats.shape[0] == 0


def test_edge_index_is_undirected() -> None:
    """For every directed edge (i, j) there must be a (j, i) counterpart."""
    g = ligand_graph_from_smiles(ASPIRIN_SMILES)
    edges = {(int(i), int(j)) for i, j in g.edge_index.T}
    for i, j in list(edges):
        assert (j, i) in edges, f"missing reverse edge for ({i}, {j})"


def test_edge_index_within_bounds() -> None:
    g = ligand_graph_from_smiles(ASPIRIN_SMILES)
    n = g.num_atoms
    assert g.edge_index.min() >= 0
    assert g.edge_index.max() < n
    assert g.edge_index.dtype == np.int64


def test_smiles_canonicalised() -> None:
    """Re-parsing a graph's SMILES should yield the same canonical form."""
    from rdkit import Chem

    g = ligand_graph_from_smiles(ASPIRIN_SMILES)
    canonical = Chem.MolToSmiles(Chem.MolFromSmiles(ASPIRIN_SMILES))
    assert g.smiles == canonical


def test_invalid_smiles_raises() -> None:
    with pytest.raises(ValueError):
        ligand_graph_from_smiles("not a real molecule!!!")


def test_ecfp_shape_and_some_bits_set() -> None:
    fp = ecfp_fingerprint(ASPIRIN_SMILES, radius=2, n_bits=2048)
    assert fp.shape == (2048,)
    assert fp.dtype == np.uint8
    # Aspirin should hit substantially more than zero bits.
    assert int(fp.sum()) > 5


def test_ecfp_invalid_smiles_raises() -> None:
    with pytest.raises(ValueError):
        ecfp_fingerprint("not a real molecule!!!", radius=2)


def test_ecfp_distinguishes_distinct_molecules() -> None:
    fp1 = ecfp_fingerprint(ASPIRIN_SMILES)
    fp2 = ecfp_fingerprint(BENZENE_SMILES)
    assert not np.array_equal(fp1, fp2)


def test_ligand_graph_from_mol_rejects_none() -> None:
    with pytest.raises(ValueError):
        ligand_graph_from_mol(None)


def test_ligand_graph_dataclass_properties() -> None:
    g = ligand_graph_from_smiles(BENZENE_SMILES)
    assert isinstance(g, LigandGraph)
    assert g.num_atoms == g.node_feats.shape[0]
    assert g.num_edges == g.edge_index.shape[1]
