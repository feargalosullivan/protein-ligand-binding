"""Critical correctness tests for the train/val/test split.

These tests guard against the most common pitfall in PDBbind affinity papers:
training-set leakage of CASF-2016 complexes, which inflates the headline
metric. We test the split function on synthetic IDs so we never need the
actual data to run the test suite.
"""

from __future__ import annotations

import pytest

from plb.data.splits import Split, make_splits


def _fake_refined(n: int = 100) -> list[str]:
    return [f"pdb{i:04d}" for i in range(n)]


def test_casf_excluded_from_train_and_val() -> None:
    refined = _fake_refined(100)
    casf = set(refined[:10])

    split = make_splits(refined, casf, val_frac=0.1, seed=42)

    assert set(split.test) == casf
    assert casf.isdisjoint(split.train)
    assert casf.isdisjoint(split.val)


def test_partition_is_complete_and_disjoint() -> None:
    refined = _fake_refined(100)
    casf = set(refined[:10])

    split = make_splits(refined, casf, val_frac=0.1, seed=42)

    union = set(split.train) | set(split.val) | set(split.test)
    total = len(split.train) + len(split.val) + len(split.test)
    assert union == set(refined)
    assert total == len(set(refined)), "split must be a clean partition"


def test_casf_ids_not_in_refined_are_silently_dropped() -> None:
    refined = _fake_refined(100)
    casf = {f"pdb{i:04d}" for i in range(95, 110)}  # 5 in, 5 not in

    split = make_splits(refined, casf, val_frac=0.1, seed=42)

    assert set(split.test) == casf & set(refined)
    assert len(split.test) == 5


def test_deterministic_with_seed() -> None:
    refined = _fake_refined(200)
    casf = set(refined[:20])

    a = make_splits(refined, casf, seed=42)
    b = make_splits(refined, casf, seed=42)
    c = make_splits(refined, casf, seed=43)

    assert a == b
    assert a != c, "different seeds should yield different train/val partitions"


def test_pdbid_case_is_normalised() -> None:
    """Mixed-case inputs are lower-cased throughout. Uses 50 fake IDs to
    satisfy the size sanity check inside ``make_splits``."""
    refined = [f"PDB{i:04d}" for i in range(50)]
    casf = [refined[0].lower(), refined[10].upper()]  # mixed cases on input

    split = make_splits(refined, casf, val_frac=0.2, seed=0)

    assert all(p == p.lower() for p in split.train + split.val + split.test)
    assert set(split.test) == {"pdb0000", "pdb0010"}


def test_val_frac_validation() -> None:
    refined = _fake_refined(100)
    casf = set(refined[:10])

    with pytest.raises(ValueError):
        make_splits(refined, casf, val_frac=0.0)
    with pytest.raises(ValueError):
        make_splits(refined, casf, val_frac=1.0)
    with pytest.raises(ValueError):
        make_splits(refined, casf, val_frac=-0.1)


def test_too_small_refined_set_raises() -> None:
    with pytest.raises(ValueError):
        make_splits(["pdb0001", "pdb0002"], casf_pdbids=[], val_frac=0.1)


def test_split_dataclass_rejects_overlap() -> None:
    """Constructing a Split with overlapping members must fail loudly.

    This is a defence in depth: even if a future bug let leakage past
    ``make_splits``, the dataclass will refuse to be built.
    """
    with pytest.raises(ValueError, match="train and test overlap"):
        Split(train=["a", "b"], val=["c"], test=["a", "d"])
    with pytest.raises(ValueError, match="val and test overlap"):
        Split(train=["a"], val=["b"], test=["b", "c"])
    with pytest.raises(ValueError, match="train and val overlap"):
        Split(train=["a", "b"], val=["b"], test=["c"])


def test_no_casf_means_test_is_empty() -> None:
    refined = _fake_refined(50)

    split = make_splits(refined, casf_pdbids=[], val_frac=0.2, seed=0)

    assert split.test == []
    assert len(split.train) + len(split.val) == 50
