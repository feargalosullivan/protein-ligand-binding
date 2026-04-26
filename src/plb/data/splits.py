"""Train / validation / test splits for PDBbind.

The test set is **always** the CASF-2016 core set. CASF-2016 PDB IDs are a
subset of the PDBbind refined set, so we strictly remove them from train/val.
Without this, the CASF-2016 numbers would be inflated by training-set leakage,
which is one of the most common pitfalls in published binding-affinity papers.

The split function is intentionally a pure function over PDB-ID lists so it
can be unit-tested without any of the actual PDBbind data being present.
"""

from __future__ import annotations

import random
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class Split:
    """A train / val / test partition of PDBbind PDB IDs.

    All three lists are sorted alphabetically for determinism. Every entry in
    ``test`` is guaranteed to also appear in the input refined-set list AND in
    the input CASF set (i.e. ``test == sorted(refined & casf)``).
    """

    train: list[str]
    val: list[str]
    test: list[str]

    def __post_init__(self) -> None:
        train_set = set(self.train)
        val_set = set(self.val)
        test_set = set(self.test)
        if train_set & test_set:
            raise ValueError(
                f"train and test overlap on {len(train_set & test_set)} ids; "
                "this would cause CASF-2016 leakage"
            )
        if val_set & test_set:
            raise ValueError(
                f"val and test overlap on {len(val_set & test_set)} ids; "
                "this would cause CASF-2016 leakage"
            )
        if train_set & val_set:
            raise ValueError(f"train and val overlap on {len(train_set & val_set)} ids")

    @property
    def sizes(self) -> dict[str, int]:
        return {"train": len(self.train), "val": len(self.val), "test": len(self.test)}


def make_splits(
    refined_pdbids: Iterable[str],
    casf_pdbids: Iterable[str],
    val_frac: float = 0.1,
    seed: int = 42,
) -> Split:
    """Build a train/val/test split.

    Parameters
    ----------
    refined_pdbids
        All PDB IDs in the PDBbind refined set (lower-case 4-character codes).
    casf_pdbids
        PDB IDs in the CASF-2016 core set. Any IDs not present in
        ``refined_pdbids`` are silently dropped (they cannot be in the test
        set if we don't have their structures).
    val_frac
        Fraction of the *non-CASF* refined set to use as validation.
        Must be in (0, 1).
    seed
        RNG seed for the train/val split. The test set is fully determined by
        ``casf_pdbids`` and does not depend on ``seed``.

    Returns
    -------
    A :class:`Split` whose ``test`` set equals the intersection of the refined
    set and ``casf_pdbids``, with train/val drawn from the remainder.
    """
    if not 0.0 < val_frac < 1.0:
        raise ValueError(f"val_frac must be in (0, 1); got {val_frac}")

    refined = sorted({p.lower() for p in refined_pdbids})
    casf = {p.lower() for p in casf_pdbids}

    if len(refined) < 10:
        raise ValueError(f"refined set has only {len(refined)} entries; expected thousands")

    test = sorted(set(refined) & casf)
    remaining = [p for p in refined if p not in casf]

    rng = random.Random(seed)
    shuffled = remaining[:]
    rng.shuffle(shuffled)

    n_val = max(1, int(round(len(shuffled) * val_frac)))
    val = sorted(shuffled[:n_val])
    train = sorted(shuffled[n_val:])

    return Split(train=train, val=val, test=test)
