"""Train / val / test splits with CASF-2016 as a held-out test set."""

import random
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class Split:

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
