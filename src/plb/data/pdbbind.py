"""Parse the PDBbind v2020 index and the CASF-2016 core-set list.

PDBbind ships a plain-text index file at
``refined-set/index/INDEX_refined_data.2020`` whose format is::

    # ==============================================================================
    # List of the protein-ligand complexes in the PDBbind refined set v.2020
    # 5316 protein-ligand complexes in total, sorted by their release year.
    # Latest update: July 2021
    # PDB code, resolution, release year, -logKd/Ki, Kd/Ki, reference, ligand name
    # ==============================================================================
    1a1e  2.80  1995   2.00  Ki=10mM      // 1a1e.pdf (PUR)
    ...

CASF-2016 ships its core set list at
``CASF-2016/power_scoring/CoreSet.dat`` (also whitespace-separated with a
``#`` header). We only need the PDB codes from each.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class PDBbindEntry:
    """A single row of the PDBbind refined-set index."""

    pdb_id: str
    resolution: float | None
    release_year: int
    pK: float
    affinity_raw: str
    ligand_name: str


_AFFINITY_RE = re.compile(
    r"^(?P<kind>K[id]|IC50)(?P<op>[=<>~])(?P<value>[\d.]+)(?P<unit>[a-zA-Z]+)$"
)


def _parse_index_line(line: str) -> PDBbindEntry | None:
    """Parse one non-comment line of an INDEX_*_data.2020 file.

    Returns ``None`` if the line is empty or a comment.
    """
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None

    # Drop everything after the "//" reference separator: that field can
    # contain spaces (ligand name in parentheses) which makes naive splitting
    # fragile.
    payload, _, tail = stripped.partition("//")
    parts = payload.split()
    if len(parts) < 5:
        raise ValueError(f"unexpected PDBbind index line: {line!r}")

    pdb_id = parts[0].lower()
    try:
        resolution: float | None = float(parts[1])
    except ValueError:
        resolution = None  # NMR structures have "NMR" instead of a number
    release_year = int(parts[2])
    pK = float(parts[3])
    affinity_raw = parts[4]

    ligand_name = ""
    if tail:
        m = re.search(r"\(([^)]*)\)\s*$", tail)
        if m:
            ligand_name = m.group(1)

    return PDBbindEntry(
        pdb_id=pdb_id,
        resolution=resolution,
        release_year=release_year,
        pK=pK,
        affinity_raw=affinity_raw,
        ligand_name=ligand_name,
    )


def load_refined_index(index_path: Path) -> pd.DataFrame:
    """Load the PDBbind v2020 refined-set index file as a DataFrame.

    Parameters
    ----------
    index_path
        Path to ``INDEX_refined_data.2020`` (or any sibling file in the same
        format, e.g. the general set).

    Returns
    -------
    DataFrame with columns ``pdb_id``, ``resolution``, ``release_year``,
    ``pK``, ``affinity_raw``, ``ligand_name``. Sorted by ``pdb_id``.
    """
    index_path = Path(index_path)
    if not index_path.is_file():
        raise FileNotFoundError(
            f"PDBbind index not found at {index_path}. "
            "Run `python scripts/download_pdbbind.py` after placing the "
            "PDBbind tarball in data/raw/."
        )

    rows: list[PDBbindEntry] = []
    for line in index_path.read_text(encoding="utf-8").splitlines():
        entry = _parse_index_line(line)
        if entry is not None:
            rows.append(entry)

    if not rows:
        raise ValueError(f"no entries parsed from {index_path}")

    df = pd.DataFrame([r.__dict__ for r in rows])
    return df.sort_values("pdb_id").reset_index(drop=True)


def load_casf2016_coreset_ids(coreset_path: Path) -> list[str]:
    """Load the list of CASF-2016 core-set PDB IDs.

    Parameters
    ----------
    coreset_path
        Path to ``CoreSet.dat`` from the CASF-2016 distribution. The file is
        whitespace-separated with the PDB code in the first column and a
        ``#``-prefixed header.

    Returns
    -------
    Sorted list of unique lower-case PDB IDs (typically 285 entries).
    """
    coreset_path = Path(coreset_path)
    if not coreset_path.is_file():
        raise FileNotFoundError(
            f"CASF-2016 CoreSet.dat not found at {coreset_path}. "
            "Download CASF-2016.tar.gz from PDBbind and extract it into "
            "data/raw/CASF-2016/."
        )

    ids: set[str] = set()
    for line in coreset_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        first = stripped.split()[0]
        if len(first) == 4:  # PDB codes are 4 chars
            ids.add(first.lower())

    if not ids:
        raise ValueError(f"no PDB IDs parsed from {coreset_path}")

    return sorted(ids)


def find_default_paths(data_root: Path) -> dict[str, Path]:
    """Return the canonical sub-paths within ``data_root`` for PDBbind data.

    These are the locations populated by ``scripts/download_pdbbind.py``.
    """
    data_root = Path(data_root)
    return {
        "refined_root": data_root / "raw" / "PDBbind_v2020_refined" / "refined-set",
        "refined_index": data_root
        / "raw"
        / "PDBbind_v2020_refined"
        / "refined-set"
        / "index"
        / "INDEX_refined_data.2020",
        "casf_root": data_root / "raw" / "CASF-2016",
        "casf_coreset": data_root / "raw" / "CASF-2016" / "power_scoring" / "CoreSet.dat",
    }
