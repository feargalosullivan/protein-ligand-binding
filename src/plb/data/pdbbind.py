"""PDBbind v2020 index parser and CASF-2016 core-set loader."""

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class PDBbindEntry:

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
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None

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
