"""Unit tests for the PDBbind index parser.

We test against a small synthetic file that mimics the real
``INDEX_refined_data.2020`` format, including comment lines, the ``//``
reference separator, and weird affinity strings.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from plb.data.pdbbind import (
    _parse_index_line,
    load_casf2016_coreset_ids,
    load_refined_index,
)

SAMPLE_INDEX = """\
# ==============================================================================
# List of the protein-ligand complexes in the PDBbind refined set v.2020
# 5316 protein-ligand complexes in total
# PDB code, resolution, release year, -logKd/Ki, Kd/Ki, reference, ligand name
# ==============================================================================
1a1e  2.80  1995   2.00  Ki=10mM      // 1a1e.pdf (PUR)
1a30  2.30  1996   8.00  Kd=10nM      // 1a30.pdf (XYZ)
2nmr   NMR  2003   6.50  Ki=320uM     // 2nmr.pdf (ABC)
1xxx  1.90  2010   7.10  IC50=80nM    // 1xxx.pdf (LIG-A)
"""


SAMPLE_CORESET = """\
# CASF-2016 core set
# pdb_code  cluster  affinity
1a30  1  8.00
1xxx  2  7.10
2nmr  3  6.50
"""


def test_parse_basic_line() -> None:
    entry = _parse_index_line("1a30  2.30  1996   8.00  Kd=10nM      // 1a30.pdf (XYZ)")
    assert entry is not None
    assert entry.pdb_id == "1a30"
    assert entry.resolution == 2.30
    assert entry.release_year == 1996
    assert entry.pK == 8.0
    assert entry.affinity_raw == "Kd=10nM"
    assert entry.ligand_name == "XYZ"


def test_parse_nmr_resolution() -> None:
    entry = _parse_index_line("2nmr   NMR  2003   6.50  Ki=320uM     // 2nmr.pdf (ABC)")
    assert entry is not None
    assert entry.resolution is None
    assert entry.pdb_id == "2nmr"


def test_parse_skips_comments_and_blank() -> None:
    assert _parse_index_line("# header") is None
    assert _parse_index_line("") is None
    assert _parse_index_line("   ") is None


def test_load_refined_index(tmp_path: Path) -> None:
    p = tmp_path / "INDEX_refined_data.2020"
    p.write_text(SAMPLE_INDEX, encoding="utf-8")

    df = load_refined_index(p)

    assert len(df) == 4
    assert list(df["pdb_id"]) == sorted(["1a1e", "1a30", "2nmr", "1xxx"])
    assert df.loc[df["pdb_id"] == "1xxx", "ligand_name"].iloc[0] == "LIG-A"


def test_load_refined_index_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_refined_index(tmp_path / "nope.txt")


def test_load_casf2016_coreset(tmp_path: Path) -> None:
    p = tmp_path / "CoreSet.dat"
    p.write_text(SAMPLE_CORESET, encoding="utf-8")

    ids = load_casf2016_coreset_ids(p)

    assert ids == sorted(["1a30", "1xxx", "2nmr"])
