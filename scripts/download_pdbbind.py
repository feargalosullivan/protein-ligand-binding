"""Verify and extract PDBbind v2020 refined + CASF-2016 archives.

PDBbind requires registration; we can't download for you. This script's job
is to (1) tell you exactly what to download and where to put it, (2) extract
the archives once you've placed them in ``data/raw/``, and (3) sanity-check
the result.

Usage::

    python scripts/download_pdbbind.py
    python scripts/download_pdbbind.py --data-root D:/datasets/plb

Expected files in ``<data-root>/raw/`` (filenames as PDBbind ships them):

    PDBbind_v2020_refined.tar.gz      # ~4 GB
    CASF-2016.tar.gz                  # ~600 MB

After extraction the layout is::

    <data-root>/raw/PDBbind_v2020_refined/refined-set/<pdb_id>/...
    <data-root>/raw/CASF-2016/power_scoring/CoreSet.dat
"""

from __future__ import annotations

import sys
import tarfile
from pathlib import Path

from rich.console import Console

console = Console()

REFINED_TARBALL = "PDBbind_v2020_refined.tar.gz"
CASF_TARBALL = "CASF-2016.tar.gz"

REGISTRATION_INSTRUCTIONS = """\
PDBbind / CASF-2016 require registration. To obtain the data:

  1. Register a free academic account at:
       http://www.pdbbind-plus.org.cn/

  2. Once logged in, download:
       - PDBbind v2020 refined set       -> {refined}
       - CASF-2016                       -> {casf}

  3. Place both archives in:
       {raw_dir}

  4. Re-run this script:
       python scripts/download_pdbbind.py

The total disk footprint is ~5 GB (downloaded archives + extracted files).
"""


def _safe_extract(archive: Path, dest: Path) -> None:
    """Extract a tar.gz archive into ``dest``, skipping unsafe paths."""
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tf:
        for member in tf.getmembers():
            target = (dest / member.name).resolve()
            if not str(target).startswith(str(dest.resolve())):
                console.print(f"[red]refusing unsafe path:[/red] {member.name}")
                continue
            tf.extract(member, dest)  # noqa: S202 - guarded above


def main(data_root: Path = Path("data")) -> int:
    raw = (data_root / "raw").resolve()
    raw.mkdir(parents=True, exist_ok=True)

    refined_archive = raw / REFINED_TARBALL
    casf_archive = raw / CASF_TARBALL

    missing = [p.name for p in (refined_archive, casf_archive) if not p.exists()]

    refined_extracted = raw / "PDBbind_v2020_refined" / "refined-set"
    casf_extracted = raw / "CASF-2016" / "power_scoring" / "CoreSet.dat"

    need_extract_refined = refined_archive.exists() and not refined_extracted.is_dir()
    need_extract_casf = casf_archive.exists() and not casf_extracted.is_file()

    if missing and not (refined_extracted.is_dir() and casf_extracted.is_file()):
        console.print(
            REGISTRATION_INSTRUCTIONS.format(
                refined=refined_archive,
                casf=casf_archive,
                raw_dir=raw,
            )
        )
        console.print(f"[yellow]Missing archives:[/yellow] {', '.join(missing)}")
        return 1

    if need_extract_refined:
        console.print(f"[cyan]Extracting[/cyan] {refined_archive.name} ...")
        _safe_extract(refined_archive, raw / "PDBbind_v2020_refined")

    if need_extract_casf:
        console.print(f"[cyan]Extracting[/cyan] {casf_archive.name} ...")
        _safe_extract(casf_archive, raw)

    if not refined_extracted.is_dir():
        console.print(f"[red]Refined set not found at {refined_extracted}[/red]")
        return 2
    if not casf_extracted.is_file():
        console.print(f"[red]CASF-2016 CoreSet.dat not found at {casf_extracted}[/red]")
        return 2

    n_refined = sum(1 for p in refined_extracted.iterdir() if p.is_dir())
    n_casf = sum(
        1
        for line in casf_extracted.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    )

    console.print(
        f"[green]OK[/green] PDBbind refined: {n_refined} complexes at {refined_extracted}"
    )
    console.print(f"[green]OK[/green] CASF-2016 core: {n_casf} entries at {casf_extracted}")

    if n_refined < 5000:
        console.print(
            f"[yellow]warn:[/yellow] expected ~5316 refined complexes, found {n_refined}. "
            "Did the archive extract fully?"
        )
    if n_casf < 280:
        console.print(f"[yellow]warn:[/yellow] expected ~285 CASF-2016 entries, found {n_casf}.")

    console.print("\nNext: run [bold]plb data prepare[/bold] to build the train/val/test split.")
    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Root directory containing raw/ (default: ./data)",
    )
    args = parser.parse_args()
    sys.exit(main(args.data_root))
