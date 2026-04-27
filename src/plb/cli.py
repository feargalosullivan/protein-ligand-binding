"""CLI entry point for the plb package."""

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from plb.data.pdbbind import (
    find_default_paths,
    load_casf2016_coreset_ids,
    load_refined_index,
)
from plb.data.splits import make_splits

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Protein-ligand binding affinity (PDBbind / CASF-2016).",
)
data_app = typer.Typer(no_args_is_help=True, help="Data preparation commands.")
app.add_typer(data_app, name="data")

console = Console()


@data_app.command("prepare")
def data_prepare(
    data_root: Path = typer.Option(
        Path("data"),
        "--data-root",
        help="Root directory containing raw/ and processed/ sub-folders.",
    ),
    val_frac: float = typer.Option(0.1, help="Fraction of non-CASF refined set used as val."),
    seed: int = typer.Option(42, help="RNG seed for the train/val split."),
) -> None:
    """Parse PDBbind index and write train/val/test split to data/processed/."""
    paths = find_default_paths(data_root)
    refined_df = load_refined_index(paths["refined_index"])
    casf_ids = load_casf2016_coreset_ids(paths["casf_coreset"])

    split = make_splits(
        refined_pdbids=refined_df["pdb_id"].tolist(),
        casf_pdbids=casf_ids,
        val_frac=val_frac,
        seed=seed,
    )

    processed_root = data_root / "processed"
    processed_root.mkdir(parents=True, exist_ok=True)

    refined_df.to_csv(processed_root / "refined_index.csv", index=False)
    (processed_root / "split.json").write_text(
        json.dumps(
            {"train": split.train, "val": split.val, "test": split.test},
            indent=2,
        ),
        encoding="utf-8",
    )

    table = Table(title="Split sizes")
    table.add_column("subset", style="cyan")
    table.add_column("count", justify="right", style="green")
    for k, v in split.sizes.items():
        table.add_row(k, str(v))
    table.add_row("CASF-2016 declared", str(len(casf_ids)))
    table.add_row("PDBbind refined", str(len(refined_df)))
    console.print(table)
    console.print(f"Wrote {processed_root / 'refined_index.csv'}")
    console.print(f"Wrote {processed_root / 'split.json'}")


if __name__ == "__main__":
    app()
