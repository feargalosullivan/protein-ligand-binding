# Protein-Ligand Binding Affinity Predictor

Predict the binding affinity (pKd / pKi / pIC50) of a small molecule against a protein, trained on PDBbind v2020 and benchmarked on the CASF-2016 core set.

> **Status:** Phases 1-3 complete. Classical baseline (ECFP + ESM-pool + XGBoost) hits Pearson R = 0.72, RMSE = 1.51 on CASF-2016. Deep model (Phase 4) and live demo (Phase 5) coming next.

## Why this exists

Most published protein-ligand affinity models are evaluated on the same held-out test set: the [CASF-2016](http://www.pdbbind.org.cn/casf.php) core set (285 high-quality complexes, 57 protein clusters). This project trains a small graph neural network with frozen [ESM-2](https://github.com/facebookresearch/esm) protein embeddings on PDBbind v2020, evaluates on CASF-2016, and ships a live demo so the result is verifiable in the browser.

The deliverables are three CV-facing links:

- **Code** — this repo
- **Model weights** — Hugging Face Hub (TBD)
- **Live demo** — Hugging Face Spaces (TBD)

## Headline result

CASF-2016 core set (n = 266 of the 285 declared, intersected with the v2020 refined set).
CASF-2016 PDB IDs are strictly excluded from train and val.

| Model                                | Pearson R | Spearman R | RMSE  | MAE   |
| ------------------------------------ | --------- | ---------- | ----- | ----- |
| ECFP + ESM-2 35M pool + XGBoost      | **0.719** | 0.687      | 1.514 | 1.184 |
| Ligand-GIN + ESM-2 35M pool + MLP    | TBD       | TBD        | TBD   | TBD   |
| Pafnucy (Stepniewska-Dziubinska)*    | 0.78      | -          | 1.42  | -     |
| DeepDTA (Ozturk et al. 2018)*        | 0.66      | -          | 1.59  | -     |

\* Reported on the CASF-2016 standard core set (n = 285), so not directly comparable; included for context.

![Baseline scatter](reports/baseline_casf2016_scatter.png)

## Quickstart

```powershell
# 1. Clone and create a virtual env
git clone https://github.com/<you>/protein-ligand-binding.git
cd protein-ligand-binding
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install the package (Phase 1 deps only - no torch/rdkit yet)
pip install -e ".[dev]"

# 3. Run the test suite (no data needed)
pytest

# 4. Once you have downloaded PDBbind + CASF-2016 (see scripts/download_pdbbind.py),
#    run the data preparation pipeline
plb data prepare
```

For the ML stack (Phase 2+):

```powershell
pip install -e ".[ml,dev]"
```

## Data

PDBbind v2020 and CASF-2016 are **free for academic use but require registration** at <http://www.pdbbind-plus.org.cn/>. They are **not redistributed in this repo**. After registering, place the downloaded archives in `data/raw/` and run:

```powershell
python scripts/download_pdbbind.py
```

See [scripts/download_pdbbind.py](scripts/download_pdbbind.py) for the exact filenames the script expects.

## Project layout

```
src/plb/
  data/      # PDBbind parsing, train/val/test splits, featurisation
  models/    # GNN model definitions
  cli.py     # `plb` command-line entry point
configs/     # YAML run configs
notebooks/   # EDA + the lightweight XGBoost baseline notebook
scripts/     # one-off scripts (data download, ESM precomputation, etc.)
tests/       # pytest unit tests
app/         # Streamlit demo (Phase 5)
reports/     # results write-up
```

## Roadmap

- [x] Phase 1 - Repo scaffold, data download script, splits + tests, EDA
- [x] Phase 2 - Ligand graph featurisation (RDKit), pocket extraction (Biopython), ESM-2 35M pocket embeddings + cache script
- [x] Phase 3 - ECFP4 + ESM-pool + XGBoost baseline ([notebook](notebooks/03_baseline_xgboost.ipynb)): R = 0.72, RMSE = 1.51 on CASF-2016
- [ ] Phase 4 - GNN + ESM-pool + MLP, CASF-2016 evaluation, comparison to baseline
- [ ] Phase 5 - HF Hub model card, Streamlit demo on HF Spaces, finalise README
- [ ] Phase 6 (stretch) - scaffold split, target-based split, 3D pocket SchNet, ensembling

## License

Code: MIT (see [LICENSE](LICENSE)).
PDBbind data: see PDBbind's own license; not redistributed.
