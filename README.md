# Protein-Ligand Binding Affinity Predictor

Predict the binding affinity (pKd / pKi / pIC50) of a small molecule against a protein, trained on PDBbind v2020 and benchmarked on the CASF-2016 core set.

> **Status:** Phases 1-4 complete. Phase 4 GIN + ESM-pool + MLP hits Pearson R = 0.76, RMSE = 1.50 on CASF-2016, beating the classical XGBoost baseline (R = 0.72, RMSE = 1.51) across every metric. Live demo (Phase 5) coming next.

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
| ECFP + ESM-2 35M pool + XGBoost      | 0.719     | 0.687      | 1.514 | 1.184 |
| Ligand-GIN + ESM-2 35M pool + MLP    | **0.761** | **0.736**  | **1.496** | **1.168** |
| Pafnucy (Stepniewska-Dziubinska)*    | 0.78      | -          | 1.42  | -     |
| DeepDTA (Ozturk et al. 2018)*        | 0.66      | -          | 1.59  | -     |

\* Reported on the CASF-2016 standard core set (n = 285), so not directly comparable; included for context.

The deep model lifts Pearson R by +0.042 and Spearman R by +0.049 over the baseline while cutting RMSE — a clean signal that the learned ligand encoder extracts information that 2048-bit ECFP4 fingerprints can't.

![Baseline scatter](reports/baseline_casf2016_scatter.png)
![GNN scatter](reports/gnn_casf2016_scatter.png)

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
  data/      # PDBbind parsing, train/val/test splits, ligand + ESM featurisation, PyG dataset
  models/    # GIN ligand encoder + AffinityModel
  train.py   # AdamW + ReduceLROnPlateau + early-stop training loop
  eval.py    # CASF-2016 scoring-power metrics
  cli.py     # `plb` command-line entry point
configs/     # YAML run configs
notebooks/   # EDA, featurisation sanity checks, XGBoost baseline, GNN sweep
scripts/     # one-off scripts (data download, ESM + ligand-graph precomputation)
tests/       # pytest unit tests (65 tests)
app/         # Streamlit demo (Phase 5)
reports/     # scatter plots, results write-up
runs/        # per-run training artefacts (config, log, best.pt, metrics)
```

## Roadmap

- [x] Phase 1 - Repo scaffold, data download script, splits + tests, EDA
- [x] Phase 2 - Ligand graph featurisation (RDKit), pocket extraction (Biopython), ESM-2 35M pocket embeddings + cache script
- [x] Phase 3 - ECFP4 + ESM-pool + XGBoost baseline ([notebook](notebooks/03_baseline_xgboost.ipynb)): R = 0.72, RMSE = 1.51 on CASF-2016
- [x] Phase 4 - Ligand-GIN + ESM-pool + MLP ([notebook](notebooks/04_gnn_training.ipynb)): R = 0.76, RMSE = 1.50 on CASF-2016
- [ ] Phase 5 - HF Hub model card, Streamlit demo on HF Spaces, finalise README
- [ ] Phase 6 (stretch) - scaffold split, target-based split, 3D pocket SchNet, ensembling

## License

Code: MIT (see [LICENSE](LICENSE)).
PDBbind data: see PDBbind's own license; not redistributed.
