# Protein-Ligand Binding Affinity Predictor

Predict the binding affinity (pKd / pKi / pIC50) of a small molecule against a protein, trained on PDBbind v2020 and benchmarked on the CASF-2016 core set.

> **Status:** scaffolding (Phase 1). Not yet usable end-to-end. See the [project plan](.cursor/plans/protein-ligand-binding-cv_22634dc4.plan.md) for the roadmap.

## Why this exists

Most published protein-ligand affinity models are evaluated on the same held-out test set: the [CASF-2016](http://www.pdbbind.org.cn/casf.php) core set (285 high-quality complexes, 57 protein clusters). This project trains a small graph neural network with frozen [ESM-2](https://github.com/facebookresearch/esm) protein embeddings on PDBbind v2020, evaluates on CASF-2016, and ships a live demo so the result is verifiable in the browser.

The deliverables are three CV-facing links:

- **Code** — this repo
- **Model weights** — Hugging Face Hub (TBD)
- **Live demo** — Hugging Face Spaces (TBD)

## Headline result (placeholder, to be filled in)

| Model                              | Pearson R | Spearman R | RMSE | MAE  |
| ---------------------------------- | --------- | ---------- | ---- | ---- |
| Random predictor                   | 0.00      | 0.00       | -    | -    |
| ECFP + ESM-pool + XGBoost (ours)   | TBD       | TBD        | TBD  | TBD  |
| Ligand-GNN + ESM-pool + MLP (ours) | TBD       | TBD        | TBD  | TBD  |
| GraphDTA (Nguyen et al. 2021)      | 0.78      | -          | -    | -    |
| Pafnucy (Stepniewska-Dziubinska)   | 0.78      | -          | -    | -    |

All "ours" rows reported on the CASF-2016 core set with CASF-2016 PDB IDs strictly excluded from the training set.

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
- [ ] Phase 2 - Ligand graph featurisation, ESM-2 pocket embeddings (cached)
- [ ] Phase 3 - Lightweight XGBoost baseline notebook (~80 lines)
- [ ] Phase 4 - GNN + ESM-pool + MLP, CASF-2016 evaluation, comparison to baseline
- [ ] Phase 5 - HF Hub model card, Streamlit demo on HF Spaces, finalise README
- [ ] Phase 6 (stretch) - scaffold split, target-based split, 3D pocket SchNet, ensembling

## License

Code: MIT (see [LICENSE](LICENSE)).
PDBbind data: see PDBbind's own license; not redistributed.
