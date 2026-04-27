"""Affinity-prediction metrics (scoring power on CASF-2016).

We report the same four numbers as the Phase 3 baseline so the README diff is
a single row:

* ``pearson_r``  - global Pearson correlation (CASF-2016 "scoring power").
* ``spearman_r`` - rank correlation.
* ``rmse``       - root mean squared error in pK units.
* ``mae``        - mean absolute error.

Ranking-power-by-cluster is left for Phase 6.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from scipy.stats import pearsonr, spearmanr


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return the four standard scoring-power metrics as native ``float``s."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: y_true {y_true.shape}, y_pred {y_pred.shape}")
    if y_true.size < 2:
        raise ValueError("need at least 2 samples to compute correlations")

    pearson = float(pearsonr(y_true, y_pred).statistic)
    spearman = float(spearmanr(y_true, y_pred).statistic)
    diff = y_pred - y_true
    rmse = float(np.sqrt(np.mean(diff**2)))
    mae = float(np.mean(np.abs(diff)))
    return {"pearson_r": pearson, "spearman_r": spearman, "rmse": rmse, "mae": mae}


def format_metrics_row(metrics: Mapping[str, float], precision: int = 3) -> str:
    """Pretty single-line summary, matching the baseline notebook's print format."""
    return (
        f"Pearson R = {metrics['pearson_r']:.{precision}f} | "
        f"Spearman R = {metrics['spearman_r']:.{precision}f} | "
        f"RMSE = {metrics['rmse']:.{precision}f} | "
        f"MAE = {metrics['mae']:.{precision}f}"
    )


__all__ = ["format_metrics_row", "regression_metrics"]
