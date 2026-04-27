"""CASF-2016 scoring-power metrics: Pearson R, Spearman R, RMSE, MAE."""

from collections.abc import Mapping

import numpy as np
from scipy.stats import pearsonr, spearmanr


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
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
    # TODO: add ranking power per-cluster (CASF-2016 "ranking power" protocol)
    return {"pearson_r": pearson, "spearman_r": spearman, "rmse": rmse, "mae": mae}


def format_metrics_row(metrics: Mapping[str, float], precision: int = 3) -> str:
    return (
        f"Pearson R = {metrics['pearson_r']:.{precision}f} | "
        f"Spearman R = {metrics['spearman_r']:.{precision}f} | "
        f"RMSE = {metrics['rmse']:.{precision}f} | "
        f"MAE = {metrics['mae']:.{precision}f}"
    )
