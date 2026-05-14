"""Regression evaluation metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from utils.logging import get_logger

logger = get_logger(__name__)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return dict of r2, mae, rmse, mape for a regression prediction."""
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    # avoid division by zero for zero-fare rows (shouldn't exist but guard anyway)
    mask = y_true != 0
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
    return {"r2": round(r2, 4), "mae": round(mae, 2), "rmse": round(rmse, 2), "mape": round(mape, 2)}


def _fmt(v: float) -> str:
    return f"{v:.4f}" if v < 10 else f"{v:,.0f}"


def log_metrics(metrics: dict[str, float], split: str) -> None:
    logger.info(
        "[%s]  R²=%.4f  MAE=%s  RMSE=%s  MAPE=%.2f%%",
        split.upper(),
        metrics["r2"],
        _fmt(metrics["mae"]),
        _fmt(metrics["rmse"]),
        metrics["mape"],
    )
