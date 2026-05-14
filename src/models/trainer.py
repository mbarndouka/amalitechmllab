"""Step 4 — Baseline model training: Linear Regression."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from evaluation.metrics import compute_metrics, log_metrics
from utils.logging import get_logger

logger = get_logger(__name__)


def load_features(
    features_dir: str | Path,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Load train/val/test feature splits from parquet files."""
    d = Path(features_dir)
    X_train = pd.read_parquet(d / "X_train.parquet")
    X_val = pd.read_parquet(d / "X_val.parquet")
    X_test = pd.read_parquet(d / "X_test.parquet")
    y_train = pd.read_parquet(d / "y_train.parquet").squeeze().to_numpy()
    y_val = pd.read_parquet(d / "y_val.parquet").squeeze().to_numpy()
    y_test = pd.read_parquet(d / "y_test.parquet").squeeze().to_numpy()

    logger.info(
        "Features loaded  train=%d  val=%d  test=%d  features=%d",
        len(X_train),
        len(X_val),
        len(X_test),
        X_train.shape[1],
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_linear_regression(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
) -> LinearRegression:
    logger.info("Training LinearRegression on %d samples, %d features…", *X_train.shape)
    model = LinearRegression()
    model.fit(X_train, y_train)
    logger.info("Training complete.")
    return model


def evaluate_all_splits(
    model: LinearRegression,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    log_target: bool = False,
) -> dict[str, dict[str, float]]:
    """Evaluate model on all three splits and log results.

    log_target: when True, predictions and y are in log-scale — inverse-transform
    with expm1 before computing metrics so RMSE/MAE are in original BDT units.
    """
    results: dict[str, dict[str, float]] = {}
    for name, X, y in [("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)]:
        preds = model.predict(X)
        if log_target:
            preds = np.expm1(preds)
            y = np.expm1(y)
        m = compute_metrics(y, preds)
        log_metrics(m, name)
        results[name] = m
    return results


def save_artifacts(
    model: LinearRegression,
    metrics: dict[str, dict[str, float]],
    feature_names: list[str],
    models_dir: str | Path,
    reports_dir: str | Path,
) -> None:
    models_dir = Path(models_dir)
    reports_dir = Path(reports_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "linear_regression.pkl"
    metrics_path = reports_dir / "metrics_linear_regression.json"

    joblib.dump(model, model_path)
    logger.info("Model saved → %s", model_path)

    report = {
        "model": "LinearRegression",
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "metrics": metrics,
        "intercept": round(float(model.intercept_), 4),
    }
    metrics_path.write_text(json.dumps(report, indent=2))
    logger.info("Metrics saved → %s", metrics_path)


def run(cfg: dict[str, Any]) -> None:
    """Stage entry point — train baseline linear regression and save artifacts."""
    data_cfg = cfg.get("data", {})
    features_cfg = cfg.get("features", {})
    mlflow_cfg = cfg.get("mlflow", {})
    features_dir = data_cfg.get("features_dir", "data/features")
    log_target = bool(features_cfg.get("log_target", False))
    eval_log_space = bool(features_cfg.get("eval_log_space", False))
    models_dir = "models"
    reports_dir = "reports"

    logger.info("━━━━━━  Model Training: Linear Regression Baseline  ━━━━━━")
    if log_target and eval_log_space:
        logger.info("log_target=True + eval_log_space=True — metrics computed in log space")
    elif log_target:
        logger.info("log_target=True — metrics will be computed in original BDT scale via expm1")

    X_train, X_val, X_test, y_train, y_val, y_test = load_features(features_dir)

    model = train_linear_regression(X_train, y_train)

    logger.info("── Evaluation ──")
    # inverse-transform only when log_target=True AND we want BDT-scale metrics
    metrics = evaluate_all_splits(
        model, X_train, X_val, X_test, y_train, y_val, y_test, log_target=(log_target and not eval_log_space)
    )

    # Gap check — flag possible overfitting
    train_r2 = metrics["train"]["r2"]
    val_r2 = metrics["val"]["r2"]
    if train_r2 - val_r2 > 0.05:
        logger.warning(
            "Possible overfitting — train R²=%.4f vs val R²=%.4f (gap=%.4f)",
            train_r2,
            val_r2,
            train_r2 - val_r2,
        )
    elif val_r2 < 0.5:
        logger.warning("Low val R²=%.4f — model may be underfitting.", val_r2)
    else:
        logger.info("Train/val R² gap within acceptable range (%.4f vs %.4f).", train_r2, val_r2)

    save_artifacts(model, metrics, X_train.columns.tolist(), models_dir, reports_dir)

    # Log to MLflow if enabled
    if mlflow_cfg.get("enabled", True):
        from utils.mlflow_utils import log_model_run, setup_experiment

        setup_experiment(
            tracking_uri=mlflow_cfg.get("tracking_uri", "sqlite:///mlflow.db"),
            artifact_location=mlflow_cfg.get("artifact_location", "mlartifacts"),
        )
        log_model_run(
            model_name="linear_regression",
            model=model,
            params={"log_target": log_target},
            metrics=metrics,
            feature_names=X_train.columns.tolist(),
        )

    logger.info("━━━━━━  Training complete  ━━━━━━")
