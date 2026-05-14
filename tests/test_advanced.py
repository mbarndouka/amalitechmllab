"""Smoke tests for models/advanced.py — each trainer fits and returns valid metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.advanced import (
    _evaluate,
    build_comparison_table,
    train_decision_tree,
    train_gradient_boosting,
    train_lasso,
    train_ridge,
    train_xgboost,
)


@pytest.fixture()
def small_data():
    """Small linear dataset — fast to train, clear signal."""
    np.random.seed(42)
    n = 300
    X = pd.DataFrame(
        {
            "a": np.random.randn(n),
            "b": np.random.randn(n),
            "c": np.random.randn(n),
        }
    )
    y = 5 * X["a"] - 3 * X["b"] + np.random.randn(n) * 0.5
    # log1p-transformed target (realistic for fare pipeline)
    y_log = np.log1p(np.abs(y) * 1000 + 5000)

    split = int(n * 0.7)
    val_split = int(n * 0.85)
    return (
        X.iloc[:split],
        X.iloc[split:val_split],
        X.iloc[val_split:],
        y_log.iloc[:split],
        y_log.iloc[split:val_split],
        y_log.iloc[val_split:],
    )


@pytest.fixture()
def minimal_cfg() -> dict:
    return {
        "models": {
            "ridge": {"alpha": [0.1, 1.0]},
            "lasso": {"alpha": [0.01, 0.1], "max_iter": 1000},
            "decision_tree": {"max_depth": [3, 5], "min_samples_split": [2], "min_samples_leaf": [1]},
            "random_forest": {"random_state": 42},
            "gradient_boosting": {"n_estimators": 20, "learning_rate": 0.1, "max_depth": 3, "random_state": 42},
            "xgboost": {"n_estimators": 20, "learning_rate": 0.1, "max_depth": 3, "random_state": 42},
        }
    }


# ── Ridge ──────────────────────────────────────────────────────────────────────


def test_train_ridge_returns_model_and_params(small_data, minimal_cfg):
    X_train, _, _, y_train, _, _ = small_data
    model, params = train_ridge(X_train, y_train, minimal_cfg, cv=2)
    assert hasattr(model, "predict")
    assert "alpha" in params


def test_train_ridge_predictions_shape(small_data, minimal_cfg):
    X_train, X_val, _, y_train, _, _ = small_data
    model, _ = train_ridge(X_train, y_train, minimal_cfg, cv=2)
    preds = model.predict(X_val)
    assert len(preds) == len(X_val)


# ── Lasso ──────────────────────────────────────────────────────────────────────


def test_train_lasso_returns_model_and_params(small_data, minimal_cfg):
    X_train, _, _, y_train, _, _ = small_data
    model, params = train_lasso(X_train, y_train, minimal_cfg, cv=2)
    assert hasattr(model, "predict")
    assert "alpha" in params


# ── Decision Tree ──────────────────────────────────────────────────────────────


def test_train_decision_tree_returns_model_and_params(small_data, minimal_cfg):
    X_train, _, _, y_train, _, _ = small_data
    model, params = train_decision_tree(X_train, y_train, minimal_cfg, cv=2)
    assert hasattr(model, "predict")
    assert "max_depth" in params


def test_train_decision_tree_respects_max_depth(small_data, minimal_cfg):
    X_train, _, _, y_train, _, _ = small_data
    model, _ = train_decision_tree(X_train, y_train, minimal_cfg, cv=2)
    assert model.max_depth in [3, 5]


# ── Gradient Boosting ──────────────────────────────────────────────────────────


def test_train_gradient_boosting_returns_model(small_data, minimal_cfg):
    X_train, _, _, y_train, _, _ = small_data
    model, params = train_gradient_boosting(X_train, y_train, minimal_cfg)
    assert hasattr(model, "predict")
    assert "n_estimators" in params


# ── XGBoost ────────────────────────────────────────────────────────────────────


def test_train_xgboost_returns_model_and_params(small_data, minimal_cfg):
    X_train, X_val, _, y_train, y_val, _ = small_data
    model, params = train_xgboost(X_train, y_train, X_val, y_val, minimal_cfg)
    assert hasattr(model, "predict")
    assert "n_estimators" in params


def test_train_xgboost_uses_early_stopping(small_data, minimal_cfg):
    X_train, X_val, _, y_train, y_val, _ = small_data
    model, params = train_xgboost(X_train, y_train, X_val, y_val, minimal_cfg)
    assert "best_iteration" in params


# ── _evaluate helper ───────────────────────────────────────────────────────────


def test_evaluate_returns_all_splits(small_data, minimal_cfg):
    X_train, X_val, X_test, y_train, y_val, y_test = small_data
    model, _ = train_ridge(X_train, y_train, minimal_cfg, cv=2)
    metrics = _evaluate(model, X_train, X_val, X_test, y_train, y_val, y_test, "ridge")
    assert set(metrics.keys()) == {"train", "val", "test"}


def test_evaluate_log_target_produces_bdt_scale_mae(small_data, minimal_cfg):
    X_train, X_val, X_test, y_train, y_val, y_test = small_data
    model, _ = train_ridge(X_train, y_train, minimal_cfg, cv=2)

    m_log = _evaluate(model, X_train, X_val, X_test, y_train, y_val, y_test, "ridge", log_target=False)
    m_orig = _evaluate(model, X_train, X_val, X_test, y_train, y_val, y_test, "ridge", log_target=True)

    # log-scale MAE is tiny (~0–12); BDT-scale MAE is thousands
    assert m_orig["test"]["mae"] > m_log["test"]["mae"] * 100


# ── build_comparison_table ─────────────────────────────────────────────────────


def test_build_comparison_table_shape(small_data, minimal_cfg):
    X_train, X_val, X_test, y_train, y_val, y_test = small_data
    model, params = train_ridge(X_train, y_train, minimal_cfg, cv=2)
    metrics = _evaluate(model, X_train, X_val, X_test, y_train, y_val, y_test, "ridge")
    all_results = {"ridge": {"metrics": metrics, "best_params": params}}
    df = build_comparison_table(all_results)
    # 1 model × 3 splits = 3 rows
    assert len(df) == 3
    assert set(df.columns) >= {"model", "split", "r2", "mae", "rmse", "mape"}


def test_build_comparison_table_multi_model(small_data, minimal_cfg):
    X_train, X_val, X_test, y_train, y_val, y_test = small_data
    r_model, r_params = train_ridge(X_train, y_train, minimal_cfg, cv=2)
    l_model, l_params = train_lasso(X_train, y_train, minimal_cfg, cv=2)
    r_metrics = _evaluate(r_model, X_train, X_val, X_test, y_train, y_val, y_test, "ridge")
    l_metrics = _evaluate(l_model, X_train, X_val, X_test, y_train, y_val, y_test, "lasso")
    all_results = {
        "ridge": {"metrics": r_metrics, "best_params": r_params},
        "lasso": {"metrics": l_metrics, "best_params": l_params},
    }
    df = build_comparison_table(all_results)
    assert len(df) == 6  # 2 models × 3 splits
    assert set(df["model"].unique()) == {"ridge", "lasso"}
