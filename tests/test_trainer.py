"""Tests for models/trainer.py"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.trainer import evaluate_all_splits, train_linear_regression


@pytest.fixture()
def simple_data():
    np.random.seed(42)
    n = 200
    X = pd.DataFrame({"a": np.random.randn(n), "b": np.random.randn(n)})
    y = 3 * X["a"] + 2 * X["b"] + np.random.randn(n) * 0.1
    return X.iloc[:140], X.iloc[140:170], X.iloc[170:], y.iloc[:140].values, y.iloc[140:170].values, y.iloc[170:].values


def test_train_linear_regression_returns_fitted_model(simple_data):
    X_train, _, _, y_train, _, _ = simple_data
    model = train_linear_regression(X_train, y_train)
    preds = model.predict(X_train)
    assert len(preds) == len(X_train)


def test_evaluate_all_splits_returns_three_splits(simple_data):
    X_train, X_val, X_test, y_train, y_val, y_test = simple_data
    model = train_linear_regression(X_train, y_train)
    metrics = evaluate_all_splits(model, X_train, X_val, X_test, y_train, y_val, y_test)
    assert set(metrics.keys()) == {"train", "val", "test"}


def test_evaluate_all_splits_metrics_keys(simple_data):
    X_train, X_val, X_test, y_train, y_val, y_test = simple_data
    model = train_linear_regression(X_train, y_train)
    metrics = evaluate_all_splits(model, X_train, X_val, X_test, y_train, y_val, y_test)
    for split in ("train", "val", "test"):
        assert set(metrics[split].keys()) == {"r2", "mae", "rmse", "mape"}


def test_evaluate_r2_reasonable_for_linear_data(simple_data):
    X_train, X_val, X_test, y_train, y_val, y_test = simple_data
    model = train_linear_regression(X_train, y_train)
    metrics = evaluate_all_splits(model, X_train, X_val, X_test, y_train, y_val, y_test)
    # linear data → linear model should fit well
    assert metrics["train"]["r2"] > 0.95
    assert metrics["test"]["r2"] > 0.90


def test_evaluate_log_target_inverse_transforms(simple_data):
    """With log_target=True, predictions and y are expm1-transformed before metrics."""
    X_train, X_val, X_test, y_train, y_val, y_test = simple_data
    # simulate realistic log1p(fare) values — fares in 5,000–200,000 BDT range
    np.random.seed(42)
    fare_train = np.random.uniform(5000, 200000, len(y_train))
    fare_val = np.random.uniform(5000, 200000, len(y_val))
    fare_test = np.random.uniform(5000, 200000, len(y_test))
    y_train_log = np.log1p(fare_train)
    y_val_log = np.log1p(fare_val)
    y_test_log = np.log1p(fare_test)

    model = train_linear_regression(X_train, y_train_log)

    metrics_log = evaluate_all_splits(
        model,
        X_train,
        X_val,
        X_test,
        y_train_log,
        y_val_log,
        y_test_log,
        log_target=False,
    )
    metrics_orig = evaluate_all_splits(
        model,
        X_train,
        X_val,
        X_test,
        y_train_log,
        y_val_log,
        y_test_log,
        log_target=True,
    )
    # MAE in BDT scale (thousands) must be far larger than in log scale (~0–12)
    assert metrics_orig["train"]["mae"] > metrics_log["train"]["mae"] * 1000


def test_train_linear_regression_is_deterministic(simple_data):
    X_train, _, _, y_train, _, _ = simple_data
    m1 = train_linear_regression(X_train, y_train)
    m2 = train_linear_regression(X_train, y_train)
    np.testing.assert_array_equal(m1.coef_, m2.coef_)
