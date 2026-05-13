"""Tests for evaluation/metrics.py"""
from __future__ import annotations

import numpy as np
import pytest

from evaluation.metrics import compute_metrics


def test_perfect_predictions():
    y = np.array([100.0, 200.0, 300.0])
    m = compute_metrics(y, y)
    assert m["r2"] == 1.0
    assert m["mae"] == 0.0
    assert m["rmse"] == 0.0
    assert m["mape"] == 0.0


def test_known_values():
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 190.0, 310.0])
    m = compute_metrics(y_true, y_pred)
    assert m["mae"] == pytest.approx(10.0, abs=0.01)
    expected_mape = (10 / 100 + 10 / 200 + 10 / 300) / 3 * 100
    assert m["mape"] == pytest.approx(expected_mape, rel=0.01)


def test_zero_fare_rows_excluded_from_mape():
    # rows where y_true=0 must be masked — no ZeroDivisionError
    y_true = np.array([0.0, 100.0, 200.0])
    y_pred = np.array([10.0, 110.0, 210.0])
    m = compute_metrics(y_true, y_pred)
    assert np.isfinite(m["mape"])


def test_all_zeros_mape_is_nan():
    # if every true value is 0, mape should be nan (no valid rows)
    y_true = np.array([0.0, 0.0, 0.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    m = compute_metrics(y_true, y_pred)
    assert np.isnan(m["mape"])


def test_output_keys():
    y = np.array([100.0, 200.0])
    m = compute_metrics(y, y)
    assert set(m.keys()) == {"r2", "mae", "rmse", "mape"}


def test_values_are_rounded_to_4dp():
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([105.0, 195.0, 305.0])
    m = compute_metrics(y_true, y_pred)
    for key, val in m.items():
        if not np.isnan(val):
            assert round(val, 4) == val, f"{key} not rounded to 4 dp: {val}"


def test_negative_r2_for_bad_model():
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    m = compute_metrics(y_true, y_pred)
    assert m["r2"] < 0
