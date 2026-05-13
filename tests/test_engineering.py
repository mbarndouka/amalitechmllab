"""Tests for features/engineering.py"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from features.engineering import (
    add_route_feature,
    drop_redundant_columns,
    fit_and_scale,
    log_transform_numerics,
    one_hot_encode,
    split_features_target,
    split_train_val_test,
    target_encode,
)


# ── drop_redundant_columns ─────────────────────────────────────────────────────

def test_drop_redundant_removes_cols():
    df = pd.DataFrame({"source_name": ["Dhaka"], "fare": [100]})
    result = drop_redundant_columns(df, ("source_name",))
    assert "source_name" not in result.columns
    assert "fare" in result.columns


def test_drop_redundant_noop_when_absent():
    df = pd.DataFrame({"fare": [100]})
    result = drop_redundant_columns(df, ("source_name",))
    assert list(result.columns) == ["fare"]


# ── add_route_feature ──────────────────────────────────────────────────────────

def test_add_route_creates_column():
    df = pd.DataFrame({"source": ["DAC", "CGP"], "destination": ["DXB", "DEL"]})
    result = add_route_feature(df)
    assert "route" in result.columns
    assert result["route"].tolist() == ["DAC_DXB", "CGP_DEL"]


def test_add_route_noop_when_source_missing():
    df = pd.DataFrame({"destination": ["DXB"]})
    result = add_route_feature(df)
    assert "route" not in result.columns


def test_add_route_does_not_mutate_input():
    df = pd.DataFrame({"source": ["DAC"], "destination": ["DXB"]})
    _ = add_route_feature(df)
    assert "route" not in df.columns


# ── one_hot_encode ─────────────────────────────────────────────────────────────

def test_ohe_expands_categories():
    df = pd.DataFrame({"airline": ["Air India", "IndiGo", "Air India"], "fare": [1, 2, 3]})
    result = one_hot_encode(df, ("airline",))
    assert "airline_Air India" in result.columns
    assert "airline_IndiGo" in result.columns
    assert "airline" not in result.columns


def test_ohe_columns_are_sorted():
    df = pd.DataFrame({"airline": ["IndiGo", "Air India"], "fare": [1, 2]})
    result = one_hot_encode(df, ("airline",))
    non_target = [c for c in result.columns if c != "fare"]
    assert non_target == sorted(non_target)


def test_ohe_skips_absent_columns():
    df = pd.DataFrame({"fare": [100]})
    result = one_hot_encode(df, ("airline",))
    assert "fare" in result.columns


# ── split_features_target ──────────────────────────────────────────────────────

def test_split_features_target_separates_correctly():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "fare": [10, 20]})
    X, y = split_features_target(df, "fare")
    assert "fare" not in X.columns
    assert list(y) == [10, 20]


def test_split_features_target_raises_on_missing():
    df = pd.DataFrame({"a": [1]})
    with pytest.raises(KeyError, match="fare"):
        split_features_target(df, "fare")


# ── split_train_val_test ───────────────────────────────────────────────────────

def test_split_sizes_are_correct():
    np.random.seed(0)
    X = pd.DataFrame({"a": range(1000)})
    y = pd.Series(range(1000))
    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(
        X, y, test_size=0.2, val_size=0.1, random_state=42
    )
    total = len(X_train) + len(X_val) + len(X_test)
    assert total == 1000
    assert len(X_test) == pytest.approx(200, abs=5)
    assert len(X_val) == pytest.approx(100, abs=10)


def test_split_no_overlap():
    X = pd.DataFrame({"a": range(200)})
    y = pd.Series(range(200))
    X_train, X_val, X_test, *_ = split_train_val_test(X, y, 0.2, 0.1, 42)
    train_idx = set(X_train.index)
    val_idx = set(X_val.index)
    test_idx = set(X_test.index)
    assert train_idx.isdisjoint(val_idx)
    assert train_idx.isdisjoint(test_idx)
    assert val_idx.isdisjoint(test_idx)


def test_split_is_reproducible():
    X = pd.DataFrame({"a": range(100)})
    y = pd.Series(range(100))
    r1 = split_train_val_test(X, y, 0.2, 0.1, 42)
    r2 = split_train_val_test(X, y, 0.2, 0.1, 42)
    pd.testing.assert_frame_equal(r1[0], r2[0])


# ── log_transform_numerics ─────────────────────────────────────────────────────

def test_log_transform_applies_log1p():
    df = pd.DataFrame({"duration": [0.0, 1.0, 9.0]})
    result = log_transform_numerics(df, ("duration",))
    expected = np.log1p([0.0, 1.0, 9.0])
    np.testing.assert_allclose(result["duration"].values, expected)


def test_log_transform_clips_negatives():
    df = pd.DataFrame({"duration": [-5.0, 3.0]})
    result = log_transform_numerics(df, ("duration",))
    assert result["duration"].iloc[0] == pytest.approx(np.log1p(0.0))


def test_log_transform_does_not_mutate_input():
    df = pd.DataFrame({"duration": [1.0, 2.0]})
    original = df.copy()
    _ = log_transform_numerics(df, ("duration",))
    pd.testing.assert_frame_equal(df, original)


def test_log_transform_skips_absent_cols():
    df = pd.DataFrame({"fare": [100.0]})
    result = log_transform_numerics(df, ("duration",))
    pd.testing.assert_frame_equal(result, df)


# ── fit_and_scale ──────────────────────────────────────────────────────────────

def test_fit_and_scale_standardizes_train():
    X_train = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
    X_val   = pd.DataFrame({"a": [2.0, 3.0]})
    X_test  = pd.DataFrame({"a": [4.0, 5.0]})
    X_train_s, _, _, _ = fit_and_scale(X_train, X_val, X_test, ("a",))
    assert X_train_s["a"].mean() == pytest.approx(0.0, abs=1e-10)
    assert X_train_s["a"].std(ddof=0) == pytest.approx(1.0, abs=1e-10)


def test_fit_and_scale_scaler_fitted_on_train_only():
    X_train = pd.DataFrame({"a": [10.0, 20.0]})
    X_val   = pd.DataFrame({"a": [1000.0]})
    X_test  = pd.DataFrame({"a": [2000.0]})
    _, _, _, scaler = fit_and_scale(X_train, X_val, X_test, ("a",))
    # scaler mean should match train, not val/test
    assert scaler.mean_[0] == pytest.approx(15.0, abs=0.01)


def test_fit_and_scale_skips_absent_cols():
    X_train = pd.DataFrame({"fare": [100.0, 200.0]})
    X_val   = pd.DataFrame({"fare": [150.0]})
    X_test  = pd.DataFrame({"fare": [180.0]})
    X_train_s, X_val_s, X_test_s, _ = fit_and_scale(X_train, X_val, X_test, ("duration",))
    pd.testing.assert_frame_equal(X_train_s, X_train)


# ── target_encode ──────────────────────────────────────────────────────────────

def test_target_encode_replaces_col_with_te():
    X_train = pd.DataFrame({"route": ["DAC_DXB", "DAC_DEL", "DAC_DXB"]})
    y_train = pd.Series([100.0, 50.0, 120.0])
    X_val   = pd.DataFrame({"route": ["DAC_DXB"]})
    X_test  = pd.DataFrame({"route": ["DAC_DEL"]})
    X_tr, X_v, X_te = target_encode(X_train, y_train, X_val, X_test, ("route",))
    assert "route_te" in X_tr.columns
    assert "route" not in X_tr.columns


def test_target_encode_uses_train_means():
    X_train = pd.DataFrame({"route": ["A", "A", "B"]})
    y_train = pd.Series([100.0, 200.0, 50.0])
    X_val   = pd.DataFrame({"route": ["A"]})
    X_test  = pd.DataFrame({"route": ["B"]})
    X_tr, X_v, X_te = target_encode(X_train, y_train, X_val, X_test, ("route",))
    assert X_tr["route_te"].iloc[0] == pytest.approx(150.0)  # mean of A
    assert X_v["route_te"].iloc[0] == pytest.approx(150.0)   # same mean used
    assert X_te["route_te"].iloc[0] == pytest.approx(50.0)   # mean of B


def test_target_encode_unseen_category_uses_global_mean():
    X_train = pd.DataFrame({"route": ["A", "B"]})
    y_train = pd.Series([100.0, 200.0])
    X_val   = pd.DataFrame({"route": ["C"]})  # unseen
    X_test  = pd.DataFrame({"route": ["A"]})
    X_tr, X_v, X_te = target_encode(X_train, y_train, X_val, X_test, ("route",))
    global_mean = y_train.mean()
    assert X_v["route_te"].iloc[0] == pytest.approx(global_mean)


def test_target_encode_skips_absent_col():
    X_train = pd.DataFrame({"fare": [100.0]})
    y_train = pd.Series([100.0])
    X_val   = pd.DataFrame({"fare": [150.0]})
    X_test  = pd.DataFrame({"fare": [180.0]})
    X_tr, X_v, X_te = target_encode(X_train, y_train, X_val, X_test, ("route",))
    assert "route_te" not in X_tr.columns
