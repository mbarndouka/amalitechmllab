"""Integration tests for features/engineering.py — engineer() orchestrator."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from features.engineering import FeatureSet, engineer


@pytest.fixture()
def eng_cfg() -> dict:
    return {
        "data": {"test_size": 0.2, "val_size": 0.1, "random_state": 42},
        "features": {
            "target": "fare",
            "log_target": False,
            "redundant": ["source_name", "destination_name"],
            "categorical": ["airline", "travel_class"],
            "numerical": ["duration", "days_left", "stopovers"],
            "log_numerics": False,
            "log_numeric_cols": [],
            "target_encode_cols": [],
        },
    }


@pytest.fixture()
def eng_df() -> pd.DataFrame:
    """Cleaned dataframe ready for feature engineering (post-cleaning schema)."""
    np.random.seed(0)
    n = 200
    return pd.DataFrame({
        "airline":      np.random.choice(["Air India", "IndiGo", "Emirates"], n),
        "source":       np.random.choice(["DAC", "CGP"], n),
        "source_name":  ["Dhaka"] * n,
        "destination":  np.random.choice(["DXB", "DEL", "LHR"], n),
        "destination_name": ["Dubai"] * n,
        "travel_class": np.random.choice(["Economy", "Business"], n),
        "duration":     np.random.uniform(2, 14, n),
        "days_left":    np.random.randint(1, 180, n).astype(float),
        "stopovers":    np.random.randint(0, 3, n).astype(float),
        "fare":         np.random.uniform(5000, 200000, n),
    })


def test_engineer_returns_feature_set(eng_df, eng_cfg):
    fset = engineer(eng_df, eng_cfg)
    assert isinstance(fset, FeatureSet)


def test_engineer_splits_cover_full_dataset(eng_df, eng_cfg):
    fset = engineer(eng_df, eng_cfg)
    total = len(fset.X_train) + len(fset.X_val) + len(fset.X_test)
    assert total == len(eng_df)


def test_engineer_no_overlap_between_splits(eng_df, eng_cfg):
    fset = engineer(eng_df, eng_cfg)
    train_idx = set(fset.X_train.index)
    val_idx   = set(fset.X_val.index)
    test_idx  = set(fset.X_test.index)
    assert train_idx.isdisjoint(val_idx)
    assert train_idx.isdisjoint(test_idx)
    assert val_idx.isdisjoint(test_idx)


def test_engineer_drops_redundant_columns(eng_df, eng_cfg):
    fset = engineer(eng_df, eng_cfg)
    assert "source_name" not in fset.X_train.columns
    assert "destination_name" not in fset.X_train.columns


def test_engineer_target_not_in_features(eng_df, eng_cfg):
    fset = engineer(eng_df, eng_cfg)
    assert "fare" not in fset.X_train.columns


def test_engineer_ohe_expands_categoricals(eng_df, eng_cfg):
    fset = engineer(eng_df, eng_cfg)
    ohe_cols = [c for c in fset.X_train.columns if c.startswith("airline_") or c.startswith("travel_class_")]
    assert len(ohe_cols) > 0


def test_engineer_route_feature_created(eng_df, eng_cfg):
    fset = engineer(eng_df, eng_cfg)
    route_cols = [c for c in fset.X_train.columns if c.startswith("route_")]
    assert len(route_cols) > 0


def test_engineer_numerical_cols_are_scaled(eng_df, eng_cfg):
    fset = engineer(eng_df, eng_cfg)
    # scaled duration should have mean ~0 on train set
    assert fset.X_train["duration"].mean() == pytest.approx(0.0, abs=0.1)


def test_engineer_scaler_fitted_on_train_only(eng_df, eng_cfg):
    fset = engineer(eng_df, eng_cfg)
    # scaler must exist and have feature_names_in_
    assert hasattr(fset.scaler, "mean_")


def test_engineer_y_shapes_match_x(eng_df, eng_cfg):
    fset = engineer(eng_df, eng_cfg)
    assert len(fset.X_train) == len(fset.y_train)
    assert len(fset.X_val)   == len(fset.y_val)
    assert len(fset.X_test)  == len(fset.y_test)


def test_engineer_log_target_transforms_y(eng_df, eng_cfg):
    cfg_log = {**eng_cfg, "features": {**eng_cfg["features"], "log_target": True}}
    cfg_raw = eng_cfg

    fset_log = engineer(eng_df, cfg_log)
    fset_raw = engineer(eng_df, cfg_raw)

    # log-transformed y must be much smaller than raw fares
    assert fset_log.y_train.max() < 15        # log1p(200000) ≈ 12.2
    assert fset_raw.y_train.max() > 1000


def test_engineer_log_numerics_transforms_duration(eng_df, eng_cfg):
    cfg = {**eng_cfg, "features": {
        **eng_cfg["features"],
        "log_numerics": True,
        "log_numeric_cols": ["duration"],
    }}
    fset = engineer(eng_df, cfg)
    # After log1p + scaling, all duration values finite
    assert fset.X_train["duration"].isna().sum() == 0


def test_engineer_target_encode_replaces_col(eng_df, eng_cfg):
    cfg = {**eng_cfg, "features": {
        **eng_cfg["features"],
        "categorical": ["airline", "travel_class"],
        "target_encode_cols": ("route",),
    }}
    fset = engineer(eng_df, cfg)
    assert "route_te" in fset.X_train.columns
    assert "route" not in fset.X_train.columns


def test_engineer_is_reproducible(eng_df, eng_cfg):
    fset1 = engineer(eng_df, eng_cfg)
    fset2 = engineer(eng_df, eng_cfg)
    pd.testing.assert_frame_equal(fset1.X_train, fset2.X_train)
