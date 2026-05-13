"""Tests for features/cleaning.py"""
from __future__ import annotations

import pandas as pd
import pytest

from features.cleaning import (
    CleaningReport,
    clean,
    drop_leakage_columns,
    drop_unnamed_columns,
    encode_stopovers,
    extract_temporal_features,
    impute_missing,
    rename_columns,
    COLUMN_RENAME_MAP,
)


# ── drop_unnamed_columns ───────────────────────────────────────────────────────

def test_drop_unnamed_columns_removes_them():
    df = pd.DataFrame({"Unnamed: 0": [1], "Unnamed: 1": [2], "fare": [100]})
    result = drop_unnamed_columns(df)
    assert "Unnamed: 0" not in result.columns
    assert "fare" in result.columns


def test_drop_unnamed_columns_noop_when_clean():
    df = pd.DataFrame({"airline": ["Air India"], "fare": [100]})
    result = drop_unnamed_columns(df)
    assert list(result.columns) == ["airline", "fare"]


# ── rename_columns ─────────────────────────────────────────────────────────────

def test_rename_columns_applies_map():
    df = pd.DataFrame({"Airline": ["IndiGo"], "Total Fare (BDT)": [50000]})
    result = rename_columns(df, COLUMN_RENAME_MAP)
    assert "airline" in result.columns
    assert "fare" in result.columns
    assert "Airline" not in result.columns


def test_rename_columns_ignores_missing():
    df = pd.DataFrame({"SomeOtherCol": [1]})
    result = rename_columns(df, COLUMN_RENAME_MAP)
    assert "SomeOtherCol" in result.columns


# ── drop_leakage_columns ───────────────────────────────────────────────────────

def test_drop_leakage_removes_target_cols():
    df = pd.DataFrame({
        "base_fare": [1000],
        "tax_surcharge": [500],
        "fare": [1500],
    })
    result = drop_leakage_columns(df, frozenset({"base_fare", "tax_surcharge"}))
    assert "base_fare" not in result.columns
    assert "tax_surcharge" not in result.columns
    assert "fare" in result.columns


def test_drop_leakage_noop_when_absent():
    df = pd.DataFrame({"fare": [1500]})
    result = drop_leakage_columns(df, frozenset({"base_fare"}))
    assert list(result.columns) == ["fare"]


# ── impute_missing ─────────────────────────────────────────────────────────────

def test_impute_median_fills_numeric_nan():
    df = pd.DataFrame({"duration": [2.0, None, 6.0], "airline": ["Air India"] * 3})
    result = impute_missing(df, "median", "Unknown")
    assert result["duration"].isna().sum() == 0
    assert result["duration"].iloc[1] == 4.0


def test_impute_fills_categorical_nan():
    df = pd.DataFrame({"airline": ["Air India", None, "IndiGo"]})
    result = impute_missing(df, "median", "Unknown")
    assert result["airline"].iloc[1] == "Unknown"


def test_impute_noop_when_no_missing():
    df = pd.DataFrame({"fare": [100.0, 200.0]})
    result = impute_missing(df, "median", "Unknown")
    pd.testing.assert_frame_equal(result, df)


# ── encode_stopovers ───────────────────────────────────────────────────────────

def test_encode_stopovers_ordinal_values():
    df = pd.DataFrame({"stopovers": ["Direct", "1 Stop", "2 Stops"]})
    result = encode_stopovers(df, "stopovers", {"Direct": 0, "1 Stop": 1, "2 Stops": 2})
    assert list(result["stopovers"]) == [0, 1, 2]


def test_encode_stopovers_raises_on_unknown_value():
    df = pd.DataFrame({"stopovers": ["Direct", "3 Stops"]})
    with pytest.raises(ValueError, match="Unexpected values"):
        encode_stopovers(df, "stopovers", {"Direct": 0, "1 Stop": 1, "2 Stops": 2})


def test_encode_stopovers_noop_when_col_absent():
    df = pd.DataFrame({"fare": [100]})
    result = encode_stopovers(df, "stopovers", {"Direct": 0})
    pd.testing.assert_frame_equal(result, df)


# ── extract_temporal_features ──────────────────────────────────────────────────

def test_extract_temporal_features_creates_hour_dow_month():
    df = pd.DataFrame({
        "departure_datetime": pd.to_datetime(["2024-06-15 08:30:00"]),
    })
    result = extract_temporal_features(df, ("departure_datetime",))
    assert "departure_hour" in result.columns
    assert "departure_day_of_week" in result.columns
    assert "departure_month" in result.columns
    assert result["departure_hour"].iloc[0] == 8
    assert result["departure_month"].iloc[0] == 6


def test_extract_temporal_features_drops_original_col():
    df = pd.DataFrame({
        "departure_datetime": pd.to_datetime(["2024-06-15 08:30:00"]),
    })
    result = extract_temporal_features(df, ("departure_datetime",))
    assert "departure_datetime" not in result.columns


def test_extract_temporal_features_skips_missing_col():
    df = pd.DataFrame({"fare": [100]})
    result = extract_temporal_features(df, ("departure_datetime",))
    pd.testing.assert_frame_equal(result, df)


# ── clean() integration ────────────────────────────────────────────────────────

def test_clean_returns_dataframe_and_report(raw_df, minimal_cfg):
    result, report = clean(raw_df, minimal_cfg)
    assert isinstance(result, pd.DataFrame)
    assert isinstance(report, CleaningReport)


def test_clean_removes_leakage_columns(raw_df, minimal_cfg):
    result, _ = clean(raw_df, minimal_cfg)
    assert "Base Fare (BDT)" not in result.columns
    assert "Tax & Surcharge (BDT)" not in result.columns
    assert "base_fare" not in result.columns
    assert "tax_surcharge" not in result.columns


def test_clean_renames_columns(raw_df, minimal_cfg):
    result, _ = clean(raw_df, minimal_cfg)
    assert "fare" in result.columns
    assert "airline" in result.columns
    assert "Total Fare (BDT)" not in result.columns


def test_clean_adds_temporal_features(raw_df, minimal_cfg):
    result, _ = clean(raw_df, minimal_cfg)
    assert "departure_hour" in result.columns
    assert "arrival_month" in result.columns


def test_clean_encodes_stopovers_numerically(raw_df, minimal_cfg):
    result, _ = clean(raw_df, minimal_cfg)
    assert result["stopovers"].dtype in ("int8", "int64", "int32")


def test_clean_no_missing_values_after(raw_df, minimal_cfg):
    result, _ = clean(raw_df, minimal_cfg)
    assert result.isna().sum().sum() == 0


def test_clean_report_shape_consistency(raw_df, minimal_cfg):
    result, report = clean(raw_df, minimal_cfg)
    assert report.final_shape == result.shape
    assert report.initial_shape[0] == len(raw_df)
