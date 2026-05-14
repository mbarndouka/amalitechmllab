"""Shared fixtures for the flight fare prediction test suite."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make src importable without installing the package
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture()
def minimal_cfg() -> dict:
    return {
        "data": {
            "test_size": 0.2,
            "val_size": 0.1,
            "random_state": 42,
            "features_dir": "data/features",
        },
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
        "cleaning": {
            "missingness_drop_threshold": 0.5,
            "numerical_impute_strategy": "median",
            "categorical_impute_value": "Unknown",
            "leakage_columns": ["Base Fare (BDT)", "Tax & Surcharge (BDT)"],
        },
    }


@pytest.fixture()
def raw_df() -> pd.DataFrame:
    """Minimal raw dataframe mirroring the real dataset schema."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame(
        {
            "Airline": np.random.choice(["Air India", "IndiGo", "Emirates"], n),
            "Source": np.random.choice(["DAC", "CGP"], n),
            "Source Name": ["Dhaka"] * n,
            "Destination": np.random.choice(["DXB", "DEL", "LHR"], n),
            "Destination Name": ["Dubai"] * n,
            "Stopovers": np.random.choice(["Direct", "1 Stop", "2 Stops"], n),
            "Aircraft Type": np.random.choice(["Boeing 737", "Airbus A320"], n),
            "Class": np.random.choice(["Economy", "Business"], n),
            "Booking Source": np.random.choice(["Online Website", "Travel Agency"], n),
            "Seasonality": np.random.choice(["Regular", "Eid"], n),
            "Duration (hrs)": np.random.uniform(2, 14, n),
            "Days Before Departure": np.random.randint(1, 180, n),
            "Total Fare (BDT)": np.random.uniform(5000, 200000, n),
            "Base Fare (BDT)": np.random.uniform(3000, 150000, n),
            "Tax & Surcharge (BDT)": np.random.uniform(1000, 50000, n),
            "Departure Date & Time": pd.date_range("2024-01-01", periods=n, freq="6h").astype(str),
            "Arrival Date & Time": pd.date_range("2024-01-01 04:00", periods=n, freq="6h").astype(str),
        }
    )


@pytest.fixture()
def cleaned_df(raw_df, minimal_cfg) -> pd.DataFrame:
    from features.cleaning import clean

    df, _ = clean(raw_df, minimal_cfg)
    return df
