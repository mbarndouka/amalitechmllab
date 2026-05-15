"""Shared prediction logic — used by both FastAPI and Streamlit."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Allow running from project root without installing the package
_SRC = Path(__file__).parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

_ROOT = _SRC.parent
_MODELS_DIR = _ROOT / "models"
_FEATURES_DIR = _ROOT / "data" / "features"
_PROCESSED_PATH = _ROOT / "data" / "processed" / "Flight_Price_Dataset_of_Bangladesh.parquet"
_METRICS_PATH = _ROOT / "reports" / "model_comparison.json"

AVAILABLE_MODELS = [
    "stacking",
    "random_forest",
    "gradient_boosting",
    "xgboost",
    "xgboost_optuna",
    "decision_tree",
    "ridge",
    "lasso",
    "linear_regression",
]

_NUMERICAL_COLS = (
    "duration",
    "days_left",
    "stopovers",
    "departure_hour",
    "departure_day_of_week",
    "departure_month",
    "arrival_hour",
    "arrival_day_of_week",
    "arrival_month",
)

_LOG_NUMERIC_COLS = ("duration", "days_left")

_CATEGORICAL_COLS = (
    "airline",
    "source",
    "destination",
    "aircraft_type",
    "travel_class",
    "booking_source",
    "seasonality",
)


class Predictor:
    def __init__(self, model_name: str = "stacking") -> None:
        self.model_name = model_name
        self.model = joblib.load(_MODELS_DIR / f"{model_name}.pkl")

        with open(_FEATURES_DIR / "scaler.pkl", "rb") as f:
            self.scaler = pickle.load(f)

        route_te_path = _FEATURES_DIR / "route_te_map.pkl"
        if route_te_path.exists():
            with open(route_te_path, "rb") as f:
                full_map: dict = pickle.load(f)
            route_map = dict(full_map.get("route", {}))
            self._route_global_mean: float = float(route_map.pop("__global__", 10.6267))
            self._route_map: dict[str, float] = route_map
        else:
            self._route_map, self._route_global_mean = self._build_route_map_fallback()

        self._feature_cols: list[str] = list(
            pd.read_parquet(_FEATURES_DIR / "X_train.parquet").columns
        )
        self._options, self._source_names, self._dest_names = self._load_options()

    def _build_route_map_fallback(self) -> tuple[dict[str, float], float]:
        """Compute route→mean_log_fare from processed data when route_te_map.pkl missing."""
        df = pd.read_parquet(_PROCESSED_PATH)
        df["route"] = df["source"] + "_" + df["destination"]
        df["log_fare"] = np.log1p(df["fare"])
        route_map = df.groupby("route")["log_fare"].mean().to_dict()
        global_mean = float(df["log_fare"].mean())
        return route_map, global_mean

    def _load_options(self) -> tuple[dict[str, list], dict[str, str], dict[str, str]]:
        df = pd.read_parquet(_PROCESSED_PATH)
        options: dict[str, list] = {}
        for col in _CATEGORICAL_COLS:
            if col in df.columns:
                options[col] = sorted(df[col].dropna().unique().tolist())

        source_names: dict[str, str] = {}
        dest_names: dict[str, str] = {}
        if "source_name" in df.columns:
            source_names = dict(df[["source", "source_name"]].drop_duplicates().values)
        if "destination_name" in df.columns:
            dest_names = dict(df[["destination", "destination_name"]].drop_duplicates().values)

        return options, source_names, dest_names

    @property
    def options(self) -> dict[str, list]:
        return self._options

    @property
    def source_names(self) -> dict[str, str]:
        return self._source_names

    @property
    def dest_names(self) -> dict[str, str]:
        return self._dest_names

    def predict(self, inputs: dict) -> float:
        """Return predicted fare in BDT."""
        df = self._preprocess(inputs)
        raw = self.model.predict(df)
        return float(np.expm1(raw[0]))

    def _preprocess(self, inputs: dict) -> pd.DataFrame:
        row: dict = {
            "airline": inputs["airline"],
            "source": inputs["source"],
            "destination": inputs["destination"],
            "aircraft_type": inputs["aircraft_type"],
            "travel_class": inputs["travel_class"],
            "booking_source": inputs["booking_source"],
            "seasonality": inputs["seasonality"],
            "stopovers": float(inputs["stopovers"]),
            "duration": float(inputs["duration"]),
            "days_left": float(inputs["days_left"]),
            "departure_hour": int(inputs["departure_hour"]),
            "departure_day_of_week": int(inputs["departure_day_of_week"]),
            "departure_month": int(inputs["departure_month"]),
            "arrival_hour": int(inputs["arrival_hour"]),
            "arrival_day_of_week": int(inputs["arrival_day_of_week"]),
            "arrival_month": int(inputs["arrival_month"]),
        }
        df = pd.DataFrame([row])

        # Route target encoding (replaces OHE for route)
        route_key = f"{inputs['source']}_{inputs['destination']}"
        df["route_te"] = self._route_map.get(route_key, self._route_global_mean)

        # log1p on right-skewed numericals
        for col in _LOG_NUMERIC_COLS:
            df[col] = np.log1p(df[col].clip(lower=0))

        # OHE categoricals (not route — already target-encoded)
        df = pd.get_dummies(df, columns=list(_CATEGORICAL_COLS), dtype="uint8")

        # Scale numericals
        if hasattr(self.scaler, "feature_names_in_"):
            cols_to_scale = [c for c in _NUMERICAL_COLS if c in df.columns and c in self.scaler.feature_names_in_]
            if cols_to_scale:
                scaled = self.scaler.transform(df[cols_to_scale])
                for i, col in enumerate(cols_to_scale):
                    df[col] = scaled[:, i]

        # Align to training feature columns (add missing as 0, reorder)
        for col in self._feature_cols:
            if col not in df.columns:
                df[col] = 0
        df = df[self._feature_cols]

        return df
