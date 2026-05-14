"""FastAPI inference server — loads champion model from MLflow registry and serves predictions."""
from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path
from typing import Any

import mlflow.pyfunc
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Flight Fare Predictor",
    description="Predict Bangladesh airline ticket fares (BDT)",
    version="1.0.0",
)
logger = logging.getLogger(__name__)

# Loaded once at startup — not per request
_model = None
_scaler = None
_feature_names: list[str] = []


# ── Request / Response schemas ─────────────────────────────────────────────────

class FareRequest(BaseModel):
    airline: str                    = Field(example="Emirates")
    source: str                     = Field(example="DAC")
    destination: str                = Field(example="DXB")
    travel_class: str               = Field(example="Economy")
    aircraft_type: str              = Field(example="Boeing 777")
    booking_source: str             = Field(example="Online Website")
    seasonality: str                = Field(example="Regular")
    stopovers: int                  = Field(example=0, ge=0, le=2)
    duration: float                 = Field(example=4.5, gt=0)
    days_left: int                  = Field(example=30, ge=1)
    departure_hour: int             = Field(example=8, ge=0, le=23)
    departure_day_of_week: int      = Field(example=2, ge=0, le=6)
    departure_month: int            = Field(example=6, ge=1, le=12)
    arrival_hour: int               = Field(example=12, ge=0, le=23)
    arrival_day_of_week: int        = Field(example=2, ge=0, le=6)
    arrival_month: int              = Field(example=6, ge=1, le=12)


class FareResponse(BaseModel):
    predicted_fare_bdt: float
    model_source: str


# ── Model loading ──────────────────────────────────────────────────────────────

def _load_model_from_registry() -> Any:
    """Load champion model from MLflow registry."""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    registry_name = os.getenv("MLFLOW_REGISTRY_NAME", "FarePredictor")
    mlflow.set_tracking_uri(tracking_uri)
    model_uri = f"models:/{registry_name}@champion"
    model = mlflow.pyfunc.load_model(model_uri)
    return model, f"mlflow:{model_uri}"


def _load_model_from_file() -> Any:
    """Fallback: load best available .pkl from models/ directory."""
    model_dir = Path(os.getenv("MODELS_DIR", "models"))
    # Preference order
    candidates = ["linear_regression.pkl", "xgboost.pkl", "random_forest.pkl", "gradient_boosting.pkl"]
    for name in candidates:
        path = model_dir / name
        if path.exists():
            import joblib
            return joblib.load(path), f"file:{path}"
    raise FileNotFoundError(f"No model found in {model_dir}")


@app.on_event("startup")
async def load_model() -> None:
    global _model, _scaler, _feature_names

    # Try MLflow registry first, fall back to pkl file
    try:
        _model, source = _load_model_from_registry()
    except Exception as exc:
        logger.exception("Failed to load MLflow registry model; falling back to local model files: %s", exc)
        _model, source = _load_model_from_file()

    # Load scaler saved during feature engineering
    scaler_path = Path(os.getenv("FEATURES_DIR", "data/features")) / "scaler.pkl"
    if scaler_path.exists():
        with open(scaler_path, "rb") as f:
            _scaler = pickle.load(f)

    print(f"Model loaded from: {source}")


# ── Feature engineering (inference-time) ──────────────────────────────────────

_NUMERICAL_COLS = ["duration", "days_left", "stopovers",
                   "departure_hour", "departure_day_of_week", "departure_month",
                   "arrival_hour", "arrival_day_of_week", "arrival_month"]

_LOG_NUMERIC_COLS = ["duration", "days_left"]


def _build_feature_vector(req: FareRequest) -> pd.DataFrame:
    """Replicate the feature engineering pipeline for a single request.

    WHY: The model was trained on OHE + log-transformed + scaled features.
    Inference must apply exactly the same transforms, in the same order.
    """
    raw = {
        "airline":              req.airline,
        "source":               req.source,
        "destination":          req.destination,
        "travel_class":         req.travel_class,
        "aircraft_type":        req.aircraft_type,
        "booking_source":       req.booking_source,
        "seasonality":          req.seasonality,
        "stopovers":            float(req.stopovers),
        "duration":             req.duration,
        "days_left":            float(req.days_left),
        "departure_hour":       req.departure_hour,
        "departure_day_of_week": req.departure_day_of_week,
        "departure_month":      req.departure_month,
        "arrival_hour":         req.arrival_hour,
        "arrival_day_of_week":  req.arrival_day_of_week,
        "arrival_month":        req.arrival_month,
        "route":                f"{req.source}_{req.destination}",
    }
    df = pd.DataFrame([raw])

    # log1p transform numerical cols (matches feature engineering pipeline)
    for col in _LOG_NUMERIC_COLS:
        if col in df.columns:
            df[col] = np.log1p(df[col].clip(lower=0))

    # one-hot encode categoricals
    categorical_cols = ["airline", "source", "destination", "travel_class",
                        "aircraft_type", "booking_source", "seasonality", "route"]
    df = pd.get_dummies(df, columns=categorical_cols, dtype="uint8")

    # align to model's expected feature columns
    if _scaler is not None and hasattr(_scaler, "feature_names_in_"):
        # apply scaler to numerical cols that exist in the request
        num_present = [c for c in _NUMERICAL_COLS if c in df.columns]
        num_in_scaler = [c for c in num_present if c in _scaler.feature_names_in_]
        if num_in_scaler:
            scaled = _scaler.transform(df[num_in_scaler])
            for i, col in enumerate(num_in_scaler):
                df[col] = scaled[:, i]

    return df


def _align_features(df: pd.DataFrame, expected_cols: list[str]) -> pd.DataFrame:
    """Add missing OHE columns as 0, drop unexpected columns, reorder to match training."""
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0
    return df[expected_cols]


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/predict", response_model=FareResponse)
def predict(req: FareRequest) -> FareResponse:
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    df = _build_feature_vector(req)

    # Align feature columns to what model expects
    if hasattr(_model, "feature_names_in_"):
        expected = list(_model.feature_names_in_)
        df = _align_features(df, expected)
    elif hasattr(_model, "metadata") and _model.metadata:
        # MLflow pyfunc model — get schema from metadata if available
        pass

    raw_pred = _model.predict(df)

    # Inverse log1p transform — model predicts log(fare+1), we return BDT
    predicted_bdt = float(np.expm1(raw_pred[0]))

    return FareResponse(
        predicted_fare_bdt=round(predicted_bdt, 2),
        model_source="champion",
    )
