"""FastAPI inference server — loads champion model and serves predictions."""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from serving.predictor import AVAILABLE_MODELS, Predictor

# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Flight Fare Predictor",
    description="Predict Bangladesh airline ticket fares (BDT)",
    version="1.0.0",
)
logger = logging.getLogger(__name__)

_predictor: Predictor | None = None


# ── Request / Response schemas ─────────────────────────────────────────────────


class FareRequest(BaseModel):
    airline: str = Field(example="Emirates")
    source: str = Field(example="DAC")
    destination: str = Field(example="DXB")
    travel_class: str = Field(example="Economy")
    aircraft_type: str = Field(example="Boeing 777")
    booking_source: str = Field(example="Online Website")
    seasonality: str = Field(example="Regular")
    stopovers: int = Field(example=0, ge=0, le=2)
    duration: float = Field(example=4.5, gt=0)
    days_left: int = Field(example=30, ge=1)
    departure_hour: int = Field(example=8, ge=0, le=23)
    departure_day_of_week: int = Field(example=2, ge=0, le=6)
    departure_month: int = Field(example=6, ge=1, le=12)
    arrival_hour: int = Field(example=12, ge=0, le=23)
    arrival_day_of_week: int = Field(example=2, ge=0, le=6)
    arrival_month: int = Field(example=6, ge=1, le=12)
    model: str = Field(default="stacking", description=f"One of: {AVAILABLE_MODELS}")


class FareResponse(BaseModel):
    predicted_fare_bdt: float
    model_used: str


# ── Startup ────────────────────────────────────────────────────────────────────


@app.on_event("startup")
async def load_model() -> None:
    global _predictor
    _predictor = Predictor("stacking")
    logger.info("Predictor loaded (default: stacking)")


# ── Routes ─────────────────────────────────────────────────────────────────────


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": _predictor is not None}


@app.post("/predict", response_model=FareResponse)
def predict(req: FareRequest) -> FareResponse:
    if _predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Allow per-request model override
    if req.model != _predictor.model_name:
        try:
            p = Predictor(req.model)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Could not load model '{req.model}': {exc}") from exc
    else:
        p = _predictor

    inputs = req.model_dump(exclude={"model"})
    fare = p.predict(inputs)

    return FareResponse(predicted_fare_bdt=round(fare, 2), model_used=req.model)
