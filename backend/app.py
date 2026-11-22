
"""PredictIQ: historical bike-demand prediction API."""

from contextlib import asynccontextmanager
from pathlib import Path
import math

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, StrictBool

from src.predictiq.day66_experiment import FEATURES


ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    ROOT
    / "reports"
    / "generated"
    / "day66"
    / "selected_pipeline.joblib"
)

WARNING = (
    "Historical demonstration using 2011-2012 "
    "Washington, DC bike-sharing data. "
    "Not a verified forecast for current conditions."
)


class PredictionInput(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
    )

    # UCI dataset year coding: 0=2011, 1=2012.
    yr: int = Field(ge=0, le=1, strict=True)

    # Historical dataset season code.
    season: int = Field(ge=1, le=4, strict=True)

    mnth: int = Field(ge=1, le=12, strict=True)
    hr: int = Field(ge=0, le=23, strict=True)

    # Dataset weekday: 0=Sunday, 6=Saturday.
    weekday: int = Field(ge=0, le=6, strict=True)

    holiday: StrictBool

    weather_code: int = Field(
        ge=1, le=4, strict=True
    )

    # These weather inputs use the same normalized
    # 0-to-1 representation as the training dataset.
    temp: float = Field(ge=0, le=1)
    atemp: float = Field(ge=0, le=1)
    hum: float = Field(ge=0, le=1)
    windspeed: float = Field(ge=0, le=1)


def build_model_features(payload: PredictionInput):
    """Convert validated user inputs to the 15 model features."""

    is_weekend = int(payload.weekday in (0, 6))

    workingday = int(
        not is_weekend and not payload.holiday
    )

    angle = 2 * math.pi * payload.hr / 24

    values = {
        "season": payload.season,
        "yr": payload.yr,
        "mnth": payload.mnth,
        "hr": payload.hr,
        "holiday": int(payload.holiday),
        "weekday": payload.weekday,
        "workingday": workingday,
        "weather_code": payload.weather_code,
        "temp": payload.temp,
        "atemp": payload.atemp,
        "hum": payload.hum,
        "windspeed": payload.windspeed,
        "is_weekend": is_weekend,
        "hour_sin": math.sin(angle),
        "hour_cos": math.cos(angle),
    }

    expected = list(FEATURES)

    if set(values) != set(expected):
        raise ValueError(
            "API fields do not match trained model features"
        )

    return pd.DataFrame([values], columns=expected)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODEL_PATH.is_file():
        raise RuntimeError(
            "Trained Day 66 model is missing. "
            "Run the documented model preparation first."
        )

    # Only load our own trusted local model file.
    # Never accept uploaded joblib/pickle files.
    app.state.pipeline = joblib.load(MODEL_PATH)

    yield

    del app.state.pipeline


app = FastAPI(
    title="PredictIQ Prediction API",
    version="0.1.0",
    description=(
        "Educational historical bike-rental demand "
        "prediction using a trained Random Forest."
    ),
    lifespan=lifespan,
)


@app.get("/api/health")
def health(request: Request):
    return {
        "status": "ok",
        "model_loaded": hasattr(
            request.app.state,
            "pipeline",
        ),
    }


@app.get("/api/model-info")
def model_info():
    return {
        "model": "Day 66 selected Random Forest pipeline",
        "dataset": "UCI Bike Sharing",
        "training_period": "2011-2012",
        "feature_count": len(FEATURES),
        "validation_mae": 47.68,
        "validation_rmse": 74.14,
        "warning": WARNING,
    }


@app.post("/api/predict")
def predict(payload: PredictionInput, request: Request):
    pipeline = getattr(
        request.app.state,
        "pipeline",
        None,
    )

    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Prediction model not ready",
        )

    inputs = build_model_features(payload)

    try:
        result = pipeline.predict(inputs)

        if len(result) != 1:
            raise ValueError("Unexpected prediction shape")

        prediction = float(result[0])

        if not math.isfinite(prediction):
            raise ValueError("Nonfinite prediction")

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Model prediction failed",
        ) from exc

    return {
        "estimated_hourly_rentals": round(
            max(0.0, prediction),
            1,
        ),
        "unit": "rentals per hour",
        "model": "Random Forest",
        "historical_scenario": True,
        "warning": WARNING,
    }




# Serve the local frontend from the same FastAPI origin.
from fastapi.responses import FileResponse


@app.get("/", include_in_schema=False)
def frontend_home():
    page = ROOT / "frontend" / "index.html"

    if not page.is_file():
        raise HTTPException(
            status_code=404,
            detail="Frontend unavailable",
        )

    return FileResponse(page, media_type="text/html")

# Serve only the public brand assets, without exposing any project files.
from fastapi.staticfiles import StaticFiles

app.mount("/assets", StaticFiles(directory=ROOT / "frontend" / "assets"), name="assets")
