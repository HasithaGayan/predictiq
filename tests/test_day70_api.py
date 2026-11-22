
"""Day 70: real PredictIQ FastAPI integration tests."""

import math

import pytest
from fastapi.testclient import TestClient

from backend.app import (
    app,
    PredictionInput,
    build_model_features,
)
from src.predictiq.day66_experiment import FEATURES


EXAMPLE = {
    "yr": 0,
    "season": 1,
    "mnth": 1,
    "hr": 8,
    "weekday": 1,
    "holiday": False,
    "weather_code": 1,
    "temp": 0.24,
    "atemp": 0.2879,
    "hum": 0.81,
    "windspeed": 0.1,
}


@pytest.fixture(scope="module")
def client():
    # Loads the existing trusted Random Forest once.
    with TestClient(app) as test_client:
        yield test_client


def test_health_check(client):
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_loaded"] is True


def test_model_information(client):
    response = client.get("/api/model-info")

    assert response.status_code == 200

    info = response.json()

    assert info["feature_count"] == 15
    assert info["validation_mae"] == pytest.approx(47.68)
    assert info["training_period"] == "2011-2012"


def test_frontend_page(client):
    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "PredictIQ" in response.text
    assert 'id="prediction-form"' in response.text
    assert 'id="day-chart"' in response.text


def test_custom_logo(client):
    response = client.get(
        "/assets/predictiq-mark.svg"
    )

    assert response.status_code == 200
    assert "<svg" in response.text


def test_feature_engineering():
    payload = PredictionInput(**EXAMPLE)

    features = build_model_features(payload)

    assert list(features.columns) == list(FEATURES)
    assert features.shape == (1, 15)

    assert features.iloc[0]["workingday"] == 1
    assert features.iloc[0]["is_weekend"] == 0

    expected_sin = math.sin(
        2 * math.pi * 8 / 24
    )

    assert features.iloc[0]["hour_sin"] == pytest.approx(
        expected_sin
    )


def test_real_prediction(client):
    response = client.post(
        "/api/predict",
        json=EXAMPLE,
    )

    assert response.status_code == 200

    result = response.json()
    estimate = result["estimated_hourly_rentals"]

    assert isinstance(estimate, (int, float))
    assert math.isfinite(estimate)
    assert estimate >= 0

    assert result["historical_scenario"] is True
    assert result["unit"] == "rentals per hour"


@pytest.mark.parametrize(
    "field,value",
    [
        ("hr", 30),
        ("temp", -0.1),
        ("windspeed", 1.2),
        ("holiday", "false"),
    ],
)
def test_invalid_inputs_rejected(client, field, value):
    payload = {**EXAMPLE, field: value}

    response = client.post(
        "/api/predict",
        json=payload,
    )

    assert response.status_code == 422


def test_missing_field_rejected(client):
    payload = EXAMPLE.copy()
    payload.pop("hum")

    response = client.post(
        "/api/predict",
        json=payload,
    )

    assert response.status_code == 422


def test_unexpected_field_rejected(client):
    payload = {
        **EXAMPLE,
        "unknown_feature": 100,
    }

    response = client.post(
        "/api/predict",
        json=payload,
    )

    assert response.status_code == 422
