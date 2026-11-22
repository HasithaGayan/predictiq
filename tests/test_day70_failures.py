
"""Day 70: error handling and application failure tests."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import backend.app as api_module


VALID_INPUT = {
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


def test_invalid_hour_returns_422():
    """Invalid scenario values must be rejected."""

    with TestClient(api_module.app) as client:
        response = client.post(
            "/api/predict",
            json={**VALID_INPUT, "hr": 30},
        )

    assert response.status_code == 422


def test_unavailable_model_returns_503():
    """An unavailable model must not produce a fake result."""

    with TestClient(api_module.app) as client:
        original = api_module.app.state.pipeline

        try:
            # Temporarily simulate an unavailable model.
            api_module.app.state.pipeline = None

            response = client.post(
                "/api/predict",
                json=VALID_INPUT,
            )
        finally:
            # Restore the original model immediately.
            api_module.app.state.pipeline = original

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "Prediction model not ready"
    )


def test_missing_frontend_returns_404(
    monkeypatch,
    tmp_path,
):
    """Missing HTML must produce a clear 404 response."""

    with TestClient(api_module.app) as client:
        # Temporarily point frontend lookup to an empty folder.
        monkeypatch.setattr(
            api_module,
            "ROOT",
            tmp_path,
        )

        response = client.get("/")

    assert response.status_code == 404
    assert response.json()["detail"] == (
        "Frontend unavailable"
    )


def test_missing_model_blocks_startup(
    monkeypatch,
    tmp_path,
):
    """Do not start the API with missing model artifacts."""

    missing_model = Path(tmp_path) / "missing.joblib"

    monkeypatch.setattr(
        api_module,
        "MODEL_PATH",
        missing_model,
    )

    with pytest.raises(
        RuntimeError,
        match="Trained Day 66 model is missing",
    ):
        with TestClient(api_module.app):
            pass
