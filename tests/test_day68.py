
"""Day 68: PyTorch neural-network tests."""

import json

import numpy as np
import pandas as pd
import pytest
import torch

from src.predictiq.day68_neural import (
    INPUT_FEATURES,
    TinyDemandNet,
    train_one,
    run_day68,
)


def sample_data(rows=60):
    i = np.arange(rows)
    hours = i % 24

    return pd.DataFrame({
        "dteday": pd.date_range(
            "2011-01-01",
            periods=rows,
            freq="D",
        ).strftime("%Y-%m-%d"),
        "cnt": 40 + i * 3,
        "season": [1] * rows,
        "yr": [0] * rows,
        "mnth": [1] * rows,
        "hr": hours,
        "holiday": [0] * rows,
        "weekday": i % 7,
        "workingday": (i % 7 < 5).astype(int),
        "weather_code": [1] * rows,
        "temp": 0.2 + (i % 12) * 0.05,
        "atemp": [0.4] * rows,
        "hum": 0.3 + (i % 10) * 0.05,
        "windspeed": 0.1 + (i % 5) * 0.02,
        "is_weekend": (i % 7 >= 5).astype(int),
        "hour_sin": np.sin(2 * np.pi * hours / 24),
        "hour_cos": np.cos(2 * np.pi * hours / 24),
    })


def test_input_features_exclude_target_leakage():
    forbidden = {"cnt", "casual", "registered", "instant"}

    assert forbidden.isdisjoint(INPUT_FEATURES)
    assert len(INPUT_FEATURES) == 7


def test_forward_pass_and_gradients():
    torch.manual_seed(42)

    model = TinyDemandNet()
    inputs = torch.randn(8, len(INPUT_FEATURES))

    predictions = model(inputs)

    assert predictions.shape == (8, 1)
    assert torch.isfinite(predictions).all().item()

    loss = torch.nn.MSELoss()(
        predictions,
        torch.ones_like(predictions),
    )

    loss.backward()

    assert any(
        parameter.grad is not None
        and torch.any(parameter.grad != 0).item()
        for parameter in model.parameters()
    )


def test_training_reduces_loss():
    torch.manual_seed(7)

    inputs = torch.randn(60, len(INPUT_FEATURES))
    targets = (
        0.8 * inputs[:, :1]
        + 0.2 * inputs[:, 1:2]
    )

    model, history = train_one(
        inputs,
        targets,
        inputs,
        targets,
        lr=0.01,
        epochs=75,
    )

    assert isinstance(model, TinyDemandNet)
    assert len(history["training_loss"]) == 75
    assert len(history["validation_loss"]) == 75

    assert (
        history["training_loss"][-1]
        < history["training_loss"][0]
    )

    assert np.isfinite(history["validation_loss"]).all()


def test_invalid_training_settings_rejected():
    inputs = torch.zeros(4, len(INPUT_FEATURES))
    targets = torch.zeros(4, 1)

    for invalid_lr in (0, -0.01):
        with pytest.raises(ValueError):
            train_one(
                inputs,
                targets,
                inputs,
                targets,
                lr=invalid_lr,
            )

    with pytest.raises(ValueError):
        train_one(
            inputs,
            targets,
            inputs,
            targets,
            lr=0.01,
            epochs=0,
        )


def test_training_artifacts_and_data_preservation(tmp_path):
    source = tmp_path / "clean.csv"
    output = tmp_path / "results"

    sample_data().to_csv(source, index=False)
    original = source.read_bytes()

    report = run_day68(
        source,
        output,
        epochs=3,
    )

    assert report["training_rows"] == 42
    assert report["validation_rows"] == 9
    assert report["test_rows_not_used"] == 9

    assert len(report["experiments"]) == 2
    assert report["selected_learning_rate"] in (
        0.01,
        0.001,
    )

    assert (output / "tiny_network.pt").is_file()
    assert (output / "learning_curves.png").stat().st_size > 0
    assert (output / "report.json").is_file()

    metadata = torch.load(
        output / "tiny_network.pt",
        map_location="cpu",
        weights_only=True,
    )

    assert metadata["features"] == INPUT_FEATURES
    assert "model_state_dict" in metadata
    assert len(metadata["feature_mean"]) == 7

    saved_report = json.loads(
        (output / "report.json").read_text(
            encoding="utf-8"
        )
    )

    assert saved_report["epochs"] == 3
    assert source.read_bytes() == original


def test_missing_and_invalid_data_rejected(tmp_path):
    missing = tmp_path / "missing.csv"
    output = tmp_path / "results"

    with pytest.raises(FileNotFoundError):
        run_day68(missing, output)

    source = tmp_path / "data.csv"
    sample_data().to_csv(source, index=False)

    with pytest.raises(ValueError):
        run_day68(source, output, epochs=0)

    invalid = sample_data()
    invalid.loc[0, "temp"] = np.nan
    invalid.to_csv(source, index=False)

    with pytest.raises(ValueError):
        run_day68(source, output, epochs=2)
