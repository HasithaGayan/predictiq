
import pandas as pd
import pytest

from src.predictiq.day64_eda import (
    FEATURES,
    split_chronologically,
    build_baseline,
    run_day64,
)


def toy_data():
    dates = pd.date_range(
        "2011-01-01",
        periods=20,
        freq="D",
    )

    return pd.DataFrame({
        "dteday": dates.astype(str),
        "cnt": [10 + i for i in range(20)],
        "season": [1] * 20,
        "yr": [0] * 20,
        "mnth": [1] * 20,
        "hr": [10] * 20,
        "holiday": [0] * 20,
        "weekday": [i % 7 for i in range(20)],
        "workingday": [i % 2 for i in range(20)],
        "weather_code": [1] * 20,
        "temp": [0.1 + i * 0.01 for i in range(20)],
        "atemp": [0.2] * 20,
        "hum": [0.5] * 20,
        "windspeed": [0.1] * 20,
        "is_weekend": [0] * 20,
        "hour_sin": [0.5] * 20,
        "hour_cos": [-0.8] * 20,
    })


def test_chronological_split_does_not_overlap():
    train, valid, test = split_chronologically(toy_data())

    assert [len(train), len(valid), len(test)] == [14, 3, 3]

    assert (
        train["dteday"].max()
        < valid["dteday"].min()
        < test["dteday"].min()
    )


def test_rejects_leakage_columns():
    data = toy_data()
    data["registered"] = 5

    with pytest.raises(ValueError, match="Target-leakage"):
        split_chronologically(data)


def test_baseline_fits_training_mean():
    train, valid, _ = split_chronologically(toy_data())

    model = build_baseline()
    model.fit(train[FEATURES], train["cnt"])

    predictions = model.predict(valid[FEATURES])

    assert predictions.tolist() == pytest.approx(
        [train["cnt"].mean()] * len(valid)
    )


def test_eda_does_not_change_processed_data(tmp_path):
    source = tmp_path / "hour_clean.csv"
    toy_data().to_csv(source, index=False)

    original = source.read_bytes()

    report = run_day64(
        source,
        tmp_path / "output",
    )

    assert source.read_bytes() == original
    assert len(report["charts"]) == 4

    assert (
        tmp_path / "output" / "day64_report.json"
    ).is_file()

    assert len(report["training_only_observations"]) == 3
