import json

import pandas as pd
import pytest

from src.predictiq.clean_data import clean_dataset


@pytest.fixture
def sample(tmp_path):
    df = pd.DataFrame({
        "instant": [1, 2, 3],
        "dteday": [
            "2011-01-01",
            "2011-01-01",
            "2011-01-01",
        ],
        "season": [1, 1, 1],
        "yr": [0, 0, 0],
        "mnth": [1, 1, 1],
        "hr": [0, 1, 2],
        "holiday": [0, 0, 0],
        "weekday": [6, 6, 6],
        "workingday": [0, 0, 0],
        "weathersit": [1, 1, 1],
        "temp": [0.2, 0.3, 0.4],
        "atemp": [0.2, 0.3, 0.4],
        "hum": [0.5, 0.6, 0.7],
        "windspeed": [0.1, 0.2, 0.3],
        "casual": [1, 2, 1],
        "registered": [2, 3, 4],
        "cnt": [3, 5, 5],
    })

    raw = tmp_path / "hour.csv"
    df.to_csv(raw, index=False)

    return raw


def run_cleaning(sample):
    folder = sample.parent

    return clean_dataset(
        sample,
        folder / "processed.csv",
        folder / "report.json",
    )


def test_clean_dataset(sample):
    original = sample.read_bytes()

    report = run_cleaning(sample)

    output = pd.read_csv(
        sample.parent / "processed.csv"
    )

    assert report["raw_rows"] == 3
    assert report["cleaned_rows"] == 3
    assert len(output) == 3

    assert "casual" not in output.columns
    assert "registered" not in output.columns
    assert "instant" not in output.columns

    assert "weather_code" in output.columns
    assert "is_weekend" in output.columns
    assert "hour_sin" in output.columns

    assert output["is_weekend"].tolist() == [1, 1, 1]

    # Original raw file must stay unchanged.
    assert sample.read_bytes() == original


def test_quality_report(sample):
    run_cleaning(sample)

    report = json.loads(
        (sample.parent / "report.json").read_text(
            encoding="utf-8"
        )
    )

    assert report["missing_cells"] == 0
    assert report["target"] == "cnt"
    assert len(report["source_sha256"]) == 64


def test_missing_values_rejected(sample):
    df = pd.read_csv(sample)
    df.loc[0, "hum"] = None
    df.to_csv(sample, index=False)

    with pytest.raises(ValueError, match="missing"):
        run_cleaning(sample)


def test_invalid_category_rejected(sample):
    df = pd.read_csv(sample)
    df.loc[0, "weathersit"] = 9
    df.to_csv(sample, index=False)

    with pytest.raises(ValueError, match="Invalid category"):
        run_cleaning(sample)


def test_target_mismatch_rejected(sample):
    df = pd.read_csv(sample)
    df.loc[0, "cnt"] = 999
    df.to_csv(sample, index=False)

    with pytest.raises(ValueError, match="Target relationship"):
        run_cleaning(sample)


def test_duplicate_hour_rejected(sample):
    df = pd.read_csv(sample)

    df = pd.concat(
        [df, df.iloc[[0]]],
        ignore_index=True,
    )

    df.to_csv(sample, index=False)

    with pytest.raises(ValueError, match="Duplicate date/hour"):
        run_cleaning(sample)


def test_missing_column_rejected(sample):
    df = pd.read_csv(sample)
    df = df.drop(columns=["cnt"])
    df.to_csv(sample, index=False)

    with pytest.raises(ValueError, match="Missing required"):
        run_cleaning(sample)


def test_raw_overwrite_prevented(sample):
    with pytest.raises(ValueError, match="Never overwrite"):
        clean_dataset(
            sample,
            sample,
            sample.parent / "report.json",
        )