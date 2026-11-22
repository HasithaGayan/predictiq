
import joblib
import pandas as pd
import pytest

from src.predictiq.day64_eda import FEATURES
from src.predictiq.day65_train import (
    build_candidates,
    train_and_compare,
)


def sample_data():
    dates = pd.date_range(
        "2011-01-01", periods=40, freq="D"
    )

    return pd.DataFrame({
        "dteday": dates.astype(str),
        "cnt": [20 + i * 2 for i in range(40)],
        "season": [1] * 40,
        "yr": [0] * 40,
        "mnth": [1] * 40,
        "hr": [i % 24 for i in range(40)],
        "holiday": [0] * 40,
        "weekday": [i % 7 for i in range(40)],
        "workingday": [i % 2 for i in range(40)],
        "weather_code": [1] * 40,
        "temp": [0.2 + i * 0.01 for i in range(40)],
        "atemp": [0.3] * 40,
        "hum": [0.5] * 40,
        "windspeed": [0.1] * 40,
        "is_weekend": [0] * 40,
        "hour_sin": [0.5] * 40,
        "hour_cos": [-0.8] * 40,
    })


def test_three_candidate_models():
    candidates = build_candidates(forest_trees=5)

    assert set(candidates) == {
        "mean_baseline", "ridge", "random_forest"
    }


def test_training_and_model_artifact(tmp_path):
    source = tmp_path / "clean.csv"
    data = sample_data()
    data.to_csv(source, index=False)

    original = source.read_bytes()

    output = tmp_path / "results"

    report = train_and_compare(
        source,
        output,
        forest_trees=5,
    )

    assert len(report["models"]) == 3
    assert report["training_rows"] == 28
    assert report["validation_rows"] == 6
    assert report["test_rows_held_out"] == 6

    # Model selection must use validation MAE.
    smallest_mae = min(
        result["validation_mae"]
        for result in report["models"]
    )

    assert report["best_validation_mae"] == smallest_mae

    assert (output / "best_pipeline.joblib").is_file()
    assert (output / "training_report.json").is_file()

    # Never modify the input dataset.
    assert source.read_bytes() == original

    saved_model = joblib.load(
        output / "best_pipeline.joblib"
    )

    predictions = saved_model.predict(
        data[FEATURES].iloc[:2]
    )

    assert len(predictions) == 2


def test_missing_dataset_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        train_and_compare(
            tmp_path / "missing.csv",
            tmp_path / "results",
        )


def test_invalid_tree_count_rejected():
    with pytest.raises(ValueError):
        build_candidates(forest_trees=0)
