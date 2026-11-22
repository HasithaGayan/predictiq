import joblib
import pandas as pd
import pytest

from src.predictiq.day64_eda import (
    FEATURES,
    split_chronologically,
)

from src.predictiq.features import DemandFeatureEngineer

from src.predictiq.day66_experiment import (
    expanding_date_folds,
    run_day66,
)


def toy_data():

    dates = pd.date_range(
        "2011-01-01",
        periods=50,
        freq="D",
    )

    return pd.DataFrame({
        "dteday": dates.astype(str),
        "cnt": [30 + i * 2 for i in range(50)],
        "season": [1] * 50,
        "yr": [0] * 50,
        "mnth": [1] * 50,
        "hr": [i % 24 for i in range(50)],
        "holiday": [0] * 50,
        "weekday": [i % 7 for i in range(50)],
        "workingday": [
            int(i % 7 < 5) for i in range(50)
        ],
        "weather_code": [1] * 50,
        "temp": [
            0.2 + i * 0.005 for i in range(50)
        ],
        "atemp": [0.3] * 50,
        "hum": [0.5] * 50,
        "windspeed": [0.1] * 50,
        "is_weekend": [0] * 50,
        "hour_sin": [0.5] * 50,
        "hour_cos": [-0.8] * 50,
    })


def test_features_use_only_prediction_inputs():

    raw = toy_data()[FEATURES]

    engineered = (
        DemandFeatureEngineer()
        .fit_transform(raw)
    )

    assert "commuter_peak" in engineered
    assert "temperature_humidity" in engineered
    assert len(engineered) == len(raw)
    assert "commuter_peak" not in raw


def test_expanding_folds_never_use_future_days():

    train, _, _ = split_chronologically(
        toy_data()
    )

    for fit_mask, check_mask in expanding_date_folds(train):

        assert (
            train.iloc[fit_mask]["dteday"].max()
            < train.iloc[check_mask]["dteday"].min()
        )

        assert not (fit_mask & check_mask).any()


def test_day66_trains_and_saves_reusable_pipeline(
    tmp_path
):

    csv = tmp_path / "clean.csv"
    data = toy_data()

    data.to_csv(csv, index=False)
    original = csv.read_bytes()

    options = [
        {
            "name": "standard",
            "engineered": False,
            "trees": 3,
            "depth": 5,
            "leaf": 2,
        },
        {
            "name": "engineered",
            "engineered": True,
            "trees": 3,
            "depth": 5,
            "leaf": 2,
        },
    ]

    report = run_day66(
        csv,
        tmp_path / "results",
        configs=options,
        cv_splits=2,
        reference_trees=3,
    )

    assert len(report["cv_results"]) == 2
    assert report["test_rows_untouched"] == 8

    assert report["selected_model"] in (
        "day65_reference",
        "day66_cv_selected",
    )

    model = joblib.load(
        tmp_path / "results" / "selected_pipeline.joblib"
    )

    assert len(
        model.predict(data[FEATURES].head(2))
    ) == 2

    assert csv.read_bytes() == original


def test_invalid_fold_count():

    train, _, _ = split_chronologically(
        toy_data()
    )

    with pytest.raises(ValueError):
        list(
            expanding_date_folds(
                train,
                n_splits=1,
            )
        )


def test_missing_dataset(tmp_path):

    with pytest.raises(FileNotFoundError):
        run_day66(
            tmp_path / "missing.csv",
            tmp_path / "output",
        )
