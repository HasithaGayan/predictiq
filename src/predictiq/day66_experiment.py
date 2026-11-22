"""Day 66: chronological cross-validation and tuning."""

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.predictiq.day64_eda import (
    CATEGORICAL,
    NUMERICAL,
    FEATURES,
    build_baseline,
    split_chronologically,
)

from src.predictiq.day65_train import build_candidates
from src.predictiq.features import DemandFeatureEngineer


CONFIGS = [
    {
        "name": "standard",
        "engineered": False,
        "trees": 40,
        "depth": 12,
        "leaf": 3,
    },
    {
        "name": "features_only",
        "engineered": True,
        "trees": 40,
        "depth": 12,
        "leaf": 3,
    },
    {
        "name": "deeper",
        "engineered": True,
        "trees": 70,
        "depth": 18,
        "leaf": 2,
    },
    {
        "name": "regularized",
        "engineered": True,
        "trees": 70,
        "depth": 12,
        "leaf": 5,
    },
]


def model_pipeline(config):

    if (
        config["trees"] < 1
        or config["leaf"] < 1
        or config["depth"] < 1
    ):
        raise ValueError("Model settings must be positive")

    numeric = NUMERICAL + (
        ["commuter_peak", "temperature_humidity"]
        if config["engineered"]
        else []
    )

    preprocess = ColumnTransformer([
        (
            "categorical",
            Pipeline([
                (
                    "imputer",
                    SimpleImputer(strategy="most_frequent"),
                ),
                (
                    "onehot",
                    OneHotEncoder(handle_unknown="ignore"),
                ),
            ]),
            CATEGORICAL,
        ),
        (
            "numeric",
            Pipeline([
                (
                    "imputer",
                    SimpleImputer(strategy="median"),
                ),
                ("scaler", StandardScaler()),
            ]),
            numeric,
        ),
    ])

    steps = []

    if config["engineered"]:
        steps.append(
            ("features", DemandFeatureEngineer())
        )

    steps.extend([
        ("preprocess", preprocess),
        (
            "model",
            RandomForestRegressor(
                n_estimators=config["trees"],
                max_depth=config["depth"],
                min_samples_leaf=config["leaf"],
                random_state=42,
                n_jobs=2,
            ),
        ),
    ])

    return Pipeline(steps)


def expanding_date_folds(training, n_splits=3):

    dates = np.array(
        sorted(training["dteday"].unique())
    )

    if n_splits < 2 or len(dates) <= n_splits:
        raise ValueError(
            "Need more training dates than CV folds"
        )

    splitter = TimeSeriesSplit(n_splits=n_splits)

    for early, later in splitter.split(dates):

        fit_rows = training["dteday"].isin(
            dates[early]
        ).to_numpy()

        check_rows = training["dteday"].isin(
            dates[later]
        ).to_numpy()

        if not fit_rows.any() or not check_rows.any():
            raise ValueError("Empty chronological fold")

        yield fit_rows, check_rows


def metrics(actual, predicted):

    return {
        "mae": round(
            float(mean_absolute_error(actual, predicted)),
            4,
        ),
        "rmse": round(
            float(np.sqrt(
                mean_squared_error(actual, predicted)
            )),
            4,
        ),
    }


def run_day66(
    clean_csv,
    out_dir,
    configs=None,
    cv_splits=3,
    reference_trees=100,
):

    clean_csv = Path(clean_csv)
    out_dir = Path(out_dir)

    if not clean_csv.is_file():
        raise FileNotFoundError(clean_csv)

    configurations = (
        CONFIGS if configs is None else configs
    )

    if not configurations:
        raise ValueError("No experiment configurations")

    data = pd.read_csv(clean_csv)

    train, valid, test = split_chronologically(data)

    # Cross-validation stays within training dates.
    folds = list(
        expanding_date_folds(train, cv_splits)
    )

    experiments = []

    for config in configurations:

        scores = []
        start = time.perf_counter()

        for fit_rows, check_rows in folds:

            fold_train = train.iloc[fit_rows]
            fold_check = train.iloc[check_rows]

            model = model_pipeline(config)

            model.fit(
                fold_train[FEATURES],
                fold_train["cnt"],
            )

            predicted = model.predict(
                fold_check[FEATURES]
            )

            scores.append(
                metrics(
                    fold_check["cnt"],
                    predicted,
                )["mae"]
            )

        experiments.append({
            "name": config["name"],
            "engineered": config["engineered"],
            "trees": config["trees"],
            "depth": config["depth"],
            "leaf": config["leaf"],
            "cv_mae": round(
                float(np.mean(scores)), 4
            ),
            "cv_folds_mae": scores,
            "runtime_seconds": round(
                time.perf_counter() - start, 2
            ),
        })

    # Select tuning configuration using training-only CV.
    best_cv = min(
        experiments,
        key=lambda entry: entry["cv_mae"],
    )

    selected_config = next(
        config
        for config in configurations
        if config["name"] == best_cv["name"]
    )

    # Compare with our original Day 65 Random Forest.
    candidates = {
        "day65_reference": build_candidates(
            forest_trees=reference_trees
        )["random_forest"],

        "day66_cv_selected": model_pipeline(
            selected_config
        ),
    }

    comparison = []
    predictions = {}

    for name, model in candidates.items():

        start = time.perf_counter()

        model.fit(
            train[FEATURES],
            train["cnt"],
        )

        predicted = model.predict(
            valid[FEATURES]
        )

        predictions[name] = predicted

        comparison.append({
            "name": name,
            **metrics(valid["cnt"], predicted),
            "fit_and_predict_seconds": round(
                time.perf_counter() - start, 2
            ),
        })

    baseline = build_baseline()

    baseline.fit(
        train[FEATURES],
        train["cnt"],
    )

    baseline_metrics = metrics(
        valid["cnt"],
        baseline.predict(valid[FEATURES]),
    )

    # Choose once using the validation comparison.
    chosen = min(
        comparison,
        key=lambda entry: entry["mae"],
    )

    worst = valid[
        [
            "dteday",
            "hr",
            "holiday",
            "workingday",
            "cnt",
        ]
    ].copy()

    worst["predicted"] = predictions[
        chosen["name"]
    ]

    worst["error"] = abs(
        worst["cnt"] - worst["predicted"]
    )

    worst = worst.nlargest(5, "error")

    worst["dteday"] = worst[
        "dteday"
    ].dt.strftime("%Y-%m-%d")

    # Save only to ignored/generated local directory.
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        candidates[chosen["name"]],
        out_dir / "selected_pipeline.joblib",
    )

    report = {
        "training_rows": len(train),
        "validation_rows": len(valid),
        "test_rows_untouched": len(test),
        "baseline": baseline_metrics,
        "cv_results": experiments,
        "cv_best": best_cv["name"],
        "validation_comparison": comparison,
        "selected_model": chosen["name"],
        "validation_mae": chosen["mae"],
        "validation_rmse": chosen["rmse"],
        "largest_errors": worst.round(
            2
        ).to_dict("records"),
        "limitations": [
            "Historical 2011-2012 data only",
            "CV uses training dates only",
            "Validation compares prior and tuned models",
            "Held-out test not evaluated",
        ],
    }

    (
        out_dir / "experiment_report.json"
    ).write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    return report


def main():

    root = Path(__file__).resolve().parents[2]

    report = run_day66(
        root / "data/processed/hour_clean.csv",
        root / "reports/generated/day66",
    )

    print("\nDAY 66 - CONTROLLED ML EXPERIMENTS")

    print("\nTraining-only cross-validation:")

    for item in report["cv_results"]:
        print(
            f'{item["name"]:17} '
            f'CV MAE={item["cv_mae"]:8.2f} '
            f'Runtime={item["runtime_seconds"]}s'
        )

    print("\nValidation comparison:")

    print(
        "Mean baseline MAE:",
        report["baseline"]["mae"],
    )

    for item in report["validation_comparison"]:
        print(
            f'{item["name"]:17} '
            f'MAE={item["mae"]:8.2f} '
            f'RMSE={item["rmse"]:8.2f}'
        )

    print(
        "\nSelected for next stage:",
        report["selected_model"],
    )

    print(
        "Held-out test rows untouched:",
        report["test_rows_untouched"],
    )

    print("\nFive largest validation errors:")

    for item in report["largest_errors"]:
        print(item)

    print("\nDay 66 results saved locally.")


if __name__ == "__main__":
    main()
