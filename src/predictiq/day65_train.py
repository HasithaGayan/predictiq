
"""Day 65: supervised regression and model comparison."""

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline

from src.predictiq.day64_eda import (
    FEATURES,
    build_baseline,
    split_chronologically,
)


def build_candidates(forest_trees=100):
    if not isinstance(forest_trees, int) or forest_trees < 1:
        raise ValueError("forest_trees must be a positive integer")

    def pipeline(estimator):
        return Pipeline([
            (
                "preprocess",
                build_baseline().named_steps["preprocess"],
            ),
            ("regressor", estimator),
        ])

    return {
        "mean_baseline": build_baseline(),

        "ridge": pipeline(
            Ridge(alpha=10.0)
        ),

        "random_forest": pipeline(
            RandomForestRegressor(
                n_estimators=forest_trees,
                max_depth=16,
                min_samples_leaf=3,
                random_state=42,
                n_jobs=2,
            )
        ),
    }


def train_and_compare(
    cleaned_csv: Path,
    output_dir: Path,
    forest_trees: int = 100,
) -> dict:

    cleaned_csv = Path(cleaned_csv)
    output_dir = Path(output_dir)

    if not cleaned_csv.is_file():
        raise FileNotFoundError(cleaned_csv)

    candidates = build_candidates(forest_trees)

    data = pd.read_csv(cleaned_csv)
    train, validation, test = split_chronologically(data)

    for name, frame in (
        ("training", train),
        ("validation", validation),
    ):
        targets = frame["cnt"].to_numpy(dtype=float)

        if not np.isfinite(targets).all():
            raise ValueError(f"{name} targets must be finite")

        if (targets < 0).any():
            raise ValueError(f"{name} contains negative counts")

    x_train = train[FEATURES]
    y_train = train["cnt"]

    x_val = validation[FEATURES]
    y_val = validation["cnt"]

    results = []
    fitted_models = {}
    predictions_by_name = {}

    for name, model in candidates.items():
        # Fit on TRAINING records only.
        model.fit(x_train, y_train)

        # Compare every model on the SAME validation period.
        predictions = model.predict(x_val)

        mae = float(mean_absolute_error(y_val, predictions))
        rmse = float(
            np.sqrt(mean_squared_error(y_val, predictions))
        )

        results.append({
            "model": name,
            "validation_mae": round(mae, 4),
            "validation_rmse": round(rmse, 4),
        })

        fitted_models[name] = model
        predictions_by_name[name] = predictions

    # Lower MAE is the selection criterion.
    best = min(results, key=lambda item: item["validation_mae"])
    best_name = best["model"]

    best_predictions = predictions_by_name[best_name]

    # Inspect the largest validation mistakes.
    errors = pd.DataFrame({
        "date": validation["dteday"].dt.strftime("%Y-%m-%d").to_numpy(),
        "hour": validation["hr"].to_numpy(),
        "actual": y_val.to_numpy(),
        "predicted": best_predictions,
        "absolute_error": np.abs(
            y_val.to_numpy() - best_predictions
        ),
    })

    worst = errors.nlargest(
        5, "absolute_error"
    ).round(2)

    output_dir.mkdir(parents=True, exist_ok=True)

    artifact = output_dir / "best_pipeline.joblib"
    joblib.dump(fitted_models[best_name], artifact)

    report = {
        "project": "PredictIQ",
        "training_rows": len(train),
        "validation_rows": len(validation),
        "test_rows_held_out": len(test),
        "training_end": str(train["dteday"].max().date()),
        "validation_end": str(validation["dteday"].max().date()),
        "features": FEATURES,
        "target": "cnt",
        "selection_metric": "validation_mae",
        "models": results,
        "best_model": best_name,
        "best_validation_mae": best["validation_mae"],
        "best_validation_rmse": best["validation_rmse"],
        "worst_validation_errors": worst.to_dict(
            orient="records"
        ),
        "artifact": artifact.name,
        "cleaned_data_sha256": hashlib.sha256(
            cleaned_csv.read_bytes()
        ).hexdigest(),
        "versions": {
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "limitations": [
            "Historical 2011-2012 data only",
            "No modern forecasting accuracy established",
            "Test period remains held out",
            "Candidate models require validation before deployment",
        ],
    }

    (output_dir / "training_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    return report


def main():
    root = Path(__file__).resolve().parents[2]

    report = train_and_compare(
        root / "data" / "processed" / "hour_clean.csv",
        root / "reports" / "generated" / "day65",
    )

    print("\nPREDICTIQ — DAY 65 MODEL COMPARISON")
    print("=" * 48)

    for result in report["models"]:
        print(
            f'{result["model"]:15} '
            f'MAE: {result["validation_mae"]:8.2f} '
            f'RMSE: {result["validation_rmse"]:8.2f}'
        )

    print("\nBest current model:", report["best_model"])
    print("Validation MAE:", report["best_validation_mae"])
    print("Validation RMSE:", report["best_validation_rmse"])
    print("Held-out test rows:", report["test_rows_held_out"])

    print("\nLargest validation errors:")
    for error in report["worst_validation_errors"]:
        print(error)

    print("\nPipeline and metadata saved locally.")


if __name__ == "__main__":
    main()
