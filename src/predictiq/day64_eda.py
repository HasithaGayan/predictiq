
"""Day 64: training-only EDA and leakage-safe ML baseline."""

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


CATEGORICAL = [
    "season", "yr", "mnth", "hr",
    "holiday", "weekday", "workingday", "weather_code"
]

NUMERICAL = [
    "temp", "atemp", "hum", "windspeed",
    "is_weekend", "hour_sin", "hour_cos"
]

FEATURES = CATEGORICAL + NUMERICAL
FORBIDDEN = {"casual", "registered", "instant"}


def split_chronologically(data: pd.DataFrame):
    missing = set(FEATURES + ["dteday", "cnt"]) - set(data.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if FORBIDDEN & set(data.columns):
        raise ValueError("Target-leakage or identifier columns are present")

    df = data.copy()
    df["dteday"] = pd.to_datetime(df["dteday"], errors="raise")

    if df["dteday"].isna().any():
        raise ValueError("Missing dates")

    if df.duplicated(["dteday", "hr"]).any():
        raise ValueError("Duplicate date/hour observations")

    dates = sorted(df["dteday"].unique())

    if len(dates) < 10:
        raise ValueError("At least 10 different dates are required")

    # Split by whole days, not randomly shuffled rows.
    train_end = int(len(dates) * 0.70)
    valid_end = int(len(dates) * 0.85)

    train = df[
        df["dteday"].isin(dates[:train_end])
    ].sort_values(["dteday", "hr"]).copy()

    valid = df[
        df["dteday"].isin(dates[train_end:valid_end])
    ].sort_values(["dteday", "hr"]).copy()

    test = df[
        df["dteday"].isin(dates[valid_end:])
    ].sort_values(["dteday", "hr"]).copy()

    if min(len(train), len(valid), len(test)) == 0:
        raise ValueError("Empty train/validation/test split")

    assert (
        train["dteday"].max()
        < valid["dteday"].min()
        < test["dteday"].min()
    )

    return train, valid, test


def build_baseline() -> Pipeline:
    preprocess = ColumnTransformer([
        (
            "categorical",
            Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("encode", OneHotEncoder(handle_unknown="ignore")),
            ]),
            CATEGORICAL,
        ),
        (
            "numerical",
            Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]),
            NUMERICAL,
        ),
    ])

    return Pipeline([
        ("preprocess", preprocess),
        ("baseline", DummyRegressor(strategy="mean")),
    ])


def save_training_charts(
    train: pd.DataFrame,
    directory: Path
) -> list[str]:

    directory.mkdir(parents=True, exist_ok=True)
    charts = []

    # Chart 1: Distribution of rental demand.
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(train["cnt"], bins=35, color="#0F766E")
    ax.set(
        title="Training data: hourly rental counts",
        xlabel="Rentals",
        ylabel="Hours",
    )
    charts.append("01_rental_distribution.png")
    fig.tight_layout()
    fig.savefig(directory / charts[-1])
    plt.close(fig)

    # Chart 2: Average rentals by hour.
    fig, ax = plt.subplots(figsize=(7, 4))
    train.groupby("hr")["cnt"].mean().plot.bar(
        ax=ax, color="#0F766E"
    )
    ax.set(
        title="Training data: average demand by hour",
        xlabel="Hour",
        ylabel="Average rentals",
    )
    charts.append("02_mean_by_hour.png")
    fig.tight_layout()
    fig.savefig(directory / charts[-1])
    plt.close(fig)

    # Chart 3: Relationship between temperature and demand.
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(
        train["temp"],
        train["cnt"],
        alpha=0.15,
        s=10,
        color="#0F766E",
    )
    ax.set(
        title="Training data: temperature vs demand",
        xlabel="Normalized temperature",
        ylabel="Rentals",
    )
    charts.append("03_temperature_scatter.png")
    fig.tight_layout()
    fig.savefig(directory / charts[-1])
    plt.close(fig)

    # Chart 4: Working vs non-working day variation.
    fig, ax = plt.subplots(figsize=(7, 4))
    groups = [
        train.loc[train["workingday"] == value, "cnt"].to_numpy()
        for value in (0, 1)
    ]
    ax.boxplot(
        groups,
        tick_labels=["Non-working", "Working"],
    )
    ax.set(
        title="Training data: demand by working-day type",
        ylabel="Rentals",
    )
    charts.append("04_workingday_boxplot.png")
    fig.tight_layout()
    fig.savefig(directory / charts[-1])
    plt.close(fig)

    return charts


def run_day64(
    cleaned_csv: Path,
    output_directory: Path
) -> dict:

    if not cleaned_csv.is_file():
        raise FileNotFoundError(cleaned_csv)

    df = pd.read_csv(cleaned_csv)

    train, valid, test = split_chronologically(df)

    # Fit preprocessing and estimator on training data only.
    model = build_baseline()
    model.fit(train[FEATURES], train["cnt"])

    predictions = model.predict(valid[FEATURES])

    mae = float(
        mean_absolute_error(valid["cnt"], predictions)
    )

    rmse = float(
        np.sqrt(mean_squared_error(valid["cnt"], predictions))
    )

    # EDA uses training data only.
    charts = save_training_charts(
        train,
        output_directory / "figures",
    )

    hourly = train.groupby("hr")["cnt"].mean()
    peak_hour = int(hourly.idxmax())

    working_mean = float(
        train.loc[train["workingday"] == 1, "cnt"].mean()
    )

    nonworking_mean = float(
        train.loc[train["workingday"] == 0, "cnt"].mean()
    )

    temperature_corr = float(
        train["temp"].corr(train["cnt"])
    )

    observations = [
        (
            f"Highest training mean demand at {peak_hour}:00: "
            f"{hourly.max():.1f} rentals/hour."
        ),
        (
            f"Working-day mean: {working_mean:.1f}; "
            f"non-working-day mean: {nonworking_mean:.1f}."
        ),
        (
            "Training temperature-demand correlation: "
            f"{temperature_corr:.3f}."
        ),
    ]

    hypotheses = [
        "Commuting patterns change hourly peaks on working days.",
        "Temperature improves prediction beyond calendar features.",
        "Weather severity helps predict demand beyond time features.",
    ]

    report = {
        "split_strategy": "Chronological; 70/15/15 by complete dates",
        "training_rows": len(train),
        "validation_rows": len(valid),
        "test_rows_held_out": len(test),
        "training_period": [
            str(train["dteday"].min().date()),
            str(train["dteday"].max().date()),
        ],
        "validation_period": [
            str(valid["dteday"].min().date()),
            str(valid["dteday"].max().date()),
        ],
        "test_period": [
            str(test["dteday"].min().date()),
            str(test["dteday"].max().date()),
        ],
        "features": FEATURES,
        "leakage_columns_excluded": sorted(FORBIDDEN),
        "baseline": "Training-target mean (DummyRegressor)",
        "validation_mae": round(mae, 4),
        "validation_rmse": round(rmse, 4),
        "charts": charts,
        "training_only_observations": observations,
        "hypotheses_to_test": hypotheses,
        "note": (
            "Test-set labels were not used for EDA, fitting, "
            "model tuning or validation metrics."
        ),
    }

    output_directory.mkdir(parents=True, exist_ok=True)

    (output_directory / "day64_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    return report


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]

    report = run_day64(
        root / "data/processed/hour_clean.csv",
        root / "reports/generated/day64",
    )

    print("DAY 64 BASELINE + EDA COMPLETED")
    print(
        "Train / Validation / Test:",
        report["training_rows"],
        report["validation_rows"],
        report["test_rows_held_out"],
    )
    print("Validation MAE:", report["validation_mae"])
    print("Validation RMSE:", report["validation_rmse"])

    print("Three real-data observations:")
    for observation in report["training_only_observations"]:
        print("-", observation)

    print("Generated charts:", len(report["charts"]))
