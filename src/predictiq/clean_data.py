from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd


REQUIRED = {
    "instant", "dteday", "season", "yr",
    "mnth", "hr", "holiday", "weekday",
    "workingday", "weathersit", "temp",
    "atemp", "hum", "windspeed", "casual",
    "registered", "cnt",
}

CATEGORIES = {
    "season": {1, 2, 3, 4},
    "yr": {0, 1},
    "mnth": set(range(1, 13)),
    "hr": set(range(24)),
    "holiday": {0, 1},
    "weekday": set(range(7)),
    "workingday": {0, 1},
    "weathersit": {1, 2, 3, 4},
}

NORMALIZED = ["temp", "atemp", "hum", "windspeed"]

OUTPUT_COLUMNS = [
    "dteday", "season", "yr", "mnth", "hr",
    "holiday", "weekday", "workingday",
    "weathersit", "temp", "atemp", "hum",
    "windspeed", "is_weekend",
    "hour_sin", "hour_cos", "cnt",
]


def clean_dataset(
    raw_path: Path,
    output_path: Path,
    report_path: Path,
) -> dict:

    raw_path = Path(raw_path)
    output_path = Path(output_path)
    report_path = Path(report_path)

    if not raw_path.is_file():
        raise FileNotFoundError(raw_path)

    if raw_path.resolve() == output_path.resolve():
        raise ValueError("Never overwrite raw data")

    df = pd.read_csv(raw_path)

    missing_columns = REQUIRED - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    original_rows = len(df)
    missing_cells = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())

    # Explicit decision: reject incomplete records.
    # Never silently delete or fill missing data.
    if missing_cells > 0:
        raise ValueError(
            f"Dataset contains {missing_cells} missing cells"
        )

    if original_rows == 0:
        raise ValueError("Dataset is empty")

    # Validate date formatting.
    df["dteday"] = pd.to_datetime(
        df["dteday"],
        format="%Y-%m-%d",
        errors="raise",
    ).dt.strftime("%Y-%m-%d")

    # Every date/hour should identify one observation.
    repeated_hours = int(
        df.duplicated(subset=["dteday", "hr"]).sum()
    )

    if repeated_hours:
        raise ValueError(
            f"Duplicate date/hour observations: {repeated_hours}"
        )

    # Validate categorical values.
    for column, allowed in CATEGORIES.items():
        invalid = ~df[column].isin(allowed)

        if invalid.any():
            raise ValueError(
                f"Invalid category in {column}"
            )

    # Validate normalized weather measurements.
    for column in NORMALIZED:
        invalid = ~df[column].between(0, 1)

        if invalid.any():
            raise ValueError(
                f"{column} must be between 0 and 1"
            )

    # Rental counts cannot be negative or fractional.
    counts = df[["casual", "registered", "cnt"]]

    if (counts < 0).any().any():
        raise ValueError("Negative rental count")

    if (counts % 1 != 0).any().any():
        raise ValueError("Fractional rental count")

    # The dataset target must be internally consistent.
    if not (
        df["casual"] + df["registered"] == df["cnt"]
    ).all():
        raise ValueError("Target relationship failed")

    # Create useful time-related features.
    df["is_weekend"] = (
        df["weekday"].isin([0, 6]).astype(int)
    )

    df["hour_sin"] = np.sin(
        2 * np.pi * df["hr"] / 24
    )

    df["hour_cos"] = np.cos(
        2 * np.pi * df["hr"] / 24
    )

    # Select features available without knowing the target.
    # Exclude casual, registered and the row identifier.
    cleaned = df[OUTPUT_COLUMNS].copy()

    # Demonstrate explicit column renaming.
    cleaned = cleaned.rename(
        columns={"weathersit": "weather_code"}
    )

    cleaned = cleaned.sort_values(
        ["dteday", "hr"]
    ).reset_index(drop=True)

    # Record cleaning decisions and source provenance.
    report = {
        "source": raw_path.name,
        "source_sha256": hashlib.sha256(
            raw_path.read_bytes()
        ).hexdigest(),
        "raw_rows": original_rows,
        "raw_columns": len(df.columns) - 3,
        "missing_cells": missing_cells,
        "duplicate_rows": duplicate_rows,
        "duplicate_hour_keys": repeated_hours,
        "cleaned_rows": len(cleaned),
        "target": "cnt",
        "excluded_columns": [
            "instant",
            "casual",
            "registered",
        ],
        "created_features": [
            "is_weekend",
            "hour_sin",
            "hour_cos",
        ],
        "output_columns": list(cleaned.columns),
        "cleaning_decisions": [
            "Reject incomplete records",
            "Reject invalid dates and categories",
            "Reject invalid or inconsistent counts",
            "Reject duplicate date/hour observations",
            "Retain valid observations",
            "Sort chronologically",
            "Exclude identifier and target leakage columns",
        ],
    }

    output_path.parent.mkdir(
        parents=True, exist_ok=True
    )

    report_path.parent.mkdir(
        parents=True, exist_ok=True
    )

    cleaned.to_csv(
        output_path,
        index=False,
        float_format="%.8f",
    )

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    return report


def main():
    root = Path(__file__).resolve().parents[2]

    raw = root / "data" / "raw" / "hour.csv"
    output = root / "data" / "processed" / "hour_clean.csv"
    report_file = root / "reports" / "data_quality.json"

    report = clean_dataset(
        raw, output, report_file
    )

    print("PREDICTIQ CLEANING PASSED")
    print("Original rows:", report["raw_rows"])
    print("Cleaned rows:", report["cleaned_rows"])
    print("Missing cells:", report["missing_cells"])
    print("Removed leakage columns:",
          report["excluded_columns"])

    cleaned = pd.read_csv(output)
    rentals = cleaned["cnt"].to_numpy()

    # Day 63 NumPy practice
    print("NumPy array shape:", rentals.shape)
    print("First rental count:", rentals[0])
    print("First 3 counts:", rentals[:3].tolist())
    print("First 3 counts doubled:",
          (rentals[:3] * 2).tolist())
    print("Mean hourly rentals:",
          round(float(np.mean(rentals)), 2))

    # Pandas filtering without deleting records
    morning = cleaned[
        cleaned["hr"].between(7, 9)
    ]

    print("Morning observations:", len(morning))
    print("Data-quality report saved")


if __name__ == "__main__":
    main()