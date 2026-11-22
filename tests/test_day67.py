
"""Day 67: automated tests for clustering and PCA."""

import json

import numpy as np
import pandas as pd
import pytest

from src.predictiq.day67_unsupervised import (
    CLUSTER_FEATURES,
    run_day67,
)


def sample_data(rows=60):
    i = np.arange(rows)
    hour = i % 24

    return pd.DataFrame({
        "dteday": pd.date_range(
            "2011-01-01", periods=rows, freq="D"
        ).strftime("%Y-%m-%d"),
        "cnt": 40 + i * 3,
        "season": [1] * rows,
        "yr": [0] * rows,
        "mnth": [1] * rows,
        "hr": hour,
        "holiday": [0] * rows,
        "weekday": i % 7,
        "workingday": (i % 7 < 5).astype(int),
        "weather_code": [1] * rows,
        "temp": 0.2 + (i % 12) * 0.05,
        "atemp": [0.4] * rows,
        "hum": 0.3 + (i % 10) * 0.05,
        "windspeed": 0.1 + (i % 5) * 0.02,
        "is_weekend": (i % 7 >= 5).astype(int),
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
    })


def test_no_target_leakage():
    forbidden = {"cnt", "casual", "registered", "instant"}

    assert forbidden.isdisjoint(CLUSTER_FEATURES)


def test_clustering_and_pca_artifacts(tmp_path):
    source = tmp_path / "data.csv"
    output = tmp_path / "results"

    sample_data().to_csv(source, index=False)
    original = source.read_bytes()

    report = run_day67(source, output)

    assert report["n_clusters"] == 3
    assert len(report["cluster_summary"]) == 3

    total_cluster_rows = sum(
        group["rows"]
        for group in report["cluster_summary"]
    )

    assert total_cluster_rows == report["training_rows"]

    assert (
        report["training_rows"]
        + report["validation_rows_not_used"]
        + report["test_rows_not_used"]
        == 60
    )

    variance = report["pca_explained_variance_ratio"]

    assert len(variance) == 2
    assert all(0 <= value <= 1 for value in variance)
    assert sum(variance) <= 1.0001

    assert (output / "pca_clusters.png").stat().st_size > 0
    assert (output / "report.json").is_file()

    saved = json.loads(
        (output / "report.json").read_text(encoding="utf-8")
    )

    assert saved["n_clusters"] == 3

    # The input dataset must remain unchanged.
    assert source.read_bytes() == original


def test_missing_dataset_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        run_day67(
            tmp_path / "missing.csv",
            tmp_path / "results",
        )


def test_invalid_cluster_count_rejected(tmp_path):
    source = tmp_path / "data.csv"
    sample_data().to_csv(source, index=False)

    for invalid in (0, 1, 2.5):
        with pytest.raises(ValueError):
            run_day67(
                source,
                tmp_path / "results",
                n_clusters=invalid,
            )


def test_nonfinite_features_rejected(tmp_path):
    source = tmp_path / "data.csv"

    data = sample_data()
    data.loc[0, "temp"] = np.nan

    data.to_csv(source, index=False)

    with pytest.raises(ValueError):
        run_day67(source, tmp_path / "results")


def test_excessive_clusters_rejected(tmp_path):
    source = tmp_path / "data.csv"

    sample_data().to_csv(source, index=False)

    with pytest.raises(ValueError):
        run_day67(
            source,
            tmp_path / "results",
            n_clusters=100,
        )
