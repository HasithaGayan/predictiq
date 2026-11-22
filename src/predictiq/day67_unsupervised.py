
"""Day 67: clustering and PCA on historical training data."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.predictiq.day64_eda import split_chronologically


CLUSTER_FEATURES = [
    "temp",
    "hum",
    "windspeed",
    "hour_sin",
    "hour_cos",
    "workingday",
]


def run_day67(clean_csv, output_dir, n_clusters=3):
    clean_csv = Path(clean_csv)
    output_dir = Path(output_dir)

    if not clean_csv.is_file():
        raise FileNotFoundError(clean_csv)

    if not isinstance(n_clusters, int) or n_clusters < 2:
        raise ValueError("n_clusters must be an integer >= 2")

    data = pd.read_csv(clean_csv)

    # Preserve our original chronological split.
    train, validation, test = split_chronologically(data)

    if len(train) <= n_clusters:
        raise ValueError("Not enough training observations")

    # No cnt, casual or registered in clustering inputs.
    x = train[CLUSTER_FEATURES].astype(float)

    if not np.isfinite(x.to_numpy()).all():
        raise ValueError("Features must be finite")

    # Scale feature values before distance-based clustering.
    scaler = StandardScaler()
    scaled = scaler.fit_transform(x)

    # Learn three groups without target labels.
    model = KMeans(
        n_clusters=n_clusters,
        n_init=10,
        random_state=42,
    )

    labels = model.fit_predict(scaled)

    # Reduce to two dimensions for visualization.
    pca = PCA(n_components=2)
    coordinates = pca.fit_transform(scaled)

    # Inspect groups only AFTER fitting the clusters.
    inspection = train[
        ["hr", "temp", "hum", "cnt"]
    ].copy()

    inspection["cluster"] = labels

    groups = []

    for cluster_id in range(n_clusters):
        subset = inspection[
            inspection["cluster"] == cluster_id
        ]

        if subset.empty:
            raise ValueError("An empty cluster was produced")

        groups.append({
            "cluster": cluster_id,
            "rows": len(subset),
            "mean_hour": round(
                float(subset["hr"].mean()), 2
            ),
            "mean_temperature": round(
                float(subset["temp"].mean()), 4
            ),
            "mean_humidity": round(
                float(subset["hum"].mean()), 4
            ),
            # For inspection only, NOT used in clustering.
            "observed_mean_rentals": round(
                float(subset["cnt"].mean()), 2
            ),
        })

    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        c=labels,
        s=6,
        alpha=0.45,
    )

    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_title("PredictIQ — Historical Condition Clusters")

    fig.tight_layout()

    fig.savefig(
        output_dir / "pca_clusters.png",
        dpi=160,
    )

    plt.close(fig)

    report = {
        "project": "PredictIQ",
        "experiment": "Day 67 unsupervised learning",
        "training_rows": len(train),
        "validation_rows_not_used": len(validation),
        "test_rows_not_used": len(test),
        "clustering_features": CLUSTER_FEATURES,
        "n_clusters": n_clusters,
        "cluster_summary": groups,
        "pca_explained_variance_ratio": [
            round(float(value), 4)
            for value in pca.explained_variance_ratio_
        ],
        "limitations": [
            "Clusters are not ground-truth labels",
            "Cluster meanings require interpretation",
            "PCA is a visualization, not proof of usefulness",
            "Historical data is from 2011-2012",
            "Rental counts were not used to fit clusters",
        ],
    }

    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    return report


def main():
    root = Path(__file__).resolve().parents[2]

    report = run_day67(
        root / "data/processed/hour_clean.csv",
        root / "reports/generated/day67",
    )

    print("\nPREDICTIQ DAY 67 — UNSUPERVISED LEARNING")

    for group in report["cluster_summary"]:
        print(group)

    print(
        "\nPCA explained variance:",
        report["pca_explained_variance_ratio"],
    )

    print(
        "\nHeld-out test rows unused:",
        report["test_rows_not_used"],
    )

    print("\nDay 67 initial experiment completed.")


if __name__ == "__main__":
    main()
