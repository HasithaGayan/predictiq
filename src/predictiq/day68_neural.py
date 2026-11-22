
"""Day 68: small PyTorch neural-network learning experiment."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

from src.predictiq.day64_eda import split_chronologically


INPUT_FEATURES = [
    "hr",
    "workingday",
    "temp",
    "hum",
    "windspeed",
    "hour_sin",
    "hour_cos",
]


class TinyDemandNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.network = torch.nn.Sequential(
            torch.nn.Linear(len(INPUT_FEATURES), 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 1),
        )

    def forward(self, inputs):
        return self.network(inputs)


def train_one(x_train, y_train, x_val, y_val, lr, epochs=75):
    if lr <= 0 or epochs < 1:
        raise ValueError("Learning rate and epochs must be positive")

    # Reset initial weights for a fair comparison.
    torch.manual_seed(42)

    model = TinyDemandNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    training_losses = []
    validation_losses = []

    for epoch in range(epochs):
        model.train()

        # 1. Clear previous gradients.
        optimizer.zero_grad()

        # 2. Forward pass.
        prediction = model(x_train)

        # 3. Calculate training loss.
        loss = criterion(prediction, y_train)

        # 4. Backpropagation.
        loss.backward()

        # 5. Update neural-network weights.
        optimizer.step()

        model.eval()

        with torch.no_grad():
            val_prediction = model(x_val)
            val_loss = criterion(val_prediction, y_val)

        training_losses.append(float(loss.item()))
        validation_losses.append(float(val_loss.item()))

    return model, {
        "training_loss": training_losses,
        "validation_loss": validation_losses,
    }


def run_day68(clean_csv, output_dir, epochs=75):
    clean_csv = Path(clean_csv)
    output_dir = Path(output_dir)

    if not clean_csv.is_file():
        raise FileNotFoundError(clean_csv)

    if not isinstance(epochs, int) or epochs < 1:
        raise ValueError("epochs must be a positive integer")

    data = pd.read_csv(clean_csv)

    train, validation, test = split_chronologically(data)

    x_train_raw = train[INPUT_FEATURES].to_numpy(dtype=float)
    x_val_raw = validation[INPUT_FEATURES].to_numpy(dtype=float)

    y_train_raw = train["cnt"].to_numpy(dtype=float).reshape(-1, 1)
    y_val_raw = validation["cnt"].to_numpy(dtype=float).reshape(-1, 1)

    for values in (x_train_raw, x_val_raw, y_train_raw, y_val_raw):
        if not np.isfinite(values).all():
            raise ValueError("Training/validation values must be finite")

    if (y_train_raw < 0).any() or (y_val_raw < 0).any():
        raise ValueError("Rental counts cannot be negative")

    # Fit scaling parameters using TRAINING data only.
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()

    x_train_scaled = feature_scaler.fit_transform(x_train_raw)
    x_val_scaled = feature_scaler.transform(x_val_raw)

    y_train_scaled = target_scaler.fit_transform(y_train_raw)
    y_val_scaled = target_scaler.transform(y_val_raw)

    def tensor(values):
        return torch.tensor(values, dtype=torch.float32)

    x_train = tensor(x_train_scaled)
    x_val = tensor(x_val_scaled)
    y_train = tensor(y_train_scaled)
    y_val = tensor(y_val_scaled)

    results = []
    histories = {}
    fitted_models = {}

    for lr in (0.01, 0.001):
        model, history = train_one(
            x_train, y_train, x_val, y_val,
            lr=lr, epochs=epochs,
        )

        model.eval()

        with torch.no_grad():
            predicted_scaled = model(x_val).numpy()

        predicted = target_scaler.inverse_transform(
            predicted_scaled
        ).reshape(-1)

        actual = y_val_raw.reshape(-1)

        mae = float(mean_absolute_error(actual, predicted))
        rmse = float(
            np.sqrt(mean_squared_error(actual, predicted))
        )

        name = f"lr_{lr}"

        histories[name] = history
        fitted_models[name] = model

        results.append({
            "name": name,
            "learning_rate": lr,
            "validation_mae": round(mae, 4),
            "validation_rmse": round(rmse, 4),
            "initial_training_loss": round(
                history["training_loss"][0], 6
            ),
            "final_training_loss": round(
                history["training_loss"][-1], 6
            ),
            "final_validation_loss": round(
                history["validation_loss"][-1], 6
            ),
        })

    # Select only within this neural-network experiment.
    chosen = min(results, key=lambda r: r["validation_mae"])
    best_model = fitted_models[chosen["name"]]

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save weights and preprocessing metadata.
    torch.save(
        {
            "model_state_dict": best_model.state_dict(),
            "features": INPUT_FEATURES,
            "feature_mean": feature_scaler.mean_.tolist(),
            "feature_scale": feature_scaler.scale_.tolist(),
            "target_mean": target_scaler.mean_.tolist(),
            "target_scale": target_scaler.scale_.tolist(),
            "learning_rate": chosen["learning_rate"],
        },
        output_dir / "tiny_network.pt",
    )

    fig, ax = plt.subplots(figsize=(9, 6))

    for name, history in histories.items():
        ax.plot(
            history["training_loss"],
            label=f"{name} training",
        )
        ax.plot(
            history["validation_loss"],
            linestyle="--",
            label=f"{name} validation",
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE on scaled rental targets")
    ax.set_title("Day 68 — Neural Network Training")
    ax.legend()
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(output_dir / "learning_curves.png", dpi=150)
    plt.close(fig)

    report = {
        "project": "PredictIQ",
        "experiment": "Day 68 PyTorch fundamentals",
        "pytorch_version": torch.__version__,
        "features": INPUT_FEATURES,
        "training_rows": len(train),
        "validation_rows": len(validation),
        "test_rows_not_used": len(test),
        "epochs": epochs,
        "experiments": results,
        "selected_learning_rate": chosen["learning_rate"],
        "limitations": [
            "Learning experiment, not production model",
            "Historical bike data from 2011-2012",
            "Held-out test period not evaluated",
            "Validation results are not final generalization estimates",
        ],
    }

    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    return report


def main():
    root = Path(__file__).resolve().parents[2]

    report = run_day68(
        root / "data/processed/hour_clean.csv",
        root / "reports/generated/day68",
    )

    print("\nPREDICTIQ DAY 68 — PYTORCH NEURAL NETWORK")
    print("PyTorch:", report["pytorch_version"])

    for item in report["experiments"]:
        print(
            f'{item["name"]}: '
            f'MAE={item["validation_mae"]:.2f}, '
            f'RMSE={item["validation_rmse"]:.2f}, '
            f'Loss={item["initial_training_loss"]:.4f}'
            f' -> {item["final_training_loss"]:.4f}'
        )

    print(
        "Selected learning rate:",
        report["selected_learning_rate"],
    )

    print(
        "Held-out test rows unused:",
        report["test_rows_not_used"],
    )

    print("DAY 68 INITIAL EXPERIMENT COMPLETE")


if __name__ == "__main__":
    main()
