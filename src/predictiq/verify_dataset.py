from pathlib import Path
import sys

import numpy as np
import pandas as pd


def main():
    path = Path(sys.argv[1])

    if not path.is_file():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    required = {
        "dteday",
        "hr",
        "temp",
        "hum",
        "windspeed",
        "casual",
        "registered",
        "cnt",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if len(df) <= 17000:
        raise ValueError("Unexpected dataset row count")

    if not (
        df["casual"] + df["registered"] == df["cnt"]
    ).all():
        raise ValueError("Target relationship failed")

    print("PREDICTIQ DATASET VERIFIED")
    print("Rows:", len(df))
    print("Columns:", len(df.columns))
    print("Missing cells:", int(df.isna().sum().sum()))
    print("Duplicate rows:", int(df.duplicated().sum()))
    print("Target: cnt")
    print("Leakage columns: casual, registered")
    print("NumPy version:", np.__version__)
    print("Pandas version:", pd.__version__)


if __name__ == "__main__":
    main()
