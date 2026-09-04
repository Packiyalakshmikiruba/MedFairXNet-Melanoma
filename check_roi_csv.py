import pandas as pd
from pathlib import Path

base = Path("data/processed/roi_splits")

files = [
    "train_roi.csv",
    "validation_roi.csv",
    "test_roi.csv"
]

for filename in files:

    path = base / filename

    print("\n" + "=" * 70)
    print(filename)
    print("=" * 70)

    df = pd.read_csv(path)

    print("\nRows:", len(df))

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nDiagnosis distribution:")
    if "dx" in df.columns:
        print(df["dx"].value_counts().sort_index())
    else:
        print("WARNING: 'dx' column not found")

    print("\nFirst 5 rows:")
    print(df.head().to_string(index=False))