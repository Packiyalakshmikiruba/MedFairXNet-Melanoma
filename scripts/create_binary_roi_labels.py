from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_DIR = BASE_DIR / "data" / "processed" / "roi_splits"
OUTPUT_DIR = BASE_DIR / "data" / "processed" / "binary_roi_splits"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FILES = [
    "train_roi.csv",
    "validation_roi.csv",
    "test_roi.csv",
]

# Melanoma = 1
# All other diagnoses = 0
BINARY_MAPPING = {
    "mel": 1,
    "akiec": 0,
    "bcc": 0,
    "bkl": 0,
    "df": 0,
    "nv": 0,
    "vasc": 0,
}


print("=" * 70)
print("CREATING BINARY MELANOMA ROI DATASETS")
print("=" * 70)

for filename in FILES:

    input_path = INPUT_DIR / filename
    output_path = OUTPUT_DIR / filename.replace(
        "_roi.csv",
        "_binary_roi.csv"
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}"
        )

    df = pd.read_csv(input_path)

    print("\n" + "=" * 70)
    print(filename)
    print("=" * 70)

    print("Original rows:", len(df))

    # Check diagnosis values
    unknown = set(df["dx"].dropna().unique()) - set(BINARY_MAPPING.keys())

    if unknown:
        raise ValueError(
            f"Unknown diagnosis labels found: {unknown}"
        )

    # Create binary target
    df["binary_label"] = df["dx"].map(BINARY_MAPPING)

    if df["binary_label"].isna().any():
        raise ValueError(
            "Some rows have missing binary labels."
        )

    df["binary_label"] = df["binary_label"].astype(int)

    # Human-readable label
    df["binary_diagnosis"] = df["binary_label"].map({
        0: "Non-melanoma",
        1: "Melanoma"
    })

    # Statistics
    print("\nBinary distribution:")

    print(
        df["binary_diagnosis"]
        .value_counts()
    )

    print("\nBinary counts:")

    print(
        df["binary_label"]
        .value_counts()
        .sort_index()
    )

    # Save
    df.to_csv(
        output_path,
        index=False
    )

    print("\nSaved:")
    print(output_path)

print("\n" + "=" * 70)
print("BINARY DATASET CREATION COMPLETED")
print("=" * 70)