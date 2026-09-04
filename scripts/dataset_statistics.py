from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data" / "processed" / "binary_roi_splits"

FILES = {
    "Train": "train_binary_roi.csv",
    "Validation": "validation_binary_roi.csv",
    "Test": "test_binary_roi.csv",
}

print("=" * 75)
print("HAM10000 BINARY ROI DATASET STATISTICS")
print("=" * 75)

all_data = []

for split, filename in FILES.items():

    path = DATA_DIR / filename
    df = pd.read_csv(path)

    df["split"] = split
    all_data.append(df)

    print("\n" + "=" * 75)
    print(split.upper())
    print("=" * 75)

    print("Images:", len(df))
    print("Unique lesions:", df["lesion_id"].nunique())

    print("\nBinary distribution:")
    print(df["binary_diagnosis"].value_counts())

    print("\nBinary percentage:")
    print(
        (df["binary_diagnosis"].value_counts(normalize=True) * 100)
        .round(2)
    )

    print("\nSex distribution:")
    print(df["sex"].value_counts(dropna=False))

    print("\nAge statistics:")
    print(df["age"].describe())

    print("\nTop localizations:")
    print(df["localization"].value_counts().head(10))


# ---------------------------------------------------------
# COMBINED DATASET
# ---------------------------------------------------------

combined = pd.concat(all_data, ignore_index=True)

print("\n" + "=" * 75)
print("COMBINED DATASET")
print("=" * 75)

print("Total images:", len(combined))
print("Unique lesions:", combined["lesion_id"].nunique())

print("\nSplit distribution:")
print(combined["split"].value_counts())

print("\nBinary distribution:")
print(combined["binary_diagnosis"].value_counts())

print("\nBinary percentage:")
print(
    (combined["binary_diagnosis"].value_counts(normalize=True) * 100)
    .round(2)
)

print("\nMissing values:")
print(combined.isna().sum())

print("\nImage path existence:")

missing = 0

for path in combined["image_path"]:

    full_path = BASE_DIR / path

    if not full_path.exists():
        missing += 1

print("Missing images:", missing)

print("\n" + "=" * 75)
print("DATASET STATISTICS COMPLETED")
print("=" * 75)