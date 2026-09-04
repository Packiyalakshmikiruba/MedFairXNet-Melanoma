from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data" / "processed" / "binary_roi_splits"

FILES = {
    "train": DATA_DIR / "train_binary_roi.csv",
    "validation": DATA_DIR / "validation_binary_roi.csv",
    "test": DATA_DIR / "test_binary_roi.csv",
}

datasets = {}

print("=" * 70)
print("LESION-LEVEL LEAKAGE CHECK")
print("=" * 70)

for split, path in FILES.items():

    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(path)

    datasets[split] = df

    print(f"\n{split.upper()}")
    print("-" * 70)
    print("Rows:", len(df))
    print("Unique lesion IDs:", df["lesion_id"].nunique())


# ---------------------------------------------------------
# CHECK OVERLAP
# ---------------------------------------------------------

train_ids = set(datasets["train"]["lesion_id"])
val_ids = set(datasets["validation"]["lesion_id"])
test_ids = set(datasets["test"]["lesion_id"])

train_val = train_ids & val_ids
train_test = train_ids & test_ids
val_test = val_ids & test_ids

print("\n" + "=" * 70)
print("OVERLAP RESULTS")
print("=" * 70)

print("\nTrain ∩ Validation:", len(train_val))
print("Train ∩ Test:", len(train_test))
print("Validation ∩ Test:", len(val_test))


# ---------------------------------------------------------
# DETAILS
# ---------------------------------------------------------

if train_val:
    print("\nWARNING: Train/Validation leakage detected!")
    print("Example IDs:", list(train_val)[:10])

if train_test:
    print("\nWARNING: Train/Test leakage detected!")
    print("Example IDs:", list(train_test)[:10])

if val_test:
    print("\nWARNING: Validation/Test leakage detected!")
    print("Example IDs:", list(val_test)[:10])


# ---------------------------------------------------------
# FINAL STATUS
# ---------------------------------------------------------

if not train_val and not train_test and not val_test:

    print("\n" + "=" * 70)
    print("PASS: NO LESION-LEVEL LEAKAGE DETECTED")
    print("=" * 70)

else:

    print("\n" + "=" * 70)
    print("FAIL: LESION-LEVEL LEAKAGE DETECTED")
    print("=" * 70)