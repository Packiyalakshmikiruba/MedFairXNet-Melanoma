from pathlib import Path
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_CSV = BASE_DIR / "data" / "processed" / "dataset_clean.csv"
OUTPUT_DIR = BASE_DIR / "data" / "processed" / "splits"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("HAM10000 DATASET SPLITTING")
print("=" * 70)

# Load dataset
df = pd.read_csv(INPUT_CSV)

print(f"\nTotal records: {len(df)}")
print(f"Unique lesions: {df['lesion_id'].nunique()}")

# ---------------------------------------------------------
# 1. Train + Temporary split
#    80% Train
#    20% Temporary
# ---------------------------------------------------------

gss = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, temp_idx = next(
    gss.split(
        df,
        groups=df["lesion_id"]
    )
)

train_df = df.iloc[train_idx].copy()
temp_df = df.iloc[temp_idx].copy()

# ---------------------------------------------------------
# 2. Validation + Test split
#    Temporary 20% → 10% Validation + 10% Test
# ---------------------------------------------------------

gss_test = GroupShuffleSplit(
    n_splits=1,
    test_size=0.50,
    random_state=42
)

val_idx, test_idx = next(
    gss_test.split(
        temp_df,
        groups=temp_df["lesion_id"]
    )
)

val_df = temp_df.iloc[val_idx].copy()
test_df = temp_df.iloc[test_idx].copy()

# ---------------------------------------------------------
# 3. Save splits
# ---------------------------------------------------------

train_path = OUTPUT_DIR / "train.csv"
val_path = OUTPUT_DIR / "validation.csv"
test_path = OUTPUT_DIR / "test.csv"

train_df.to_csv(train_path, index=False)
val_df.to_csv(val_path, index=False)
test_df.to_csv(test_path, index=False)

# ---------------------------------------------------------
# 4. Display results
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("SPLIT COMPLETED")
print("=" * 70)

print(f"\nTrain records      : {len(train_df)}")
print(f"Validation records : {len(val_df)}")
print(f"Test records       : {len(test_df)}")

print("\nPercentages:")

total = len(df)

print(f"Train      : {len(train_df) / total * 100:.2f}%")
print(f"Validation : {len(val_df) / total * 100:.2f}%")
print(f"Test       : {len(test_df) / total * 100:.2f}%")

# ---------------------------------------------------------
# 5. Check lesion leakage
# ---------------------------------------------------------

train_lesions = set(train_df["lesion_id"])
val_lesions = set(val_df["lesion_id"])
test_lesions = set(test_df["lesion_id"])

train_val_overlap = train_lesions & val_lesions
train_test_overlap = train_lesions & test_lesions
val_test_overlap = val_lesions & test_lesions

print("\n" + "=" * 70)
print("LESION LEAKAGE CHECK")
print("=" * 70)

print(f"\nTrain ↔ Validation overlap : {len(train_val_overlap)}")
print(f"Train ↔ Test overlap       : {len(train_test_overlap)}")
print(f"Validation ↔ Test overlap  : {len(val_test_overlap)}")

# ---------------------------------------------------------
# 6. Diagnosis distribution
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("TRAIN DIAGNOSIS DISTRIBUTION")
print("=" * 70)

print(train_df["dx"].value_counts())

print("\n" + "=" * 70)
print("VALIDATION DIAGNOSIS DISTRIBUTION")
print("=" * 70)

print(val_df["dx"].value_counts())

print("\n" + "=" * 70)
print("TEST DIAGNOSIS DISTRIBUTION")
print("=" * 70)

print(test_df["dx"].value_counts())

print("\n" + "=" * 70)
print("OUTPUT FILES")
print("=" * 70)

print(f"\nTrain      : {train_path}")
print(f"Validation : {val_path}")
print(f"Test       : {test_path}")

print("\nDataset splitting completed successfully.")
print("=" * 70)