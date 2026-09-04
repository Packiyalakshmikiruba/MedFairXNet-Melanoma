from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "PH2"
    / "ph2_binary_metadata.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "PH2"
    / "splits"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RANDOM_STATE = 42


print("=" * 75)
print("MELONMA - PH2 GROUP-AWARE DATASET SPLIT")
print("=" * 75)

print("\nInput:")
print(INPUT_FILE)

print("\nOutput:")
print(OUTPUT_DIR)


if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input file not found:\n{INPUT_FILE}"
    )


df = pd.read_csv(INPUT_FILE)

print("\nTotal records:", len(df))


# ---------------------------------------------------------
# BASIC VALIDATION
# ---------------------------------------------------------

required_columns = [
    "image_id",
    "clinical_class",
    "binary_label",
    "binary_diagnosis",
    "image_path",
    "lesion_mask_path",
]

for column in required_columns:

    if column not in df.columns:

        raise ValueError(
            f"Missing required column: {column}"
        )


if df["image_id"].duplicated().any():

    raise ValueError(
        "Duplicate image IDs detected."
    )


# ---------------------------------------------------------
# CLASS DISTRIBUTION
# ---------------------------------------------------------

print("\nOriginal class distribution:")

print(
    df["binary_diagnosis"]
    .value_counts()
)


# ---------------------------------------------------------
# 70 / 15 / 15 SPLIT
# ---------------------------------------------------------
#
# PH2 has one image per lesion.
# Therefore image_id acts as the lesion-level grouping key.
#
# Stratification preserves melanoma/non-melanoma ratio.
#


train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=RANDOM_STATE,
    stratify=df["binary_label"],
)


validation_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=RANDOM_STATE,
    stratify=temp_df["binary_label"],
)


# ---------------------------------------------------------
# LEAKAGE CHECK
# ---------------------------------------------------------

train_ids = set(train_df["image_id"])
validation_ids = set(validation_df["image_id"])
test_ids = set(test_df["image_id"])


train_validation_overlap = (
    train_ids & validation_ids
)

train_test_overlap = (
    train_ids & test_ids
)

validation_test_overlap = (
    validation_ids & test_ids
)


print("\n" + "=" * 75)
print("CROSS-SPLIT LEAKAGE CHECK")
print("=" * 75)

print(
    "\nTrain ∩ Validation:",
    len(train_validation_overlap)
)

print(
    "Train ∩ Test:",
    len(train_test_overlap)
)

print(
    "Validation ∩ Test:",
    len(validation_test_overlap)
)


if (
    train_validation_overlap
    or train_test_overlap
    or validation_test_overlap
):

    raise ValueError(
        "PH2 cross-split leakage detected."
    )


# ---------------------------------------------------------
# SAVE SPLITS
# ---------------------------------------------------------

train_path = (
    OUTPUT_DIR
    / "train_ph2.csv"
)

validation_path = (
    OUTPUT_DIR
    / "validation_ph2.csv"
)

test_path = (
    OUTPUT_DIR
    / "test_ph2.csv"
)


train_df.to_csv(
    train_path,
    index=False
)

validation_df.to_csv(
    validation_path,
    index=False
)

test_df.to_csv(
    test_path,
    index=False
)


# ---------------------------------------------------------
# REPORT
# ---------------------------------------------------------

print("\n" + "=" * 75)
print("PH2 SPLIT SUMMARY")
print("=" * 75)


def print_split(name, split_df):

    print(f"\n=== {name} ===")

    print(
        "Records:",
        len(split_df)
    )

    print(
        split_df["binary_diagnosis"]
        .value_counts()
    )

    print(
        "\nBinary counts:"
    )

    print(
        split_df["binary_label"]
        .value_counts()
        .sort_index()
    )


print_split(
    "TRAIN",
    train_df
)

print_split(
    "VALIDATION",
    validation_df
)

print_split(
    "TEST",
    test_df
)


print("\n" + "=" * 75)
print("OUTPUT FILES")
print("=" * 75)

print("\nTrain:")
print(train_path)

print("\nValidation:")
print(validation_path)

print("\nTest:")
print(test_path)


print("\n" + "=" * 75)
print("PH2 DATASET SPLIT COMPLETED")
print("=" * 75)

print("\nTotal records:", len(df))

print(
    "Train + Validation + Test:",
    len(train_df)
    + len(validation_df)
    + len(test_df)
)

print("\nStatus: PASS")

print("=" * 75)