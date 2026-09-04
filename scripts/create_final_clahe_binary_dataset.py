from pathlib import Path
import pandas as pd


# =========================================================
# CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "binary_roi_splits"
)

CLAHE_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "images_clahe"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "final_binary_clahe_splits"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


FILES = [
    "train_binary_roi.csv",
    "validation_binary_roi.csv",
    "test_binary_roi.csv",
]


# =========================================================
# HEADER
# =========================================================

print("=" * 75)
print("MELONMA - FINAL CLAHE BINARY DATASET CREATION")
print("=" * 75)

print("\nInput metadata:")
print(INPUT_DIR)

print("\nCLAHE images:")
print(CLAHE_DIR)

print("\nOutput:")
print(OUTPUT_DIR)


# =========================================================
# CHECK CLAHE DIRECTORY
# =========================================================

if not CLAHE_DIR.exists():
    raise FileNotFoundError(
        f"CLAHE directory not found:\n{CLAHE_DIR}"
    )


# =========================================================
# PROCESS SPLITS
# =========================================================

all_image_ids = {}

total_records = 0
total_missing = 0


for filename in FILES:

    print("\n" + "=" * 75)
    print(f"PROCESSING: {filename}")
    print("=" * 75)

    input_csv = INPUT_DIR / filename

    if not input_csv.exists():
        raise FileNotFoundError(
            f"CSV not found:\n{input_csv}"
        )

    df = pd.read_csv(input_csv)

    print("\nRecords:", len(df))

    # -----------------------------------------------------
    # Required columns
    # -----------------------------------------------------

    required_columns = [
        "image_id",
        "binary_label",
        "binary_diagnosis",
        "image_path",
    ]

    missing_columns = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns in {filename}: "
            f"{missing_columns}"
        )

    # -----------------------------------------------------
    # Create CLAHE path
    # -----------------------------------------------------

    df["original_roi_path"] = df["image_path"]

    df["image_path"] = df["image_path"].apply(
        lambda p: (
            Path(
                "data"
            )
            / "processed"
            / "images_clahe"
            / Path(p).name
        ).as_posix()
    )

    # -----------------------------------------------------
    # Verify CLAHE files
    # -----------------------------------------------------

    missing = []

    for path in df["image_path"]:

        absolute_path = BASE_DIR / path

        if not absolute_path.exists():
            missing.append(path)

    print(
        "Missing CLAHE images:",
        len(missing)
    )

    total_missing += len(missing)

    if missing:

        print("\nFirst missing files:")

        for path in missing[:10]:
            print(path)

        raise FileNotFoundError(
            f"{len(missing)} CLAHE images are missing."
        )

    # -----------------------------------------------------
    # Check binary labels
    # -----------------------------------------------------

    labels = set(
        df["binary_label"]
        .astype(int)
        .unique()
    )

    print(
        "Binary labels:",
        sorted(labels)
    )

    if not labels.issubset({0, 1}):
        raise ValueError(
            f"Invalid binary labels: {labels}"
        )

    # -----------------------------------------------------
    # Check image ID uniqueness
    # -----------------------------------------------------

    duplicate_ids = (
        df["image_id"]
        .duplicated()
        .sum()
    )

    print(
        "Duplicate image IDs:",
        duplicate_ids
    )

    if duplicate_ids > 0:
        raise ValueError(
            f"Duplicate image IDs found in {filename}"
        )

    # -----------------------------------------------------
    # Store IDs for cross-split leakage check
    # -----------------------------------------------------

    split_name = filename.replace(
        "_binary_roi.csv",
        ""
    )

    all_image_ids[split_name] = set(
        df["image_id"]
    )

    # -----------------------------------------------------
    # Save final CSV
    # -----------------------------------------------------

    output_filename = (
        filename.replace(
            "_binary_roi.csv",
            "_final_clahe.csv"
        )
    )

    output_csv = (
        OUTPUT_DIR
        / output_filename
    )

    df.to_csv(
        output_csv,
        index=False
    )

    print(
        "\nSaved:",
        output_csv
    )

    total_records += len(df)


# =========================================================
# CROSS-SPLIT LEAKAGE CHECK
# =========================================================

print("\n" + "=" * 75)
print("CROSS-SPLIT IMAGE LEAKAGE CHECK")
print("=" * 75)

train_ids = all_image_ids["train"]
val_ids = all_image_ids["validation"]
test_ids = all_image_ids["test"]

train_val = train_ids & val_ids
train_test = train_ids & test_ids
val_test = val_ids & test_ids

print(
    "\nTrain ∩ Validation:",
    len(train_val)
)

print(
    "Train ∩ Test:",
    len(train_test)
)

print(
    "Validation ∩ Test:",
    len(val_test)
)

if train_val or train_test or val_test:

    raise ValueError(
        "DATA LEAKAGE DETECTED BETWEEN SPLITS."
    )


# =========================================================
# TOTAL UNIQUE IMAGES
# =========================================================

all_ids = (
    train_ids
    | val_ids
    | test_ids
)

print(
    "\nTotal records:",
    total_records
)

print(
    "Total unique image IDs:",
    len(all_ids)
)

print(
    "Total missing CLAHE images:",
    total_missing
)


# =========================================================
# CLASS DISTRIBUTION
# =========================================================

print("\n" + "=" * 75)
print("FINAL CLASS DISTRIBUTION")
print("=" * 75)

for split_name in [
    "train",
    "validation",
    "test",
]:

    csv_name = (
        f"{split_name}_final_clahe.csv"
    )

    df = pd.read_csv(
        OUTPUT_DIR / csv_name
    )

    print(
        f"\n=== {split_name.upper()} ==="
    )

    print(
        df["binary_diagnosis"]
        .value_counts()
    )


# =========================================================
# FINAL STATUS
# =========================================================

print("\n" + "=" * 75)
print("FINAL CLAHE DATASET CREATION COMPLETED")
print("=" * 75)

if total_missing == 0:

    print(
        "\nSTATUS: PASS"
    )

    print(
        "All binary split metadata now points to "
        "the CLAHE-preprocessed images."
    )

else:

    print(
        "\nSTATUS: WARNING"
    )

    print(
        "Missing CLAHE images:",
        total_missing
    )

print("\nOutput directory:")
print(OUTPUT_DIR)

print("=" * 75)