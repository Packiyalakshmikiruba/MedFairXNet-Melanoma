from pathlib import Path
import pandas as pd
from itertools import combinations

# ============================================================
# MELONMA - PH2 CORRECTED SPLIT LEAKAGE VALIDATION
# ============================================================

PROJECT_ROOT = Path(r"C:\Users\HP\Desktop\Melonma")

SPLIT_DIR = PROJECT_ROOT / "data" / "processed" / "PH2" / "splits"

OUTPUT_DIR = PROJECT_ROOT / "results" / "ph2_dataset_integrity_corrected"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 100)
print("MELONMA - PH2 CORRECTED SPLIT LEAKAGE VALIDATION")
print("=" * 100)

print()
print("Split directory:")
print(SPLIT_DIR)

print()
print("Output directory:")
print(OUTPUT_DIR)

# ------------------------------------------------------------
# BASE SPLITS ONLY
# ------------------------------------------------------------

split_files = {
    "train": SPLIT_DIR / "train_ph2.csv",
    "validation": SPLIT_DIR / "validation_ph2.csv",
    "test": SPLIT_DIR / "test_ph2.csv",
}

print()
print("=" * 100)
print("LOADING BASE SPLITS")
print("=" * 100)

data = {}

for split_name, file_path in split_files.items():

    print()
    print(f"{split_name.upper()}")

    if not file_path.exists():
        raise FileNotFoundError(
            f"Missing split file:\n{file_path}"
        )

    df = pd.read_csv(file_path)

    data[split_name] = df

    print(f"File   : {file_path}")
    print(f"Records: {len(df)}")

    required_columns = [
        "image_id",
        "binary_label",
        "image_path"
    ]

    missing_columns = [
        c for c in required_columns
        if c not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{split_name} missing columns: {missing_columns}"
        )

    print("Required columns: PASS")

# ------------------------------------------------------------
# CLASS DISTRIBUTION
# ------------------------------------------------------------

print()
print("=" * 100)
print("CLASS DISTRIBUTION")
print("=" * 100)

class_rows = []

for split_name, df in data.items():

    non_melanoma = int((df["binary_label"] == 0).sum())
    melanoma = int((df["binary_label"] == 1).sum())
    total = len(df)

    print()
    print(split_name.upper())
    print(f"Total        : {total}")
    print(f"Non-melanoma : {non_melanoma}")
    print(f"Melanoma     : {melanoma}")

    class_rows.append({
        "split": split_name,
        "total": total,
        "non_melanoma": non_melanoma,
        "melanoma": melanoma
    })

class_df = pd.DataFrame(class_rows)

class_df.to_csv(
    OUTPUT_DIR / "corrected_class_distribution.csv",
    index=False
)

# ------------------------------------------------------------
# DUPLICATE IDS WITHIN EACH SPLIT
# ------------------------------------------------------------

print()
print("=" * 100)
print("CHECKING DUPLICATE IMAGE IDS")
print("=" * 100)

duplicate_rows = []

for split_name, df in data.items():

    duplicates = df[df["image_id"].duplicated(keep=False)]

    if len(duplicates) == 0:
        print(f"{split_name}: PASS - No duplicate image IDs")
    else:
        print(
            f"{split_name}: FAIL - "
            f"{len(duplicates)} duplicate rows"
        )

        for image_id in duplicates["image_id"].unique():
            duplicate_rows.append({
                "split": split_name,
                "image_id": image_id
            })

duplicate_df = pd.DataFrame(duplicate_rows)

duplicate_df.to_csv(
    OUTPUT_DIR / "corrected_duplicate_ids.csv",
    index=False
)

# ------------------------------------------------------------
# TRUE CROSS-SPLIT LEAKAGE
# ------------------------------------------------------------

print()
print("=" * 100)
print("CHECKING TRUE CROSS-SPLIT IMAGE LEAKAGE")
print("=" * 100)

leakage_rows = []

split_pairs = list(combinations(data.keys(), 2))

for split_a, split_b in split_pairs:

    ids_a = set(data[split_a]["image_id"].astype(str))
    ids_b = set(data[split_b]["image_id"].astype(str))

    overlap = sorted(ids_a.intersection(ids_b))

    if overlap:

        print()
        print(
            f"LEAKAGE FOUND: "
            f"{split_a} <-> {split_b}"
        )

        print(f"Overlap: {len(overlap)}")

        for image_id in overlap:

            leakage_rows.append({
                "split_a": split_a,
                "split_b": split_b,
                "image_id": image_id
            })

    else:

        print(
            f"PASS: "
            f"{split_a} <-> {split_b}"
        )

leakage_df = pd.DataFrame(leakage_rows)

leakage_df.to_csv(
    OUTPUT_DIR / "corrected_cross_split_leakage.csv",
    index=False
)

# ------------------------------------------------------------
# IMAGE PATH VALIDATION
# ------------------------------------------------------------

print()
print("=" * 100)
print("CHECKING IMAGE PATHS")
print("=" * 100)

missing_rows = []

for split_name, df in data.items():

    missing_count = 0

    for _, row in df.iterrows():

        image_path = Path(str(row["image_path"]))

        if not image_path.exists():

            missing_count += 1

            missing_rows.append({
                "split": split_name,
                "image_id": row["image_id"],
                "image_path": str(image_path)
            })

    existing_count = len(df) - missing_count

    print()
    print(split_name.upper())
    print(f"Existing : {existing_count}")
    print(f"Missing  : {missing_count}")

missing_df = pd.DataFrame(missing_rows)

missing_df.to_csv(
    OUTPUT_DIR / "corrected_missing_images.csv",
    index=False
)

# ------------------------------------------------------------
# LABEL VALIDATION
# ------------------------------------------------------------

print()
print("=" * 100)
print("CHECKING LABEL CONSISTENCY")
print("=" * 100)

invalid_rows = []

for split_name, df in data.items():

    invalid = df[
        ~df["binary_label"].isin([0, 1])
    ]

    if len(invalid) == 0:

        print(
            f"{split_name}: PASS - labels are 0/1"
        )

    else:

        print(
            f"{split_name}: FAIL - "
            f"{len(invalid)} invalid labels"
        )

        for _, row in invalid.iterrows():

            invalid_rows.append({
                "split": split_name,
                "image_id": row["image_id"],
                "binary_label": row["binary_label"]
            })

invalid_df = pd.DataFrame(invalid_rows)

invalid_df.to_csv(
    OUTPUT_DIR / "corrected_invalid_labels.csv",
    index=False
)

# ------------------------------------------------------------
# IMAGE ID / PATH CONSISTENCY
# ------------------------------------------------------------

print()
print("=" * 100)
print("CHECKING IMAGE ID / PATH CONSISTENCY")
print("=" * 100)

mismatch_rows = []

for split_name, df in data.items():

    mismatch_count = 0

    for _, row in df.iterrows():

        image_id = Path(
            str(row["image_id"])
        ).stem.lower()

        image_path = Path(
            str(row["image_path"])
        ).stem.lower()

        if image_id != image_path:

            mismatch_count += 1

            mismatch_rows.append({
                "split": split_name,
                "image_id": row["image_id"],
                "image_path": row["image_path"]
            })

    print()
    print(split_name.upper())
    print(f"ID/path mismatches: {mismatch_count}")

mismatch_df = pd.DataFrame(mismatch_rows)

mismatch_df.to_csv(
    OUTPUT_DIR / "corrected_id_path_mismatches.csv",
    index=False
)

# ------------------------------------------------------------
# FINAL STATUS
# ------------------------------------------------------------

total_records = sum(
    len(df)
    for df in data.values()
)

total_missing = len(missing_df)
total_duplicates = len(duplicate_df)
total_leakage = len(leakage_df)
total_invalid = len(invalid_df)
total_mismatches = len(mismatch_df)

print()
print("=" * 100)
print("FINAL CORRECTED INTEGRITY SUMMARY")
print("=" * 100)

print()
print(f"Total base split records : {total_records}")
print(f"Missing images           : {total_missing}")
print(f"Duplicate image IDs      : {total_duplicates}")
print(f"True cross-split leakage : {total_leakage}")
print(f"Invalid labels           : {total_invalid}")
print(f"ID/path mismatches       : {total_mismatches}")

if (
    total_missing == 0
    and total_duplicates == 0
    and total_leakage == 0
    and total_invalid == 0
    and total_mismatches == 0
):

    status = "PASS"

else:

    status = "CHECK_REQUIRED"

summary = pd.DataFrame([{
    "total_records": total_records,
    "missing_images": total_missing,
    "duplicate_image_ids": total_duplicates,
    "true_cross_split_leakage": total_leakage,
    "invalid_labels": total_invalid,
    "id_path_mismatches": total_mismatches,
    "integrity_status": status
}])

summary.to_csv(
    OUTPUT_DIR / "corrected_ph2_integrity_summary.csv",
    index=False
)

print()
print("=" * 100)
print("STATUS:", status)
print("=" * 100)

if status == "PASS":

    print()
    print(
        "PH2 BASE SPLIT INTEGRITY VALIDATION PASSED."
    )
    print(
        "Original and CLAHE versions were intentionally excluded "
        "from cross-split leakage comparison."
    )

else:

    print()
    print(
        "PH2 BASE SPLIT INTEGRITY REQUIRES REVIEW."
    )

print()
print("Files saved:")
print(
    OUTPUT_DIR /
    "corrected_ph2_integrity_summary.csv"
)

print(
    OUTPUT_DIR /
    "corrected_cross_split_leakage.csv"
)

print(
    OUTPUT_DIR /
    "corrected_class_distribution.csv"
)

print("=" * 100)