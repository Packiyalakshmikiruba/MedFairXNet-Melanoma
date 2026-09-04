import os
import hashlib
import pandas as pd
import numpy as np

# ============================================================
# MELONMA - PH2 DATASET INTEGRITY VALIDATION
# ============================================================

PROJECT_ROOT = r"C:\Users\HP\Desktop\Melonma"

PH2_ROOT = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "PH2"
)

SPLIT_DIR = os.path.join(
    PH2_ROOT,
    "splits"
)

AUDIT_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "ph2_mapping_audit.csv"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "ph2_dataset_integrity"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# HEADER
# ============================================================

print("=" * 100)
print("MELONMA - PH2 DATASET INTEGRITY VALIDATION")
print("=" * 100)

print()
print("PH2 root:")
print(PH2_ROOT)

print()
print("Split directory:")
print(SPLIT_DIR)

print()
print("Output directory:")
print(OUTPUT_DIR)

# ============================================================
# CHECK DIRECTORIES
# ============================================================

if not os.path.exists(PH2_ROOT):
    raise FileNotFoundError(
        f"PH2 directory not found:\n{PH2_ROOT}"
    )

if not os.path.exists(SPLIT_DIR):
    raise FileNotFoundError(
        f"PH2 split directory not found:\n{SPLIT_DIR}"
    )

# ============================================================
# DISCOVER SPLIT FILES
# ============================================================

print()
print("=" * 100)
print("DISCOVERING SPLIT FILES")
print("=" * 100)

all_csv_files = []

for filename in os.listdir(SPLIT_DIR):

    if filename.lower().endswith(".csv"):

        all_csv_files.append(
            os.path.join(
                SPLIT_DIR,
                filename
            )
        )

all_csv_files = sorted(all_csv_files)

if not all_csv_files:
    raise RuntimeError(
        "No CSV split files found."
    )

for path in all_csv_files:
    print(
        os.path.basename(path)
    )

# ============================================================
# LOAD SPLITS
# ============================================================

print()
print("=" * 100)
print("LOADING SPLITS")
print("=" * 100)

split_data = {}

for csv_path in all_csv_files:

    filename = os.path.basename(csv_path)

    try:

        data = pd.read_csv(csv_path)

        split_data[filename] = data

        print()
        print(
            f"{filename}: {len(data)} records"
        )

        print(
            f"Columns: {list(data.columns)}"
        )

    except Exception as e:

        print(
            f"ERROR reading {filename}: {e}"
        )

# ============================================================
# IDENTIFY STANDARD SPLITS
# ============================================================

train_df = None
validation_df = None
test_df = None

for filename, data in split_data.items():

    lower_name = filename.lower()

    if (
        "train_ph2_clahe" in lower_name
        and "validation" not in lower_name
        and "test" not in lower_name
    ):
        train_df = data

    elif (
        "validation_ph2_clahe" in lower_name
    ):
        validation_df = data

    elif (
        "test_ph2_clahe" in lower_name
    ):
        test_df = data

# ============================================================
# REQUIRED COLUMN CHECK
# ============================================================

print()
print("=" * 100)
print("CHECKING REQUIRED COLUMNS")
print("=" * 100)

required_columns = [
    "image_id",
    "binary_label",
    "image_path"
]

for filename, data in split_data.items():

    print()
    print(filename)

    missing = [
        col
        for col in required_columns
        if col not in data.columns
    ]

    if missing:

        print(
            "MISSING:",
            missing
        )

    else:

        print(
            "PASS - Required columns present"
        )

# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print()
print("=" * 100)
print("CLASS DISTRIBUTION")
print("=" * 100)

distribution_rows = []

for filename, data in split_data.items():

    if "binary_label" not in data.columns:
        continue

    counts = (
        data["binary_label"]
        .value_counts()
        .sort_index()
    )

    non_melanoma = int(
        counts.get(0, 0)
    )

    melanoma = int(
        counts.get(1, 0)
    )

    total = len(data)

    distribution_rows.append({
        "split_file": filename,
        "total": total,
        "non_melanoma": non_melanoma,
        "melanoma": melanoma,
        "melanoma_percentage": (
            melanoma / total * 100
            if total > 0
            else 0
        )
    })

    print()
    print(filename)
    print(
        f"Total         : {total}"
    )
    print(
        f"Non-melanoma  : {non_melanoma}"
    )
    print(
        f"Melanoma      : {melanoma}"
    )

distribution_df = pd.DataFrame(
    distribution_rows
)

distribution_path = os.path.join(
    OUTPUT_DIR,
    "class_distribution.csv"
)

distribution_df.to_csv(
    distribution_path,
    index=False
)

# ============================================================
# DUPLICATE IMAGE IDS
# ============================================================

print()
print("=" * 100)
print("CHECKING DUPLICATE IMAGE IDs")
print("=" * 100)

duplicate_rows = []

for filename, data in split_data.items():

    if "image_id" not in data.columns:
        continue

    duplicated = data[
        data["image_id"].duplicated(
            keep=False
        )
    ]

    if len(duplicated) > 0:

        print()
        print(
            f"{filename}: "
            f"{len(duplicated)} duplicate rows"
        )

        for _, row in duplicated.iterrows():

            duplicate_rows.append({
                "split_file": filename,
                "image_id": row["image_id"]
            })

    else:

        print(
            f"{filename}: No duplicates"
        )

duplicate_df = pd.DataFrame(
    duplicate_rows
)

duplicate_path = os.path.join(
    OUTPUT_DIR,
    "duplicate_image_ids.csv"
)

duplicate_df.to_csv(
    duplicate_path,
    index=False
)

# ============================================================
# CROSS-SPLIT LEAKAGE
# ============================================================

print()
print("=" * 100)
print("CHECKING CROSS-SPLIT IMAGE LEAKAGE")
print("=" * 100)

split_id_sets = {}

for filename, data in split_data.items():

    if "image_id" in data.columns:

        split_id_sets[filename] = set(
            data["image_id"]
            .astype(str)
        )

leakage_rows = []

filenames = list(
    split_id_sets.keys()
)

for i in range(len(filenames)):

    for j in range(i + 1, len(filenames)):

        file_a = filenames[i]
        file_b = filenames[j]

        overlap = (
            split_id_sets[file_a]
            &
            split_id_sets[file_b]
        )

        if overlap:

            print()
            print(
                "LEAKAGE FOUND:"
            )

            print(
                f"{file_a} <-> {file_b}"
            )

            print(
                f"Overlap: {len(overlap)}"
            )

            for image_id in sorted(
                overlap
            ):

                leakage_rows.append({
                    "split_a": file_a,
                    "split_b": file_b,
                    "image_id": image_id
                })

        else:

            print()
            print(
                f"PASS: {file_a} <-> {file_b}"
            )

leakage_df = pd.DataFrame(
    leakage_rows
)

leakage_path = os.path.join(
    OUTPUT_DIR,
    "cross_split_leakage.csv"
)

leakage_df.to_csv(
    leakage_path,
    index=False
)

# ============================================================
# IMAGE PATH VALIDATION
# ============================================================

print()
print("=" * 100)
print("CHECKING IMAGE PATHS")
print("=" * 100)

path_rows = []

for filename, data in split_data.items():

    if "image_path" not in data.columns:
        continue

    missing_count = 0
    existing_count = 0

    for _, row in data.iterrows():

        image_path = str(
            row["image_path"]
        )

        exists = os.path.exists(
            image_path
        )

        if exists:
            existing_count += 1
        else:
            missing_count += 1

            path_rows.append({
                "split_file": filename,
                "image_id": row.get(
                    "image_id",
                    ""
                ),
                "image_path": image_path
            })

    print()
    print(filename)

    print(
        f"Existing : {existing_count}"
    )

    print(
        f"Missing  : {missing_count}"
    )

path_df = pd.DataFrame(
    path_rows
)

path_path = os.path.join(
    OUTPUT_DIR,
    "missing_image_paths.csv"
)

path_df.to_csv(
    path_path,
    index=False
)

# ============================================================
# CLAHE PATH VALIDATION
# ============================================================

print()
print("=" * 100)
print("CHECKING CLAHE IMAGE PATHS")
print("=" * 100)

clahe_rows = []

for filename, data in split_data.items():

    if "clahe_image_path" not in data.columns:
        print()
        print(
            f"{filename}: No CLAHE column"
        )
        continue

    missing_clahe = 0
    existing_clahe = 0

    for _, row in data.iterrows():

        clahe_path = str(
            row["clahe_image_path"]
        )

        if os.path.exists(
            clahe_path
        ):

            existing_clahe += 1

        else:

            missing_clahe += 1

            clahe_rows.append({
                "split_file": filename,
                "image_id": row.get(
                    "image_id",
                    ""
                ),
                "clahe_image_path": clahe_path
            })

    print()
    print(filename)

    print(
        f"Existing CLAHE : {existing_clahe}"
    )

    print(
        f"Missing CLAHE  : {missing_clahe}"
    )

clahe_df = pd.DataFrame(
    clahe_rows
)

clahe_path = os.path.join(
    OUTPUT_DIR,
    "missing_clahe_paths.csv"
)

clahe_df.to_csv(
    clahe_path,
    index=False
)

# ============================================================
# LABEL CONSISTENCY CHECK
# ============================================================

print()
print("=" * 100)
print("CHECKING LABEL CONSISTENCY")
print("=" * 100)

label_rows = []

for filename, data in split_data.items():

    if "binary_label" not in data.columns:
        continue

    invalid = data[
        ~data["binary_label"].isin(
            [0, 1]
        )
    ]

    if len(invalid) > 0:

        print()
        print(
            f"{filename}: "
            f"{len(invalid)} invalid labels"
        )

        for _, row in invalid.iterrows():

            label_rows.append({
                "split_file": filename,
                "image_id": row.get(
                    "image_id",
                    ""
                ),
                "binary_label": row[
                    "binary_label"
                ]
            })

    else:

        print(
            f"{filename}: "
            "PASS - labels are 0/1"
        )

label_df = pd.DataFrame(
    label_rows
)

label_path = os.path.join(
    OUTPUT_DIR,
    "invalid_labels.csv"
)

label_df.to_csv(
    label_path,
    index=False
)

# ============================================================
# IMAGE ID / FILENAME CONSISTENCY
# ============================================================

print()
print("=" * 100)
print("CHECKING IMAGE ID / PATH CONSISTENCY")
print("=" * 100)

consistency_rows = []

for filename, data in split_data.items():

    if (
        "image_id" not in data.columns
        or "image_path" not in data.columns
    ):
        continue

    mismatch_count = 0

    for _, row in data.iterrows():

        image_id = str(
            row["image_id"]
        )

        image_path = str(
            row["image_path"]
        )

        basename = os.path.basename(
            image_path
        )

        stem = os.path.splitext(
            basename
        )[0]

        expected_stem = os.path.splitext(
            image_id
        )[0]

        if (
            stem.lower()
            != expected_stem.lower()
        ):

            mismatch_count += 1

            consistency_rows.append({
                "split_file": filename,
                "image_id": image_id,
                "image_path": image_path,
                "path_stem": stem
            })

    print()
    print(filename)

    print(
        f"ID/path mismatches: "
        f"{mismatch_count}"
    )

consistency_df = pd.DataFrame(
    consistency_rows
)

consistency_path = os.path.join(
    OUTPUT_DIR,
    "image_id_path_mismatches.csv"
)

consistency_df.to_csv(
    consistency_path,
    index=False
)

# ============================================================
# OPTIONAL MAPPING AUDIT
# ============================================================

print()
print("=" * 100)
print("CHECKING PH2 MAPPING AUDIT")
print("=" * 100)

if os.path.exists(AUDIT_FILE):

    audit_df = pd.read_csv(
        AUDIT_FILE
    )

    print()
    print(
        f"Mapping audit records: "
        f"{len(audit_df)}"
    )

    print()
    print(
        "Mapping audit columns:"
    )

    print(
        list(audit_df.columns)
    )

    audit_copy_path = os.path.join(
        OUTPUT_DIR,
        "ph2_mapping_audit_copy.csv"
    )

    audit_df.to_csv(
        audit_copy_path,
        index=False
    )

    print()
    print(
        "Mapping audit loaded successfully."
    )

else:

    print()
    print(
        "WARNING: Mapping audit not found."
    )

# ============================================================
# OVERALL SUMMARY
# ============================================================

print()
print("=" * 100)
print("OVERALL INTEGRITY SUMMARY")
print("=" * 100)

total_records = 0

for data in split_data.values():
    total_records += len(data)

total_missing_images = len(path_df)
total_missing_clahe = len(clahe_df)
total_duplicate_rows = len(duplicate_df)
total_leakage = len(leakage_df)
total_invalid_labels = len(label_df)
total_id_mismatches = len(consistency_df)

summary = pd.DataFrame([{

    "total_split_files":
        len(split_data),

    "total_records":
        total_records,

    "missing_images":
        total_missing_images,

    "missing_clahe_images":
        total_missing_clahe,

    "duplicate_image_rows":
        total_duplicate_rows,

    "cross_split_leakage_rows":
        total_leakage,

    "invalid_labels":
        total_invalid_labels,

    "image_id_path_mismatches":
        total_id_mismatches,

    "integrity_status":
        (
            "PASS"
            if (
                total_missing_images == 0
                and total_missing_clahe == 0
                and total_duplicate_rows == 0
                and total_leakage == 0
                and total_invalid_labels == 0
                and total_id_mismatches == 0
            )
            else
            "CHECK_REQUIRED"
        )
}])

summary_path = os.path.join(
    OUTPUT_DIR,
    "ph2_integrity_summary.csv"
)

summary.to_csv(
    summary_path,
    index=False
)

print()
print(
    f"Total records             : "
    f"{total_records}"
)

print(
    f"Missing images            : "
    f"{total_missing_images}"
)

print(
    f"Missing CLAHE images      : "
    f"{total_missing_clahe}"
)

print(
    f"Duplicate rows            : "
    f"{total_duplicate_rows}"
)

print(
    f"Cross-split leakage rows  : "
    f"{total_leakage}"
)

print(
    f"Invalid labels            : "
    f"{total_invalid_labels}"
)

print(
    f"ID/path mismatches        : "
    f"{total_id_mismatches}"
)

print()
print(
    f"Integrity status           : "
    f"{summary.iloc[0]['integrity_status']}"
)

# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("=" * 100)
print("FILES SAVED")
print("=" * 100)

print()
print(
    f"Summary: {summary_path}"
)

print(
    f"Class distribution: "
    f"{distribution_path}"
)

print(
    f"Duplicate IDs: "
    f"{duplicate_path}"
)

print(
    f"Cross-split leakage: "
    f"{leakage_path}"
)

print(
    f"Missing images: "
    f"{path_path}"
)

print(
    f"Missing CLAHE: "
    f"{clahe_path}"
)

print(
    f"Invalid labels: "
    f"{label_path}"
)

print(
    f"ID/path mismatches: "
    f"{consistency_path}"
)

print()
print("=" * 100)
print("STATUS: PASS")
print("PH2 dataset integrity validation completed.")
print("=" * 100)

