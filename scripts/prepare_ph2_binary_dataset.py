from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

LABEL_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "ph2_labels.csv"
)

PH2_ROOT = (
    BASE_DIR
    / "data"
    / "raw"
    / "PH2"
    / "PH2Dataset"
    / "PH2 Dataset images"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "PH2"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "ph2_binary_metadata.csv"
)


print("=" * 75)
print("MELONMA - PH2 BINARY DATASET PREPARATION")
print("=" * 75)

print("\nLabel file:")
print(LABEL_FILE)

print("\nPH2 image root:")
print(PH2_ROOT)


if not LABEL_FILE.exists():
    raise FileNotFoundError(
        f"PH2 label file not found:\n{LABEL_FILE}"
    )

if not PH2_ROOT.exists():
    raise FileNotFoundError(
        f"PH2 image directory not found:\n{PH2_ROOT}"
    )


df = pd.read_csv(LABEL_FILE)

required_columns = [
    "image_id",
    "clinical_class",
]

for column in required_columns:

    if column not in df.columns:

        raise ValueError(
            f"Missing required column: {column}"
        )


print("\nTotal PH2 records:", len(df))


# ---------------------------------------------------------
# BINARY LABEL
# ---------------------------------------------------------

BINARY_MAPPING = {
    "melanoma": 1,
    "common_nevus": 0,
    "atypical_nevus": 0,
}


unknown_classes = (
    set(df["clinical_class"].dropna().unique())
    - set(BINARY_MAPPING.keys())
)

if unknown_classes:

    raise ValueError(
        f"Unknown PH2 classes found: {unknown_classes}"
    )


df["binary_label"] = (
    df["clinical_class"]
    .map(BINARY_MAPPING)
    .astype(int)
)


df["binary_diagnosis"] = (
    df["binary_label"]
    .map({
        0: "Non-melanoma",
        1: "Melanoma"
    })
)


# ---------------------------------------------------------
# IMAGE + MASK PATHS
# ---------------------------------------------------------

def get_image_path(image_id):

    return (
        PH2_ROOT
        / image_id
        / f"{image_id}_Dermoscopic_Image"
        / f"{image_id}.bmp"
    )


def get_lesion_mask_path(image_id):

    return (
        PH2_ROOT
        / image_id
        / f"{image_id}_lesion"
        / f"{image_id}_lesion.bmp"
    )


df["image_path"] = (
    df["image_id"]
    .apply(
        lambda x: str(
            get_image_path(x)
        )
    )
)


df["lesion_mask_path"] = (
    df["image_id"]
    .apply(
        lambda x: str(
            get_lesion_mask_path(x)
        )
    )
)


# ---------------------------------------------------------
# FILE VALIDATION
# ---------------------------------------------------------

df["image_exists"] = (
    df["image_path"]
    .apply(
        lambda x: Path(x).exists()
    )
)


df["mask_exists"] = (
    df["lesion_mask_path"]
    .apply(
        lambda x: Path(x).exists()
    )
)


missing_images = (
    (~df["image_exists"])
    .sum()
)

missing_masks = (
    (~df["mask_exists"])
    .sum()
)


print("\n" + "=" * 75)
print("FILE VALIDATION")
print("=" * 75)

print(
    "\nMissing PH2 images:",
    missing_images
)

print(
    "Missing PH2 lesion masks:",
    missing_masks
)


if missing_images > 0:

    print("\nMissing image paths:")

    print(
        df.loc[
            ~df["image_exists"],
            "image_path"
        ]
        .head(20)
        .to_string(index=False)
    )

    raise FileNotFoundError(
        "Some PH2 dermoscopic images are missing."
    )


if missing_masks > 0:

    print("\nMissing mask paths:")

    print(
        df.loc[
            ~df["mask_exists"],
            "lesion_mask_path"
        ]
        .head(20)
        .to_string(index=False)
    )

    raise FileNotFoundError(
        "Some PH2 lesion masks are missing."
    )


# ---------------------------------------------------------
# REMOVE TEMPORARY VALIDATION COLUMNS
# ---------------------------------------------------------

df = df.drop(
    columns=[
        "image_exists",
        "mask_exists",
    ]
)


# ---------------------------------------------------------
# DUPLICATE CHECK
# ---------------------------------------------------------

duplicate_ids = (
    df["image_id"]
    .duplicated()
    .sum()
)

print(
    "\nDuplicate image IDs:",
    duplicate_ids
)


if duplicate_ids > 0:

    raise ValueError(
        "Duplicate PH2 image IDs detected."
    )


# ---------------------------------------------------------
# CLASS DISTRIBUTION
# ---------------------------------------------------------

print("\n" + "=" * 75)
print("PH2 BINARY CLASS DISTRIBUTION")
print("=" * 75)

print(
    df["binary_diagnosis"]
    .value_counts()
)

print("\nBinary labels:")

print(
    df["binary_label"]
    .value_counts()
    .sort_index()
)


print("\nOriginal PH2 classes:")

print(
    df["clinical_class"]
    .value_counts()
)


# ---------------------------------------------------------
# SAVE
# ---------------------------------------------------------

df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\n" + "=" * 75)
print("PH2 BINARY DATASET CREATED")
print("=" * 75)

print("\nOutput:")

print(OUTPUT_FILE)

print("\nTotal records:", len(df))

print("\nStatus: PASS")

print("=" * 75)