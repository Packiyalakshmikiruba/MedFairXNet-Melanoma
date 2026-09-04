from pathlib import Path
import pandas as pd

# ============================================================
# HAM10000 DATASET INSPECTION
# ============================================================

DATASET_DIR = Path(
    r"C:\Users\HP\Desktop\Melonma\data\raw\HAM10000"
)

METADATA_FILE = DATASET_DIR / "HAM10000_metadata.csv"

print("=" * 70)
print("HAM10000 DATASET INSPECTION")
print("=" * 70)

# ------------------------------------------------------------
# Check folder
# ------------------------------------------------------------

if not DATASET_DIR.exists():
    print("\nERROR: Dataset folder not found!")
    print(DATASET_DIR)
    raise SystemExit

print("\nDataset folder:")
print(DATASET_DIR)

# ------------------------------------------------------------
# Find images
# ------------------------------------------------------------

image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

images = [
    p for p in DATASET_DIR.rglob("*")
    if p.is_file() and p.suffix.lower() in image_extensions
]

print("\n" + "=" * 70)
print("IMAGE INFORMATION")
print("=" * 70)

print(f"Total image files: {len(images)}")

print("\nFirst 10 images:")

for image in images[:10]:
    print(" -", image.relative_to(DATASET_DIR))

# ------------------------------------------------------------
# Metadata
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("METADATA INFORMATION")
print("=" * 70)

if not METADATA_FILE.exists():
    print("ERROR: HAM10000_metadata.csv not found!")
    print(METADATA_FILE)
    raise SystemExit

df = pd.read_csv(METADATA_FILE)

print(f"\nMetadata records: {len(df)}")

print("\nColumns:")
print(list(df.columns))

print("\nFirst 5 rows:")
print(df.head())

# ------------------------------------------------------------
# Missing values
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)

print(df.isnull().sum())

# ------------------------------------------------------------
# Diagnosis distribution
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DIAGNOSIS DISTRIBUTION")
print("=" * 70)

if "dx" in df.columns:
    print(df["dx"].value_counts())

# ------------------------------------------------------------
# Sex distribution
# ------------------------------------------------------------

if "sex" in df.columns:
    print("\n" + "=" * 70)
    print("SEX DISTRIBUTION")
    print("=" * 70)

    print(df["sex"].value_counts(dropna=False))

# ------------------------------------------------------------
# Age statistics
# ------------------------------------------------------------

if "age" in df.columns:
    print("\n" + "=" * 70)
    print("AGE INFORMATION")
    print("=" * 70)

    print(df["age"].describe())

# ------------------------------------------------------------
# Duplicate check
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DUPLICATE CHECK")
print("=" * 70)

if "image_id" in df.columns:
    duplicate_ids = df["image_id"].duplicated().sum()
    print(f"Duplicate image IDs: {duplicate_ids}")

# ------------------------------------------------------------
# Image ↔ Metadata check
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("IMAGE ↔ METADATA CHECK")
print("=" * 70)

if "image_id" in df.columns:

    metadata_ids = set(df["image_id"].astype(str))

    image_ids = {
        image.stem
        for image in images
    }

    missing_images = metadata_ids - image_ids
    images_without_metadata = image_ids - metadata_ids

    print(f"Metadata IDs: {len(metadata_ids)}")
    print(f"Image IDs: {len(image_ids)}")

    print(f"Missing images: {len(missing_images)}")
    print(f"Images without metadata: {len(images_without_metadata)}")

# ------------------------------------------------------------
# Final summary
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print(f"Total images       : {len(images)}")
print(f"Metadata records   : {len(df)}")

if "dx" in df.columns:
    print("\nDiagnosis distribution:")
    print(df["dx"].value_counts())

print("\nHAM10000 inspection completed.")
print("=" * 70)