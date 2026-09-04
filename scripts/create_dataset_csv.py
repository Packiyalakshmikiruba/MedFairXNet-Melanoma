from pathlib import Path
import pandas as pd


# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = BASE_DIR / "data" / "raw" / "HAM10000"

METADATA_FILE = DATASET_DIR / "HAM10000_metadata.csv"

OUTPUT_DIR = BASE_DIR / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "dataset.csv"


# --------------------------------------------------
# CREATE OUTPUT DIRECTORY
# --------------------------------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# LOAD METADATA
# --------------------------------------------------

print("=" * 70)
print("CREATING HAM10000 DATASET CSV")
print("=" * 70)

df = pd.read_csv(METADATA_FILE)

print(f"\nMetadata records: {len(df)}")


# --------------------------------------------------
# FIND ALL IMAGES
# --------------------------------------------------

image_files = list(DATASET_DIR.rglob("*.jpg"))

print(f"Image files: {len(image_files)}")


# --------------------------------------------------
# CREATE IMAGE ID → PATH MAPPING
# --------------------------------------------------

image_map = {
    image.stem: str(image.relative_to(BASE_DIR))
    for image in image_files
}


# --------------------------------------------------
# MAP IMAGE PATH
# --------------------------------------------------

df["image_path"] = df["image_id"].map(image_map)


# --------------------------------------------------
# CHECK MISSING IMAGE PATHS
# --------------------------------------------------

missing_images = df["image_path"].isna().sum()

print(f"Missing image paths: {missing_images}")


# --------------------------------------------------
# REMOVE ROWS WITHOUT IMAGES
# --------------------------------------------------

df = df.dropna(subset=["image_path"]).copy()


# --------------------------------------------------
# HANDLE MISSING VALUES
# --------------------------------------------------

df["age"] = df["age"].fillna(df["age"].median())

df["sex"] = df["sex"].fillna("unknown")

df["localization"] = df["localization"].fillna("unknown")


# --------------------------------------------------
# SAVE DATASET
# --------------------------------------------------

df.to_csv(OUTPUT_FILE, index=False)


# --------------------------------------------------
# FINAL INFORMATION
# --------------------------------------------------

print("\n" + "=" * 70)
print("DATASET CSV CREATED")
print("=" * 70)

print(f"\nOutput file:")
print(OUTPUT_FILE)

print(f"\nTotal records: {len(df)}")

print("\nColumns:")
print(df.columns.tolist())

print("\nDiagnosis distribution:")
print(df["dx"].value_counts())

print("\nFirst 5 records:")
print(df.head())

print("\n" + "=" * 70)