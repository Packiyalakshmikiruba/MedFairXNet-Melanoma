from pathlib import Path
import pandas as pd

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_CSV = BASE_DIR / "data" / "processed" / "dataset.csv"
OUTPUT_CSV = BASE_DIR / "data" / "processed" / "dataset_clean.csv"


print("=" * 70)
print("PREPARING HAM10000 DATASET")
print("=" * 70)

# Load dataset
df = pd.read_csv(INPUT_CSV)

print(f"\nOriginal records: {len(df)}")

# ---------------------------------------------------------
# 1. Remove duplicate image IDs
# ---------------------------------------------------------

before = len(df)

df = df.drop_duplicates(subset=["image_id"])

after = len(df)

print(f"Duplicate records removed: {before - after}")

# ---------------------------------------------------------
# 2. Handle missing age
# ---------------------------------------------------------

missing_age = df["age"].isna().sum()

print(f"Missing age values: {missing_age}")

# Fill missing age with median
df["age"] = df["age"].fillna(df["age"].median())

print(f"Missing age after filling: {df['age'].isna().sum()}")

# ---------------------------------------------------------
# 3. Check missing values
# ---------------------------------------------------------

print("\nMissing values after preprocessing:")

print(df.isnull().sum())

# ---------------------------------------------------------
# 4. Diagnosis mapping
# ---------------------------------------------------------

diagnosis_names = {
    "nv": "Melanocytic Nevus",
    "mel": "Melanoma",
    "bkl": "Benign Keratosis",
    "bcc": "Basal Cell Carcinoma",
    "akiec": "Actinic Keratosis",
    "vasc": "Vascular Lesion",
    "df": "Dermatofibroma"
}

df["diagnosis_name"] = df["dx"].map(diagnosis_names)

# ---------------------------------------------------------
# 5. Save cleaned dataset
# ---------------------------------------------------------

df.to_csv(OUTPUT_CSV, index=False)

print("\n" + "=" * 70)
print("CLEAN DATASET CREATED")
print("=" * 70)

print(f"\nOutput file:")
print(OUTPUT_CSV)

print(f"\nTotal records: {len(df)}")

print("\nColumns:")
print(df.columns.tolist())

print("\nDiagnosis distribution:")
print(df["dx"].value_counts())

print("\nFirst 5 records:")
print(df.head())

print("\nDataset preparation completed successfully.")
print("=" * 70)