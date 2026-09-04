import os
import pandas as pd

BASE_DIR = r"C:\Users\HP\Desktop\Melonma\data\raw\ISIC2018"

CLASS_IMAGE_DIR = os.path.join(
    BASE_DIR,
    "ISIC2018_Task3_Training_Input"
)

GT_DIR = os.path.join(
    BASE_DIR,
    "ISIC2018_Task3_Training_GroundTruth"
)

GT_FILE = os.path.join(
    GT_DIR,
    "ISIC2018_Task3_Training_GroundTruth.csv"
)

SEG_IMAGE_DIR = os.path.join(
    BASE_DIR,
    "ISIC2018_Task1-2_Training_Input"
)

SEG_MASK_DIR = os.path.join(
    BASE_DIR,
    "ISIC2018_Task1_Training_GroundTruth"
)


print("=" * 70)
print("ISIC2018 DATASET VERIFICATION")
print("=" * 70)


# ============================================================
# CLASSIFICATION DATA
# ============================================================

images = [
    f for f in os.listdir(CLASS_IMAGE_DIR)
    if f.lower().endswith(".jpg")
]

print("\n" + "=" * 70)
print("TASK 3 - CLASSIFICATION")
print("=" * 70)

print("Training images:", len(images))


df = pd.read_csv(GT_FILE)

print("Ground truth records:", len(df))

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head().to_string())


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("CLASS DISTRIBUTION")
print("=" * 70)

class_columns = [
    "MEL",
    "NV",
    "BCC",
    "AKIEC",
    "BKL",
    "DF",
    "VASC"
]

for col in class_columns:
    if col in df.columns:
        print(f"{col}: {df[col].sum()}")


# ============================================================
# MELANOMA / NON-MELANOMA
# ============================================================

if "MEL" in df.columns:

    df["diagnosis"] = df["MEL"].apply(
        lambda x: "melanoma" if x == 1 else "non_melanoma"
    )

    print("\nMelanoma vs Non-Melanoma:")
    print(df["diagnosis"].value_counts())


# ============================================================
# DUPLICATES
# ============================================================

print("\n" + "=" * 70)
print("DUPLICATE CHECK")
print("=" * 70)

if "image" in df.columns:

    duplicates = df["image"].duplicated().sum()

    print("Duplicate IDs:", duplicates)


# ============================================================
# MISSING VALUES
# ============================================================

print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)

print(df.isnull().sum())


# ============================================================
# IMAGE ↔ CSV CHECK
# ============================================================

print("\n" + "=" * 70)
print("IMAGE ↔ LABEL CHECK")
print("=" * 70)

csv_ids = set(df["image"].astype(str))

image_ids = set(
    os.path.splitext(f)[0]
    for f in images
)

missing_images = csv_ids - image_ids
extra_images = image_ids - csv_ids

print("CSV IDs:", len(csv_ids))
print("Image IDs:", len(image_ids))
print("Missing images:", len(missing_images))
print("Images without labels:", len(extra_images))


# ============================================================
# SEGMENTATION DATA
# ============================================================

seg_images = [
    f for f in os.listdir(SEG_IMAGE_DIR)
    if f.lower().endswith(".jpg")
]

masks = [
    f for f in os.listdir(SEG_MASK_DIR)
    if f.lower().endswith(".png")
]

print("\n" + "=" * 70)
print("TASK 1 - SEGMENTATION")
print("=" * 70)

print("Segmentation images:", len(seg_images))
print("Segmentation masks :", len(masks))


# ============================================================
# MATCH IMAGE AND MASK
# ============================================================

seg_ids = set(
    os.path.splitext(f)[0]
    for f in seg_images
)

mask_ids = set(
    f.replace("_segmentation", "")
    for f in [
        os.path.splitext(x)[0]
        for x in masks
    ]
)

missing_masks = seg_ids - mask_ids
extra_masks = mask_ids - seg_ids

print("Images without masks:", len(missing_masks))
print("Masks without images:", len(extra_masks))


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print(f"Task 3 classification images : {len(images)}")
print(f"Task 3 labels                : {len(df)}")
print(f"Task 1 segmentation images   : {len(seg_images)}")
print(f"Task 1 segmentation masks    : {len(masks)}")

if "diagnosis" in df.columns:
    print("\nDiagnosis distribution:")
    print(df["diagnosis"].value_counts())

print("\nISIC2018 verification completed.")
print("=" * 70)