from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm


# =========================================================
# CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_CSV = (
    BASE_DIR
    / "data"
    / "processed"
    / "dataset_clean.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "images_standardized"
)

IMAGE_SIZE = (224, 224)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


print("=" * 70)
print("HAM10000 IMAGE STANDARDIZATION")
print("=" * 70)


# =========================================================
# LOAD DATASET
# =========================================================

df = pd.read_csv(INPUT_CSV)

print("\nTotal images:", len(df))


# =========================================================
# IMAGE PROCESSING
# =========================================================

processed = 0
failed = 0


for _, row in tqdm(
    df.iterrows(),
    total=len(df),
    desc="Processing images"
):

    image_path = Path(row["image_path"])

    if not image_path.is_absolute():
        image_path = BASE_DIR / image_path

    image = cv2.imread(
        str(image_path)
    )

    # -----------------------------------------------------
    # Check image
    # -----------------------------------------------------

    if image is None:
        failed += 1
        continue

    # -----------------------------------------------------
    # Convert BGR → RGB
    # -----------------------------------------------------

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # -----------------------------------------------------
    # Resize to 224 × 224
    # -----------------------------------------------------

    image = cv2.resize(
        image,
        IMAGE_SIZE,
        interpolation=cv2.INTER_AREA
    )

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    output_path = (
        OUTPUT_DIR
        / f"{row['image_id']}.jpg"
    )

    # RGB → BGR before OpenCV save
    image_bgr = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2BGR
    )

    cv2.imwrite(
        str(output_path),
        image_bgr
    )

    processed += 1


# =========================================================
# SUMMARY
# =========================================================

print("\n" + "=" * 70)
print("IMAGE STANDARDIZATION COMPLETED")
print("=" * 70)

print("\nProcessed images :", processed)
print("Failed images    :", failed)

print("\nOutput directory:")
print(OUTPUT_DIR)

print("\nImage size:")
print("224 × 224 × 3")

print("\nCompleted successfully.")