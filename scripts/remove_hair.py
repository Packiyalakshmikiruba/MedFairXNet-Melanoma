from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


# =========================================================
# CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "images_standardized"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "images_hair_removed"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


print("=" * 70)
print("HAM10000 DULLRAZOR-STYLE HAIR REMOVAL")
print("=" * 70)


# =========================================================
# HAIR REMOVAL FUNCTION
# =========================================================

def remove_hair(image):
    """
    DullRazor-style hair removal.

    Steps:
    1. Convert RGB image to grayscale
    2. Detect dark hair using blackhat morphology
    3. Threshold hair mask
    4. Dilate mask slightly
    5. Inpaint detected hair
    """

    # RGB → grayscale
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY
    )

    # -----------------------------------------------------
    # Blackhat morphological operation
    # -----------------------------------------------------

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (9, 9)
    )

    blackhat = cv2.morphologyEx(
        gray,
        cv2.MORPH_BLACKHAT,
        kernel
    )

    # -----------------------------------------------------
    # Detect dark hair
    # -----------------------------------------------------

    _, hair_mask = cv2.threshold(
        blackhat,
        10,
        255,
        cv2.THRESH_BINARY
    )

    # -----------------------------------------------------
    # Slightly enlarge mask
    # -----------------------------------------------------

    mask_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3)
    )

    hair_mask = cv2.dilate(
        hair_mask,
        mask_kernel,
        iterations=1
    )

    # -----------------------------------------------------
    # Inpaint hair
    # -----------------------------------------------------

    result = cv2.inpaint(
        image,
        hair_mask,
        3,
        cv2.INPAINT_TELEA
    )

    return result


# =========================================================
# FIND IMAGES
# =========================================================

image_files = sorted(
    INPUT_DIR.glob("*.jpg")
)

print("\nInput images:", len(image_files))

if len(image_files) == 0:
    raise FileNotFoundError(
        f"No JPG images found in: {INPUT_DIR}"
    )


# =========================================================
# PROCESS IMAGES
# =========================================================

processed = 0
failed = 0


for image_path in tqdm(
    image_files,
    desc="Removing hair"
):

    # Read image
    image_bgr = cv2.imread(
        str(image_path)
    )

    if image_bgr is None:
        failed += 1
        continue

    # BGR → RGB
    image_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB
    )

    # Hair removal
    result_rgb = remove_hair(
        image_rgb
    )

    # RGB → BGR
    result_bgr = cv2.cvtColor(
        result_rgb,
        cv2.COLOR_RGB2BGR
    )

    # Output path
    output_path = (
        OUTPUT_DIR
        / image_path.name
    )

    # Save
    success = cv2.imwrite(
        str(output_path),
        result_bgr
    )

    if success:
        processed += 1
    else:
        failed += 1


# =========================================================
# FINAL SUMMARY
# =========================================================

print("\n" + "=" * 70)
print("HAIR REMOVAL COMPLETED")
print("=" * 70)

print("\nInput images     :", len(image_files))
print("Processed images :", processed)
print("Failed images    :", failed)

print("\nOutput directory:")
print(OUTPUT_DIR)

print("\nMethod:")
print("DullRazor-style blackhat + inpainting")

print("\nCompleted successfully.")