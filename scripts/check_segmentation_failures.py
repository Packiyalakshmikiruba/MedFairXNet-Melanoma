from pathlib import Path

import cv2
import numpy as np


# =========================================================
# CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "images_clahe"
)

ROI_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "images_roi"
)

MASK_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "lesion_masks"
)


# =========================================================
# CHECK
# =========================================================

print("=" * 70)
print("CHECKING LESION SEGMENTATION FAILURES")
print("=" * 70)

input_files = sorted(
    INPUT_DIR.glob("*.jpg")
)

failed_images = []

for image_path in input_files:

    roi_path = ROI_DIR / image_path.name
    mask_path = MASK_DIR / image_path.name

    # ROI or mask missing
    if not roi_path.exists() or not mask_path.exists():
        failed_images.append(
            image_path.name
        )


# =========================================================
# SUMMARY
# =========================================================

print("\nTotal input images :", len(input_files))
print("Successful images  :", len(input_files) - len(failed_images))
print("Failed images      :", len(failed_images))

print("\n" + "=" * 70)
print("FAILED IMAGE LIST")
print("=" * 70)

for name in failed_images:
    print(name)


# =========================================================
# SAVE FAILURE LIST
# =========================================================

failure_file = (
    BASE_DIR
    / "data"
    / "processed"
    / "segmentation_failures.txt"
)

with open(
    failure_file,
    "w",
    encoding="utf-8"
) as f:

    for name in failed_images:
        f.write(name + "\n")


print("\nFailure list saved at:")
print(failure_file)

print("\nCompleted.")