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
    / "images_hair_removed"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "images_normalized"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# IMAGE NORMALIZATION
# =========================================================

def normalize_illumination(image):
    """
    Normalize illumination using LAB color space.

    L channel:
        - Represents brightness
        - CLAHE/equalization is applied carefully

    A/B channels:
        - Preserve color information
    """

    # Convert BGR -> LAB
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    l_channel, a_channel, b_channel = cv2.split(lab)

    # Estimate illumination using Gaussian blur
    illumination = cv2.GaussianBlur(
        l_channel,
        (0, 0),
        sigmaX=15
    )

    # Avoid division by zero
    illumination = np.maximum(
        illumination,
        1
    )

    # Illumination correction
    normalized_l = (
        l_channel.astype(np.float32)
        / illumination.astype(np.float32)
    )

    # Scale back to 0-255
    normalized_l = cv2.normalize(
        normalized_l,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    normalized_l = normalized_l.astype(
        np.uint8
    )

    # Merge LAB channels
    normalized_lab = cv2.merge(
        [
            normalized_l,
            a_channel,
            b_channel
        ]
    )

    # LAB -> BGR
    normalized_image = cv2.cvtColor(
        normalized_lab,
        cv2.COLOR_LAB2BGR
    )

    return normalized_image


# =========================================================
# MAIN PROCESS
# =========================================================

print("=" * 70)
print("HAM10000 ILLUMINATION NORMALIZATION")
print("=" * 70)

image_files = sorted(
    INPUT_DIR.glob("*.jpg")
)

print("\nInput images:", len(image_files))

processed = 0
failed = 0


for image_path in tqdm(
    image_files,
    desc="Normalizing images"
):

    try:

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            failed += 1
            continue

        normalized = normalize_illumination(
            image
        )

        output_path = (
            OUTPUT_DIR
            / image_path.name
        )

        cv2.imwrite(
            str(output_path),
            normalized
        )

        processed += 1

    except Exception as e:

        print(
            f"\nFailed: {image_path.name}"
        )

        print(e)

        failed += 1


# =========================================================
# FINAL SUMMARY
# =========================================================

print("\n" + "=" * 70)
print("ILLUMINATION NORMALIZATION COMPLETED")
print("=" * 70)

print("\nInput images     :", len(image_files))
print("Processed images :", processed)
print("Failed images    :", failed)

print("\nOutput directory:")
print(OUTPUT_DIR)

print("\nMethod:")
print("LAB color space + illumination correction")

print("\nCompleted successfully.")