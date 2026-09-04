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

FAILURE_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "segmentation_failures.txt"
)

IMAGE_SIZE = (224, 224)
PADDING_RATIO = 0.10


# =========================================================
# FALLBACK SEGMENTATION
# =========================================================

def fallback_segmentation(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Blur noise
    gray = cv2.GaussianBlur(
        gray,
        (7, 7),
        0
    )

    # Adaptive threshold
    mask = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        51,
        5
    )

    # Morphological cleanup
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (9, 9)
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    # Find contours
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    image_area = image.shape[0] * image.shape[1]

    valid = [
        c for c in contours
        if cv2.contourArea(c) > image_area * 0.005
    ]

    if not valid:
        return None

    contour = max(
        valid,
        key=cv2.contourArea
    )

    clean_mask = np.zeros(
        gray.shape,
        dtype=np.uint8
    )

    cv2.drawContours(
        clean_mask,
        [contour],
        -1,
        255,
        cv2.FILLED
    )

    return clean_mask


# =========================================================
# ROI EXTRACTION
# =========================================================

def extract_roi(image, mask):

    ys, xs = np.where(
        mask > 0
    )

    if len(xs) == 0:
        return None

    x1 = xs.min()
    x2 = xs.max()

    y1 = ys.min()
    y2 = ys.max()

    width = x2 - x1 + 1
    height = y2 - y1 + 1

    pad_x = int(
        width * PADDING_RATIO
    )

    pad_y = int(
        height * PADDING_RATIO
    )

    x1 = max(
        0,
        x1 - pad_x
    )

    y1 = max(
        0,
        y1 - pad_y
    )

    x2 = min(
        image.shape[1] - 1,
        x2 + pad_x
    )

    y2 = min(
        image.shape[0] - 1,
        y2 + pad_y
    )

    roi = image[
        y1:y2 + 1,
        x1:x2 + 1
    ]

    if roi.size == 0:
        return None

    roi = cv2.resize(
        roi,
        IMAGE_SIZE,
        interpolation=cv2.INTER_AREA
    )

    return roi


# =========================================================
# MAIN
# =========================================================

print("=" * 70)
print("FIXING FAILED LESION SEGMENTATION")
print("=" * 70)

failed_images = [
    x.strip()
    for x in FAILURE_FILE.read_text(
        encoding="utf-8"
    ).splitlines()
    if x.strip()
]

print("\nFailed images:", len(failed_images))

processed = 0
failed = 0


for name in tqdm(
    failed_images,
    desc="Recovering failed images"
):

    try:

        image_path = INPUT_DIR / name

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            failed += 1
            continue

        mask = fallback_segmentation(
            image
        )

        if mask is None:
            failed += 1
            continue

        roi = extract_roi(
            image,
            mask
        )

        if roi is None:
            failed += 1
            continue

        roi_path = ROI_DIR / name
        mask_path = MASK_DIR / name

        if not cv2.imwrite(
            str(roi_path),
            roi
        ):
            failed += 1
            continue

        if not cv2.imwrite(
            str(mask_path),
            mask
        ):
            failed += 1
            continue

        processed += 1

    except Exception as e:

        print(
            f"\nFailed: {name}"
        )
        print(e)

        failed += 1


# =========================================================
# FINAL SUMMARY
# =========================================================

print("\n" + "=" * 70)
print("FALLBACK SEGMENTATION COMPLETED")
print("=" * 70)

print("\nOriginal failed images :", len(failed_images))
print("Recovered images       :", processed)
print("Still failed           :", failed)

print("\nROI directory:")
print(ROI_DIR)

print("\nMask directory:")
print(MASK_DIR)