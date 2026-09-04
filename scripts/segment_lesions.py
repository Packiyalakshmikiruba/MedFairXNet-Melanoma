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

ROI_DIR.mkdir(parents=True, exist_ok=True)
MASK_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (224, 224)
PADDING_RATIO = 0.10


# =========================================================
# LESION SEGMENTATION
# =========================================================

def create_lesion_mask(image):

    # Convert BGR -> LAB
    lab = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB
    )

    # Use L channel for segmentation
    l_channel = lab[:, :, 0]

    # Slight blur to reduce noise
    blurred = cv2.GaussianBlur(
        l_channel,
        (5, 5),
        0
    )

    # Otsu threshold
    _, mask = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Morphological cleanup
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (7, 7)
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

    # Select largest meaningful contour
    image_area = image.shape[0] * image.shape[1]

    valid_contours = [
        contour
        for contour in contours
        if cv2.contourArea(contour) > image_area * 0.01
    ]

    if not valid_contours:
        return None

    largest_contour = max(
        valid_contours,
        key=cv2.contourArea
    )

    # Create clean binary mask
    clean_mask = np.zeros(
        mask.shape,
        dtype=np.uint8
    )

    cv2.drawContours(
        clean_mask,
        [largest_contour],
        -1,
        255,
        thickness=cv2.FILLED
    )

    return clean_mask


# =========================================================
# ROI EXTRACTION
# =========================================================

def extract_roi(image, mask):

    # Find mask coordinates
    ys, xs = np.where(mask > 0)

    if len(xs) == 0 or len(ys) == 0:
        return None

    x_min = xs.min()
    x_max = xs.max()

    y_min = ys.min()
    y_max = ys.max()

    # Bounding box dimensions
    width = x_max - x_min + 1
    height = y_max - y_min + 1

    # Contextual padding
    pad_x = int(width * PADDING_RATIO)
    pad_y = int(height * PADDING_RATIO)

    # Apply padding safely
    x_min = max(
        0,
        x_min - pad_x
    )

    y_min = max(
        0,
        y_min - pad_y
    )

    x_max = min(
        image.shape[1] - 1,
        x_max + pad_x
    )

    y_max = min(
        image.shape[0] - 1,
        y_max + pad_y
    )

    # Crop ROI
    roi = image[
        y_min:y_max + 1,
        x_min:x_max + 1
    ]

    if roi.size == 0:
        return None

    # Resize to model input size
    roi = cv2.resize(
        roi,
        IMAGE_SIZE,
        interpolation=cv2.INTER_AREA
    )

    return roi


# =========================================================
# MAIN PROCESS
# =========================================================

print("=" * 70)
print("HAM10000 LESION SEGMENTATION + ROI EXTRACTION")
print("=" * 70)

image_files = sorted(
    INPUT_DIR.glob("*.jpg")
)

print("\nInput images:", len(image_files))

processed = 0
failed = 0


for image_path in tqdm(
    image_files,
    desc="Segmenting lesions"
):

    try:

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            failed += 1
            continue

        # Create lesion mask
        mask = create_lesion_mask(
            image
        )

        if mask is None:
            failed += 1
            continue

        # Extract ROI
        roi = extract_roi(
            image,
            mask
        )

        if roi is None:
            failed += 1
            continue

        # Output paths
        roi_path = (
            ROI_DIR
            / image_path.name
        )

        mask_path = (
            MASK_DIR
            / image_path.name
        )

        # Save
        roi_saved = cv2.imwrite(
            str(roi_path),
            roi
        )

        mask_saved = cv2.imwrite(
            str(mask_path),
            mask
        )

        if not roi_saved or not mask_saved:
            failed += 1
            continue

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
print("LESION SEGMENTATION COMPLETED")
print("=" * 70)

print("\nInput images     :", len(image_files))
print("Processed images :", processed)
print("Failed images    :", failed)

print("\nROI output:")
print(ROI_DIR)

print("\nMask output:")
print(MASK_DIR)

print("\nROI size:")
print("224 × 224 × 3")

print("\nContextual padding:")
print("10%")

print("\nCompleted successfully.")