from pathlib import Path

import cv2
import numpy as np
import pandas as pd


# ============================================================
# MELONMA
# CLAHE STATISTICAL SANITY AUDIT
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "hair_removed"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "images_clahe"
)

RESULTS_DIR = (
    BASE_DIR
    / "results"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff"
}


# ============================================================
# HEADER
# ============================================================

print("=" * 80)
print("MELONMA - CLAHE STATISTICAL SANITY AUDIT")
print("=" * 80)

print("\nInput :")
print(INPUT_DIR)

print("\nOutput:")
print(OUTPUT_DIR)


# ============================================================
# FIND FILES
# ============================================================

input_files = sorted(
    [
        f for f in INPUT_DIR.iterdir()
        if f.is_file()
        and f.suffix.lower() in VALID_EXTENSIONS
    ]
)


# ============================================================
# STORAGE
# ============================================================

records = []

failed = 0

extreme_dark = 0
extreme_bright = 0
invalid_pixels = 0


# ============================================================
# PROCESS
# ============================================================

for input_file in input_files:

    output_file = (
        OUTPUT_DIR
        / input_file.name
    )

    if not output_file.exists():
        continue

    before = cv2.imread(
        str(input_file),
        cv2.IMREAD_COLOR
    )

    after = cv2.imread(
        str(output_file),
        cv2.IMREAD_COLOR
    )

    if before is None or after is None:
        failed += 1
        continue

    # --------------------------------------------------------
    # Convert to grayscale
    # --------------------------------------------------------

    before_gray = cv2.cvtColor(
        before,
        cv2.COLOR_BGR2GRAY
    )

    after_gray = cv2.cvtColor(
        after,
        cv2.COLOR_BGR2GRAY
    )


    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    before_mean = float(
        before_gray.mean()
    )

    after_mean = float(
        after_gray.mean()
    )

    before_std = float(
        before_gray.std()
    )

    after_std = float(
        after_gray.std()
    )


    # --------------------------------------------------------
    # Contrast improvement
    # --------------------------------------------------------

    contrast_change = (
        after_std - before_std
    )

    if before_std > 0:

        contrast_ratio = (
            after_std / before_std
        )

    else:

        contrast_ratio = np.nan


    # --------------------------------------------------------
    # Brightness change
    # --------------------------------------------------------

    brightness_change = (
        after_mean - before_mean
    )


    # --------------------------------------------------------
    # Pixel range
    # --------------------------------------------------------

    min_pixel = int(
        after.min()
    )

    max_pixel = int(
        after.max()
    )


    # --------------------------------------------------------
    # Extreme brightness
    # --------------------------------------------------------

    dark_pixels = (
        after_gray <= 2
    ).mean() * 100

    bright_pixels = (
        after_gray >= 253
    ).mean() * 100


    if dark_pixels > 20:
        extreme_dark += 1

    if bright_pixels > 20:
        extreme_bright += 1


    # --------------------------------------------------------
    # Invalid pixels
    # --------------------------------------------------------

    if not np.isfinite(after).all():
        invalid_pixels += 1


    # --------------------------------------------------------
    # Record
    # --------------------------------------------------------

    records.append({

        "image_id":
            input_file.stem,

        "filename":
            input_file.name,

        "before_mean":
            before_mean,

        "after_mean":
            after_mean,

        "before_std":
            before_std,

        "after_std":
            after_std,

        "contrast_change":
            contrast_change,

        "contrast_ratio":
            contrast_ratio,

        "brightness_change":
            brightness_change,

        "min_pixel":
            min_pixel,

        "max_pixel":
            max_pixel,

        "dark_pixel_percentage":
            dark_pixels,

        "bright_pixel_percentage":
            bright_pixels
    })


# ============================================================
# DATAFRAME
# ============================================================

df = pd.DataFrame(records)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("DATASET STATISTICS")
print("=" * 80)

print(
    "Images analysed :",
    len(df)
)

print(
    "Failed images   :",
    failed
)


if len(df) > 0:

    print("\n" + "-" * 80)
    print("BRIGHTNESS")
    print("-" * 80)

    print(
        "Before mean :",
        round(
            df["before_mean"].mean(),
            4
        )
    )

    print(
        "After mean  :",
        round(
            df["after_mean"].mean(),
            4
        )
    )

    print(
        "Mean change :",
        round(
            df["brightness_change"].mean(),
            4
        )
    )


    print("\n" + "-" * 80)
    print("CONTRAST")
    print("-" * 80)

    print(
        "Before std :",
        round(
            df["before_std"].mean(),
            4
        )
    )

    print(
        "After std  :",
        round(
            df["after_std"].mean(),
            4
        )
    )

    print(
        "Mean change:",
        round(
            df["contrast_change"].mean(),
            4
        )
    )

    print(
        "Mean ratio :",
        round(
            df["contrast_ratio"].mean(),
            4
        )
    )


    print("\n" + "-" * 80)
    print("PIXEL RANGE")
    print("-" * 80)

    print(
        "Minimum pixel:",
        df["min_pixel"].min()
    )

    print(
        "Maximum pixel:",
        df["max_pixel"].max()
    )


    print("\n" + "-" * 80)
    print("EXTREME PIXELS")
    print("-" * 80)

    print(
        "Images with >20% very dark pixels:",
        extreme_dark
    )

    print(
        "Images with >20% very bright pixels:",
        extreme_bright
    )

    print(
        "Images with invalid pixels:",
        invalid_pixels
    )


# ============================================================
# SAVE REPORT
# ============================================================

report_path = (
    RESULTS_DIR
    / "clahe_statistics_audit.csv"
)

df.to_csv(
    report_path,
    index=False
)


# ============================================================
# FINAL STATUS
# ============================================================

print("\n" + "=" * 80)
print("FINAL STATUS")
print("=" * 80)


if (
    failed == 0
    and invalid_pixels == 0
    and extreme_dark == 0
    and extreme_bright == 0
    and len(df) == len(input_files)
):

    print("STATUS: PASS")

    print(
        "CLAHE statistical sanity checks "
        "completed successfully."
    )

else:

    print("STATUS: CHECK REQUIRED")

    print(
        "Some statistical anomalies "
        "were detected."
    )


print("\nReport saved:")
print(report_path)

print("=" * 80)