from pathlib import Path
import cv2
import numpy as np
import pandas as pd


# ============================================================
# MELONMA
# CLAHE OUTPUT INTEGRITY AUDIT
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
print("MELONMA - CLAHE OUTPUT INTEGRITY AUDIT")
print("=" * 80)

print("\nInput directory:")
print(INPUT_DIR)

print("\nOutput directory:")
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

output_files = sorted(
    [
        f for f in OUTPUT_DIR.iterdir()
        if f.is_file()
        and f.suffix.lower() in VALID_EXTENSIONS
    ]
)


input_names = {
    f.name for f in input_files
}

output_names = {
    f.name for f in output_files
}


missing_outputs = sorted(
    input_names - output_names
)

extra_outputs = sorted(
    output_names - input_names
)


# ============================================================
# FILE COUNT
# ============================================================

print("\n" + "=" * 80)
print("FILE COUNT")
print("=" * 80)

print("Input images       :", len(input_files))
print("CLAHE output images:", len(output_files))
print("Missing outputs    :", len(missing_outputs))
print("Extra outputs      :", len(extra_outputs))


# ============================================================
# COMPARISON
# ============================================================

records = []

read_failures = 0
shape_mismatches = 0
channel_mismatches = 0
unchanged_images = 0

for input_file in input_files:

    output_file = (
        OUTPUT_DIR
        / input_file.name
    )

    if not output_file.exists():
        continue

    original = cv2.imread(
        str(input_file),
        cv2.IMREAD_COLOR
    )

    enhanced = cv2.imread(
        str(output_file),
        cv2.IMREAD_COLOR
    )

    if original is None or enhanced is None:

        read_failures += 1

        records.append({
            "image_id": input_file.stem,
            "filename": input_file.name,
            "readable": False,
            "shape_match": False,
            "channels_match": False,
            "mean_absolute_difference": np.nan,
            "changed_pixel_percentage": np.nan,
            "status": "READ_FAILURE"
        })

        continue


    # --------------------------------------------------------
    # Shape
    # --------------------------------------------------------

    shape_match = (
        original.shape == enhanced.shape
    )

    if not shape_match:
        shape_mismatches += 1


    # --------------------------------------------------------
    # Channel
    # --------------------------------------------------------

    channels_match = (
        len(original.shape) == 3
        and len(enhanced.shape) == 3
        and original.shape[2] == enhanced.shape[2]
    )

    if not channels_match:
        channel_mismatches += 1


    # --------------------------------------------------------
    # Pixel difference
    # --------------------------------------------------------

    if shape_match:

        difference = cv2.absdiff(
            original,
            enhanced
        )

        mean_difference = float(
            difference.mean()
        )

        changed_pixels = np.any(
            difference > 0,
            axis=2
        )

        changed_percentage = (
            changed_pixels.mean() * 100
        )

        if changed_percentage == 0:
            unchanged_images += 1

    else:

        mean_difference = np.nan
        changed_percentage = np.nan


    records.append({

        "image_id":
            input_file.stem,

        "filename":
            input_file.name,

        "readable":
            True,

        "shape_match":
            shape_match,

        "channels_match":
            channels_match,

        "mean_absolute_difference":
            mean_difference,

        "changed_pixel_percentage":
            changed_percentage,

        "status":
            "PASS"
            if shape_match and channels_match
            else "CHECK"
    })


# ============================================================
# DATAFRAME
# ============================================================

results = pd.DataFrame(records)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("CLAHE COMPARISON SUMMARY")
print("=" * 80)

print(
    "Compared images       :",
    len(results)
)

print(
    "Read failures         :",
    read_failures
)

print(
    "Shape mismatches      :",
    shape_mismatches
)

print(
    "Channel mismatches    :",
    channel_mismatches
)

print(
    "Unchanged images      :",
    unchanged_images
)


# ============================================================
# PIXEL STATISTICS
# ============================================================

valid_results = results[
    results["mean_absolute_difference"].notna()
]


if len(valid_results) > 0:

    print("\n" + "=" * 80)
    print("PIXEL DIFFERENCE STATISTICS")
    print("=" * 80)

    mean_diff = (
        valid_results[
            "mean_absolute_difference"
        ]
    )

    changed_pct = (
        valid_results[
            "changed_pixel_percentage"
        ]
    )

    print(
        "Mean pixel difference   :",
        round(mean_diff.mean(), 4)
    )

    print(
        "Median pixel difference :",
        round(mean_diff.median(), 4)
    )

    print(
        "Minimum                 :",
        round(mean_diff.min(), 4)
    )

    print(
        "Maximum                 :",
        round(mean_diff.max(), 4)
    )

    print(
        "\nMean changed pixels     :",
        round(changed_pct.mean(), 4),
        "%"
    )

    print(
        "Median changed pixels   :",
        round(changed_pct.median(), 4),
        "%"
    )

    print(
        "Minimum changed pixels  :",
        round(changed_pct.min(), 4),
        "%"
    )

    print(
        "Maximum changed pixels  :",
        round(changed_pct.max(), 4),
        "%"
    )


# ============================================================
# SAVE REPORT
# ============================================================

report_path = (
    RESULTS_DIR
    / "clahe_output_audit.csv"
)

results.to_csv(
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
    len(input_files) == len(output_files)
    and len(missing_outputs) == 0
    and len(extra_outputs) == 0
    and read_failures == 0
    and shape_mismatches == 0
    and channel_mismatches == 0
):

    print("STATUS: PASS")

    print(
        "All CLAHE outputs are readable, "
        "correctly mapped, and structurally valid."
    )

else:

    print("STATUS: CHECK REQUIRED")

    if missing_outputs:
        print(
            "Missing outputs:",
            len(missing_outputs)
        )

    if extra_outputs:
        print(
            "Extra outputs:",
            len(extra_outputs)
        )

    if read_failures:
        print(
            "Read failures:",
            read_failures
        )

    if shape_mismatches:
        print(
            "Shape mismatches:",
            shape_mismatches
        )

    if channel_mismatches:
        print(
            "Channel mismatches:",
            channel_mismatches
        )


print("\nDetailed report saved:")
print(report_path)

print("=" * 80)