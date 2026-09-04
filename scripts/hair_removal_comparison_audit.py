from pathlib import Path
import cv2
import numpy as np
import pandas as pd


# ============================================================
# MELONMA
# HAIR REMOVAL BEFORE / AFTER COMPARISON AUDIT
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "images_roi"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "hair_removed"
)

RESULTS_DIR = BASE_DIR / "results"

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
print("MELONMA - HAIR REMOVAL BEFORE / AFTER AUDIT")
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


print("\n" + "=" * 80)
print("FILE COUNT")
print("=" * 80)

print("Input images  :", len(input_files))
print("Output images :", len(output_files))

print("Missing outputs :", len(missing_outputs))
print("Extra outputs   :", len(extra_outputs))


# ============================================================
# BASIC FILE MAPPING CHECK
# ============================================================

if missing_outputs:

    print("\nMissing output files:")

    for name in missing_outputs[:50]:
        print("  ", name)

    if len(missing_outputs) > 50:
        print(
            f"  ... and {len(missing_outputs) - 50} more"
        )


if extra_outputs:

    print("\nExtra output files:")

    for name in extra_outputs[:50]:
        print("  ", name)

    if len(extra_outputs) > 50:
        print(
            f"  ... and {len(extra_outputs) - 50} more"
        )


# ============================================================
# IMAGE COMPARISON
# ============================================================

records = []

read_failures = 0
shape_mismatches = 0
unchanged_images = 0

for index, input_file in enumerate(
    input_files,
    start=1
):

    output_file = OUTPUT_DIR / input_file.name

    if not output_file.exists():
        continue

    original = cv2.imread(
        str(input_file)
    )

    processed = cv2.imread(
        str(output_file)
    )

    if original is None or processed is None:

        read_failures += 1

        records.append({
            "image_id": input_file.stem,
            "filename": input_file.name,
            "readable": False,
            "shape_match": False,
            "mean_absolute_difference": np.nan,
            "changed_pixel_percentage": np.nan,
            "status": "READ_FAILURE"
        })

        continue


    # --------------------------------------------------------
    # Shape check
    # --------------------------------------------------------

    shape_match = (
        original.shape == processed.shape
    )

    if not shape_match:

        shape_mismatches += 1


    # --------------------------------------------------------
    # Pixel difference
    # --------------------------------------------------------

    if shape_match:

        difference = cv2.absdiff(
            original,
            processed
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

        "mean_absolute_difference":
            mean_difference,

        "changed_pixel_percentage":
            changed_percentage,

        "status":
            "PASS"
            if shape_match
            else "SHAPE_MISMATCH"
    })


# ============================================================
# DATAFRAME
# ============================================================

results = pd.DataFrame(records)


# ============================================================
# SUMMARY STATISTICS
# ============================================================

print("\n" + "=" * 80)
print("IMAGE COMPARISON SUMMARY")
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
    "Unchanged images      :",
    unchanged_images
)


if len(results) > 0:

    valid_results = results[
        results["mean_absolute_difference"].notna()
    ]

    if len(valid_results) > 0:

        print(
            "\nMean pixel difference:"
        )

        print(
            "Minimum :",
            round(
                valid_results[
                    "mean_absolute_difference"
                ].min(),
                4
            )
        )

        print(
            "Maximum :",
            round(
                valid_results[
                    "mean_absolute_difference"
                ].max(),
                4
            )
        )

        print(
            "Mean    :",
            round(
                valid_results[
                    "mean_absolute_difference"
                ].mean(),
                4
            )
        )

        print(
            "Median  :",
            round(
                valid_results[
                    "mean_absolute_difference"
                ].median(),
                4
            )
        )


        print(
            "\nChanged pixel percentage:"
        )

        print(
            "Minimum :",
            round(
                valid_results[
                    "changed_pixel_percentage"
                ].min(),
                4
            ),
            "%"
        )

        print(
            "Maximum :",
            round(
                valid_results[
                    "changed_pixel_percentage"
                ].max(),
                4
            ),
            "%"
        )

        print(
            "Mean    :",
            round(
                valid_results[
                    "changed_pixel_percentage"
                ].mean(),
                4
            ),
            "%"
        )

        print(
            "Median  :",
            round(
                valid_results[
                    "changed_pixel_percentage"
                ].median(),
                4
            ),
            "%"
        )


# ============================================================
# SAVE DETAILED REPORT
# ============================================================

report_path = (
    RESULTS_DIR
    / "hair_removal_comparison_audit.csv"
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
):

    print("STATUS: PASS")

    print(
        "Hair-removal input/output mapping "
        "and image integrity are valid."
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


print("\nDetailed report saved:")
print(report_path)

print("=" * 80)