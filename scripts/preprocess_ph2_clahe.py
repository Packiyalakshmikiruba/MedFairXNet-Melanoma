from pathlib import Path

import cv2
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "PH2"
    / "splits"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "PH2"
    / "images_clahe"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FILES = [
    "train_ph2.csv",
    "validation_ph2.csv",
    "test_ph2.csv",
]


print("=" * 75)
print("MELONMA - PH2 CLAHE PREPROCESSING")
print("=" * 75)

print("\nInput:")
print(INPUT_DIR)

print("\nOutput:")
print(OUTPUT_DIR)


# ---------------------------------------------------------
# CLAHE FUNCTION
# ---------------------------------------------------------

def apply_clahe(image):

    # Convert BGR -> LAB
    lab = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB
    )

    l_channel, a_channel, b_channel = (
        cv2.split(lab)
    )

    # Controlled local contrast enhancement
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced_l = clahe.apply(
        l_channel
    )

    enhanced_lab = cv2.merge(
        [
            enhanced_l,
            a_channel,
            b_channel,
        ]
    )

    # LAB -> BGR
    enhanced = cv2.cvtColor(
        enhanced_lab,
        cv2.COLOR_LAB2BGR
    )

    return enhanced


# ---------------------------------------------------------
# PROCESS EACH SPLIT
# ---------------------------------------------------------

all_processed = []


for filename in FILES:

    csv_path = INPUT_DIR / filename

    if not csv_path.exists():

        raise FileNotFoundError(
            f"Split CSV not found:\n{csv_path}"
        )

    df = pd.read_csv(csv_path)

    print("\n" + "=" * 75)
    print(f"PROCESSING: {filename}")
    print("=" * 75)

    print(
        "Records:",
        len(df)
    )

    processed_count = 0
    failed_count = 0

    output_paths = []

    for _, row in df.iterrows():

        image_id = str(
            row["image_id"]
        )

        input_path = Path(
            row["image_path"]
        )

        if not input_path.exists():

            print(
                f"Missing image: {input_path}"
            )

            failed_count += 1
            output_paths.append("")
            continue

        image = cv2.imread(
            str(input_path)
        )

        if image is None:

            print(
                f"Unable to read: {input_path}"
            )

            failed_count += 1
            output_paths.append("")
            continue

        # Ensure 3-channel image
        if len(image.shape) != 3:

            print(
                f"Invalid image channels: {input_path}"
            )

            failed_count += 1
            output_paths.append("")
            continue

        # Apply CLAHE
        enhanced = apply_clahe(
            image
        )

        output_path = (
            OUTPUT_DIR
            / f"{image_id}.png"
        )

        success = cv2.imwrite(
            str(output_path),
            enhanced
        )

        if not success:

            print(
                f"Failed to save: {output_path}"
            )

            failed_count += 1
            output_paths.append("")
            continue

        output_paths.append(
            str(output_path)
        )

        processed_count += 1

    df["clahe_image_path"] = (
        output_paths
    )

    # Save updated metadata
    output_csv = (
        INPUT_DIR
        / filename.replace(
            ".csv",
            "_clahe.csv"
        )
    )

    df.to_csv(
        output_csv,
        index=False
    )

    all_processed.append(
        df
    )

    print(
        "\nProcessed:",
        processed_count
    )

    print(
        "Failed:",
        failed_count
    )

    print(
        "Saved metadata:",
        output_csv
    )


# ---------------------------------------------------------
# FINAL VALIDATION
# ---------------------------------------------------------

combined = pd.concat(
    all_processed,
    ignore_index=True
)

missing_output = (
    combined["clahe_image_path"]
    .apply(
        lambda x: (
            not x
            or not Path(x).exists()
        )
    )
    .sum()
)


duplicate_ids = (
    combined["image_id"]
    .duplicated()
    .sum()
)


print("\n" + "=" * 75)
print("FINAL PH2 CLAHE VALIDATION")
print("=" * 75)

print(
    "\nTotal records:",
    len(combined)
)

print(
    "CLAHE images:",
    len(
        combined[
            combined["clahe_image_path"] != ""
        ]
    )
)

print(
    "Missing CLAHE images:",
    missing_output
)

print(
    "Duplicate image IDs:",
    duplicate_ids
)


if missing_output > 0:

    raise RuntimeError(
        "Some PH2 CLAHE images are missing."
    )


if duplicate_ids > 0:

    raise RuntimeError(
        "Duplicate PH2 image IDs detected."
    )


print("\n" + "=" * 75)
print("PH2 CLAHE PREPROCESSING COMPLETED")
print("=" * 75)

print("\nStatus: PASS")

print("=" * 75)