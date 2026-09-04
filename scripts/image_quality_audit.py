from pathlib import Path
from PIL import Image
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data" / "processed" / "binary_roi_splits"

FILES = [
    "train_binary_roi.csv",
    "validation_binary_roi.csv",
    "test_binary_roi.csv",
]

EXPECTED_SIZE = (224, 224)

all_results = []

print("=" * 75)
print("IMAGE QUALITY & STANDARDIZATION AUDIT")
print("=" * 75)

for filename in FILES:

    csv_path = DATA_DIR / filename
    df = pd.read_csv(csv_path)

    split = filename.replace("_binary_roi.csv", "")

    print("\n" + "=" * 75)
    print(split.upper())
    print("=" * 75)

    total = len(df)
    valid = 0
    corrupted = 0
    wrong_size = 0
    non_rgb = 0
    blank = 0
    missing = 0

    for _, row in df.iterrows():

        image_path = BASE_DIR / row["image_path"]

        if not image_path.exists():
            missing += 1
            continue

        try:
            with Image.open(image_path) as img:

                # Verify image
                img.verify()

            # Re-open after verify()
            with Image.open(image_path) as img:

                mode = img.mode
                size = img.size

                if mode != "RGB":
                    non_rgb += 1

                if size != EXPECTED_SIZE:
                    wrong_size += 1

                # Convert to RGB for pixel analysis
                rgb = img.convert("RGB")
                arr = np.asarray(rgb)

                # Detect near-blank images
                gray = arr.mean(axis=2)

                if gray.std() < 5:
                    blank += 1

                valid += 1

        except Exception:
            corrupted += 1

    print("Total images       :", total)
    print("Readable images    :", valid)
    print("Missing images     :", missing)
    print("Corrupted images   :", corrupted)
    print("Non-RGB images     :", non_rgb)
    print("Wrong-size images  :", wrong_size)
    print("Near-blank images  :", blank)

    all_results.append({
        "split": split,
        "total": total,
        "readable": valid,
        "missing": missing,
        "corrupted": corrupted,
        "non_rgb": non_rgb,
        "wrong_size": wrong_size,
        "near_blank": blank,
    })


# ---------------------------------------------------------
# FINAL SUMMARY
# ---------------------------------------------------------

results_df = pd.DataFrame(all_results)

print("\n" + "=" * 75)
print("FINAL AUDIT SUMMARY")
print("=" * 75)

print(results_df.to_string(index=False))

# Save audit report
output_path = (
    BASE_DIR
    / "results"
    / "image_quality_audit.csv"
)

output_path.parent.mkdir(
    parents=True,
    exist_ok=True
)

results_df.to_csv(
    output_path,
    index=False
)

print("\nAudit report saved:")
print(output_path)

# ---------------------------------------------------------
# FINAL STATUS
# ---------------------------------------------------------

problem_columns = [
    "missing",
    "corrupted",
    "non_rgb",
    "wrong_size",
    "near_blank",
]

problems = results_df[problem_columns].sum().sum()

print("\n" + "=" * 75)

if problems == 0:
    print("PASS: ALL ROI IMAGES ARE MODEL-READY")
else:
    print("WARNING: IMAGE QUALITY ISSUES DETECTED")
    print("Total issues:", problems)

print("=" * 75)