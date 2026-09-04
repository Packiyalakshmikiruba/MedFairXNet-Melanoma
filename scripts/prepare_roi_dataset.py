from pathlib import Path

import pandas as pd


# =========================================================
# CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

SPLIT_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "splits"
)

ROI_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "images_roi"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "roi_splits"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# PROCESS SPLIT
# =========================================================

def prepare_split(split_name):

    input_csv = SPLIT_DIR / f"{split_name}.csv"
    output_csv = OUTPUT_DIR / f"{split_name}_roi.csv"

    df = pd.read_csv(input_csv)

    print(f"\n{split_name.upper()} records:", len(df))

    roi_paths = []
    missing = []

    for _, row in df.iterrows():

        image_id = str(row["image_id"])

        roi_path = ROI_DIR / f"{image_id}.jpg"

        if roi_path.exists():
            roi_paths.append(
                str(
                    roi_path.relative_to(BASE_DIR)
                )
            )
        else:
            roi_paths.append(None)
            missing.append(image_id)

    df["original_image_path"] = df["image_path"]

    df["image_path"] = roi_paths

    # Remove rows without ROI
    df = df.dropna(
        subset=["image_path"]
    ).reset_index(drop=True)

    df.to_csv(
        output_csv,
        index=False
    )

    print("Missing ROI images:", len(missing))
    print("Final records:", len(df))
    print("Output:", output_csv)

    return df


# =========================================================
# MAIN
# =========================================================

print("=" * 70)
print("PREPARING ROI DATASET")
print("=" * 70)

train_df = prepare_split("train")
val_df = prepare_split("validation")
test_df = prepare_split("test")


# =========================================================
# FINAL SUMMARY
# =========================================================

print("\n" + "=" * 70)
print("ROI DATASET PREPARATION COMPLETED")
print("=" * 70)

print("\nTrain ROI records      :", len(train_df))
print("Validation ROI records :", len(val_df))
print("Test ROI records       :", len(test_df))

print("\nOutput directory:")
print(OUTPUT_DIR)

print("\nCompleted successfully.")