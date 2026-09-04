import os
import cv2
import numpy as np
import pandas as pd

from tqdm import tqdm
from sklearn.model_selection import train_test_split


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = r"C:\Users\HP\Desktop\Melonma"

RAW_DIR = os.path.join(
    PROJECT_DIR,
    "data",
    "raw",
    "ISIC2018"
)

PROCESSED_DIR = os.path.join(
    PROJECT_DIR,
    "data",
    "processed",
    "ISIC2018"
)

CSV_PATH = os.path.join(
    RAW_DIR,
    "ISIC2018_Task3_Training_GroundTruth",
    "ISIC2018_Task3_Training_GroundTruth.csv"
)

IMAGE_DIR = os.path.join(
    RAW_DIR,
    "ISIC2018_Task3_Training_Input"
)


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = (224, 224)

RANDOM_STATE = 42

TEST_SIZE = 0.15
VAL_SIZE = 0.15


# ============================================================
# CREATE OUTPUT FOLDERS
# ============================================================

for split in ["train", "val", "test"]:

    for label in ["melanoma", "non_melanoma"]:

        folder = os.path.join(
            PROCESSED_DIR,
            split,
            label
        )

        os.makedirs(folder, exist_ok=True)


# ============================================================
# WEIGHTED MEDIAN FILTER
# ============================================================

def weighted_median_filter(image):
    """
    Fast weighted-median-like denoising using OpenCV.
    Uses a 3x3 neighborhood while preserving lesion boundaries.
    """

    # Split channels
    channels = cv2.split(image)

    processed_channels = []

    for channel in channels:

        # Median filtering
        filtered = cv2.medianBlur(channel, 3)

        processed_channels.append(filtered)

    # Merge channels
    output = cv2.merge(processed_channels)

    return output

# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image_path):

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(
            f"Unable to read image: {image_path}"
        )

    # OpenCV BGR → RGB
    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # Resize
    image = cv2.resize(
        image,
        IMAGE_SIZE,
        interpolation=cv2.INTER_AREA
    )

    # Weighted Median Filter
    image = weighted_median_filter(image)

    # Normalize to 0–1
    image = image.astype(
        np.float32
    ) / 255.0

    # Convert back to 0–255 for saving
    image = (
        image * 255
    ).astype(np.uint8)

    return image


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

print("=" * 70)
print("ISIC2018 PREPROCESSING")
print("=" * 70)

print("\nLoading ground truth...")

df = pd.read_csv(CSV_PATH)

print(f"Total records: {len(df)}")

required_columns = [
    "image",
    "MEL"
]

for column in required_columns:

    if column not in df.columns:

        raise ValueError(
            f"Missing required column: {column}"
        )


# ============================================================
# CREATE BINARY LABEL
# ============================================================

df["label"] = df["MEL"].apply(
    lambda x: (
        "melanoma"
        if x == 1
        else "non_melanoma"
    )
)


# ============================================================
# CHECK CLASS DISTRIBUTION
# ============================================================

print("\nOriginal class distribution:")

print(
    df["label"].value_counts()
)


# ============================================================
# TRAIN / VAL / TEST SPLIT
# ============================================================

print("\nCreating stratified split...")

train_df, temp_df = train_test_split(
    df,
    test_size=TEST_SIZE + VAL_SIZE,
    stratify=df["label"],
    random_state=RANDOM_STATE
)


# 15% validation + 15% test from total dataset
relative_test_size = (
    TEST_SIZE /
    (TEST_SIZE + VAL_SIZE)
)


val_df, test_df = train_test_split(
    temp_df,
    test_size=relative_test_size,
    stratify=temp_df["label"],
    random_state=RANDOM_STATE
)


print("\nSplit statistics:")

print(
    f"Training   : {len(train_df)}"
)

print(
    f"Validation : {len(val_df)}"
)

print(
    f"Testing    : {len(test_df)}"
)


# ============================================================
# PROCESS DATASET
# ============================================================

def process_split(dataframe, split_name):

    print("\n" + "=" * 60)

    print(
        f"Processing {split_name.upper()} dataset"
    )

    print("=" * 60)

    success = 0
    failed = 0

    for _, row in tqdm(
        dataframe.iterrows(),
        total=len(dataframe)
    ):

        image_id = row["image"]

        label = row["label"]

        image_path = os.path.join(
            IMAGE_DIR,
            image_id + ".jpg"
        )

        output_folder = os.path.join(
            PROCESSED_DIR,
            split_name,
            label
        )

        output_path = os.path.join(
            output_folder,
            image_id + ".jpg"
        )

        try:

            image = preprocess_image(
                image_path
            )

            # RGB → BGR for OpenCV saving
            image_bgr = cv2.cvtColor(
                image,
                cv2.COLOR_RGB2BGR
            )

            cv2.imwrite(
                output_path,
                image_bgr
            )

            success += 1

        except Exception as e:

            print(
                f"\nERROR: {image_id}"
            )

            print(e)

            failed += 1

    print(
        f"\nSuccessfully processed: {success}"
    )

    print(
        f"Failed: {failed}"
    )


# ============================================================
# PROCESS ALL SPLITS
# ============================================================

process_split(
    train_df,
    "train"
)

process_split(
    val_df,
    "val"
)

process_split(
    test_df,
    "test"
)


# ============================================================
# SAVE SPLIT INFORMATION
# ============================================================

split_df = pd.concat(
    [
        train_df.assign(split="train"),
        val_df.assign(split="val"),
        test_df.assign(split="test")
    ]
)

split_csv_path = os.path.join(
    PROCESSED_DIR,
    "dataset_split.csv"
)

split_df.to_csv(
    split_csv_path,
    index=False
)


# ============================================================
# DATASET STATISTICS
# ============================================================

statistics = []

for split_name, split_df_temp in [
    ("train", train_df),
    ("val", val_df),
    ("test", test_df)
]:

    melanoma_count = (
        split_df_temp["label"]
        .eq("melanoma")
        .sum()
    )

    non_melanoma_count = (
        split_df_temp["label"]
        .eq("non_melanoma")
        .sum()
    )

    statistics.append({

        "split": split_name,

        "total": len(split_df_temp),

        "melanoma": melanoma_count,

        "non_melanoma": non_melanoma_count

    })


statistics_df = pd.DataFrame(
    statistics
)

statistics_path = os.path.join(
    PROCESSED_DIR,
    "dataset_statistics.csv"
)

statistics_df.to_csv(
    statistics_path,
    index=False
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)

print("ISIC2018 PREPROCESSING COMPLETED")

print("=" * 70)

print(
    f"\nProcessed dataset:\n{PROCESSED_DIR}"
)

print(
    f"\nSplit information:\n{split_csv_path}"
)

print(
    f"\nStatistics:\n{statistics_path}"
)

print("\nFinal statistics:")

print(
    statistics_df.to_string(
        index=False
    )
)

print("\n" + "=" * 70)