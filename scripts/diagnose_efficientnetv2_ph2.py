from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

TEST_CSV = (
    BASE_DIR
    / "data"
    / "processed"
    / "PH2"
    / "splits"
    / "test_ph2_clahe.csv"
)

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "melanoma_efficientnetv2_ph2_clahe_v2.keras"
)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 8


# ============================================================
# LOAD TEST CSV
# ============================================================

test_df = pd.read_csv(TEST_CSV)


# ============================================================
# PATH
# ============================================================

def make_absolute_path(path):

    path = Path(str(path))

    if path.is_absolute():
        return str(path)

    return str(BASE_DIR / path)


test_df["clahe_image_path"] = (
    test_df["clahe_image_path"]
    .apply(make_absolute_path)
)


# ============================================================
# IMAGE LOADER
# IMPORTANT:
# SAME preprocessing as training
# ============================================================

def load_image(path):

    image = tf.io.read_file(path)

    image = tf.image.decode_png(
        image,
        channels=3
    )

    image = tf.image.resize(
        image,
        IMAGE_SIZE,
        method="bilinear"
    )

    image = tf.cast(
        image,
        tf.float32
    )

    return image

# ============================================================
# DATASET
# ============================================================

test_dataset = (
    tf.data.Dataset
    .from_tensor_slices(
        test_df["clahe_image_path"].values
    )
    .map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 80)
print("LOADING EFFICIENTNETV2 V2")
print("=" * 80)

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded.")
print("Output shape:", model.output_shape)


# ============================================================
# PREDICTIONS
# ============================================================

print()
print("=" * 80)
print("RUNNING PREDICTIONS")
print("=" * 80)

predictions = model.predict(
    test_dataset,
    verbose=1
)

predictions = np.asarray(
    predictions
).reshape(-1)


# ============================================================
# RESULTS
# ============================================================

result_df = test_df[
    [
        "image_id",
        "binary_label",
        "binary_diagnosis",
        "clahe_image_path"
    ]
].copy()


result_df["prediction_probability"] = predictions

result_df["predicted_label"] = (
    predictions >= 0.5
).astype(int)


result_df["predicted_class"] = np.where(
    result_df["predicted_label"] == 1,
    "Melanoma",
    "Non-melanoma"
)


# ============================================================
# SORT BY PROBABILITY
# ============================================================

result_df = result_df.sort_values(
    "prediction_probability",
    ascending=False
)


# ============================================================
# PRINT
# ============================================================

pd.set_option(
    "display.max_rows",
    100
)

pd.set_option(
    "display.max_columns",
    20
)

pd.set_option(
    "display.width",
    200
)


print()
print("=" * 80)
print("PH2 INDIVIDUAL PREDICTIONS")
print("=" * 80)

print(
    result_df[
        [
            "image_id",
            "binary_label",
            "binary_diagnosis",
            "prediction_probability",
            "predicted_label",
            "predicted_class"
        ]
    ].to_string(index=False)
)


# ============================================================
# SCORE STATISTICS
# ============================================================

print()
print("=" * 80)
print("PREDICTION SCORE STATISTICS")
print("=" * 80)

print()

print(
    "Minimum probability:",
    predictions.min()
)

print(
    "Maximum probability:",
    predictions.max()
)

print(
    "Mean probability:",
    predictions.mean()
)

print(
    "Median probability:",
    np.median(predictions)
)


# ============================================================
# CLASS-WISE SCORE DISTRIBUTION
# ============================================================

print()
print("=" * 80)
print("CLASS-WISE PROBABILITY")
print("=" * 80)

for label, name in [
    (0, "Non-melanoma"),
    (1, "Melanoma")
]:

    scores = predictions[
        test_df["binary_label"].values == label
    ]

    print()
    print(name)

    print("Count :", len(scores))
    print("Min   :", scores.min())
    print("Max   :", scores.max())
    print("Mean  :", scores.mean())
    print("Median:", np.median(scores))


# ============================================================
# SAVE DIAGNOSTIC CSV
# ============================================================

output_path = (
    BASE_DIR
    / "results"
    / "efficientnetv2_ph2_v2_individual_predictions.csv"
)

result_df.to_csv(
    output_path,
    index=False
)


print()
print("=" * 80)
print("DIAGNOSTIC COMPLETED")
print("=" * 80)

print()
print("Saved:")
print(output_path)