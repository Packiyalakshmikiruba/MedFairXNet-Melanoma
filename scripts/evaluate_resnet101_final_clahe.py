from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


# =========================================================
# CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

TEST_CSV = (
    BASE_DIR
    / "data"
    / "processed"
    / "final_binary_clahe_splits"
    / "test_final_clahe.csv"
)

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "melanoma_resnet101_final_clahe.keras"
)

RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RESULT_PATH = (
    RESULTS_DIR
    / "resnet101_final_clahe_test_evaluation.csv"
)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 8


# =========================================================
# HEADER
# =========================================================

print("=" * 75)
print("MELONMA - FINAL CLAHE RESNET101 TEST EVALUATION")
print("=" * 75)

print("\nTest CSV:")
print(TEST_CSV)

print("\nModel:")
print(MODEL_PATH)


# =========================================================
# CHECK FILES
# =========================================================

if not TEST_CSV.exists():
    raise FileNotFoundError(
        f"Test CSV not found:\n{TEST_CSV}"
    )

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found:\n{MODEL_PATH}"
    )


# =========================================================
# LOAD TEST DATA
# =========================================================

test_df = pd.read_csv(TEST_CSV)

print("\nTest images:", len(test_df))


required = [
    "image_path",
    "binary_label",
    "binary_diagnosis",
]

for col in required:

    if col not in test_df.columns:

        raise ValueError(
            f"Missing required column: {col}"
        )


# =========================================================
# MAKE IMAGE PATH ABSOLUTE
# =========================================================

def make_absolute_path(path):

    path = Path(str(path))

    if path.is_absolute():
        return str(path)

    return str(BASE_DIR / path)


test_df["image_path"] = (
    test_df["image_path"]
    .apply(make_absolute_path)
)


# =========================================================
# VERIFY TEST IMAGES
# =========================================================

missing = test_df[
    ~test_df["image_path"]
    .apply(lambda x: Path(x).exists())
]

print("\nMissing test images:", len(missing))


if len(missing) > 0:

    print(
        missing["image_path"]
        .head(10)
        .to_string(index=False)
    )

    raise FileNotFoundError(
        "Some test images are missing."
    )


# =========================================================
# TEST CLASS DISTRIBUTION
# =========================================================

print("\n" + "=" * 75)
print("TEST CLASS DISTRIBUTION")
print("=" * 75)

print(
    test_df["binary_diagnosis"]
    .value_counts()
)

print("\nBinary labels:")

print(
    test_df["binary_label"]
    .value_counts()
    .sort_index()
)


# =========================================================
# IMAGE LOADING
# =========================================================

def load_image(path):

    image = tf.io.read_file(path)

    image = tf.image.decode_jpeg(
        image,
        channels=3
    )

    image = tf.image.resize(
        image,
        IMAGE_SIZE
    )

    image = tf.cast(
        image,
        tf.float32
    )

    return image


# =========================================================
# CREATE TEST DATASET
# =========================================================

test_dataset = (
    tf.data.Dataset
    .from_tensor_slices(
        test_df["image_path"].values
    )
    .map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)


# =========================================================
# LOAD MODEL
# =========================================================

print("\n" + "=" * 75)
print("LOADING FINAL RESNET101")
print("=" * 75)

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("\nModel loaded successfully.")


# =========================================================
# MODEL OUTPUT INFORMATION
# =========================================================

print("\nModel output shape:")

print(model.output_shape)


# =========================================================
# TEST PREDICTION
# =========================================================

print("\n" + "=" * 75)
print("RUNNING FINAL TEST PREDICTION")
print("=" * 75)


raw_predictions = model.predict(
    test_dataset,
    verbose=1
)


raw_predictions = np.asarray(
    raw_predictions
)


print(
    "\nRaw prediction shape:",
    raw_predictions.shape
)


# =========================================================
# HANDLE BINARY OUTPUT FORMAT
# =========================================================
#
# Possible formats:
#
# (999, 1)
#     sigmoid output
#
# (999, 2)
#     softmax output
#
# =========================================================

if (
    raw_predictions.ndim == 2
    and raw_predictions.shape[1] == 2
):

    # Column 1 = melanoma probability

    predictions = (
        raw_predictions[:, 1]
    )


elif (
    raw_predictions.ndim == 2
    and raw_predictions.shape[1] == 1
):

    # Sigmoid melanoma probability

    predictions = (
        raw_predictions[:, 0]
    )


elif raw_predictions.ndim == 1:

    predictions = raw_predictions


else:

    raise ValueError(
        "Unexpected model prediction shape: "
        f"{raw_predictions.shape}"
    )


predictions = np.asarray(
    predictions
).reshape(-1)


print(
    "Final prediction count:",
    len(predictions)
)


# =========================================================
# FINAL LENGTH VALIDATION
# =========================================================

if len(predictions) != len(test_df):

    raise ValueError(
        f"\nPrediction count ({len(predictions)}) "
        f"does not match test images "
        f"({len(test_df)})."
    )


print(
    "\nPrediction count matches test images."
)


# =========================================================
# TRUE LABELS
# =========================================================

y_true = (
    test_df["binary_label"]
    .astype(int)
    .values
)


# =========================================================
# CONVERT PROBABILITY TO CLASS
# =========================================================

THRESHOLD = 0.5


y_pred = (
    predictions >= THRESHOLD
).astype(int)


# =========================================================
# METRICS
# =========================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)


roc_auc = roc_auc_score(
    y_true,
    predictions
)


pr_auc = average_precision_score(
    y_true,
    predictions
)


precision = precision_score(
    y_true,
    y_pred,
    zero_division=0
)


sensitivity = recall_score(
    y_true,
    y_pred,
    zero_division=0
)


f1 = f1_score(
    y_true,
    y_pred,
    zero_division=0
)


# =========================================================
# CONFUSION MATRIX
# =========================================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1]
)


tn, fp, fn, tp = cm.ravel()


# =========================================================
# SPECIFICITY
# =========================================================

specificity = (

    tn / (tn + fp)

    if (tn + fp) > 0

    else 0.0

)


# =========================================================
# RESULTS
# =========================================================

print("\n" + "=" * 75)
print("FINAL RESNET101 TEST RESULTS")
print("=" * 75)

print(
    f"\nAccuracy      : {accuracy:.4f}"
)

print(
    f"ROC-AUC       : {roc_auc:.4f}"
)

print(
    f"PR-AUC        : {pr_auc:.4f}"
)

print(
    f"Precision     : {precision:.4f}"
)

print(
    f"Sensitivity   : {sensitivity:.4f}"
)

print(
    f"Specificity   : {specificity:.4f}"
)

print(
    f"F1-score      : {f1:.4f}"
)


# =========================================================
# CONFUSION MATRIX
# =========================================================

print("\n" + "=" * 75)
print("CONFUSION MATRIX")
print("=" * 75)

print(
    "\n              Predicted"
)

print(
    "              Non-Mel    Melanoma"
)

print(
    f"Actual Non-Mel     {tn:4d}        {fp:4d}"
)

print(
    f"Actual Melanoma    {fn:4d}        {tp:4d}"
)


# =========================================================
# CLASSIFICATION REPORT
# =========================================================

print("\n" + "=" * 75)
print("CLASSIFICATION REPORT")
print("=" * 75)


print(
    classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=[
            "Non-melanoma",
            "Melanoma"
        ],
        zero_division=0
    )
)


# =========================================================
# SAVE RESULTS
# =========================================================

results = pd.DataFrame(
    [
        {
            "model": "ResNet101",
            "dataset": "Final CLAHE Test",
            "test_images": len(test_df),

            "accuracy": accuracy,

            "roc_auc": roc_auc,

            "pr_auc": pr_auc,

            "precision": precision,

            "sensitivity": sensitivity,

            "specificity": specificity,

            "f1_score": f1,

            "true_negative": tn,

            "false_positive": fp,

            "false_negative": fn,

            "true_positive": tp,
        }
    ]
)


results.to_csv(
    RESULT_PATH,
    index=False
)


# =========================================================
# COMPLETION
# =========================================================

print("\n" + "=" * 75)
print("FINAL TEST EVALUATION COMPLETED")
print("=" * 75)

print("\nResults saved:")

print(RESULT_PATH)

print("\n" + "=" * 75)
print("STATUS: PASS")
print(
    "ResNet101 final CLAHE test evaluation "
    "completed successfully."
)
print("=" * 75)