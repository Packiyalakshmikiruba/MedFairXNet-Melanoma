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
    / "binary_roi_splits"
    / "test_binary_roi.csv"
)

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "melanoma_densenet121_clahe.keras"
)

RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

REPORT_PATH = (
    RESULTS_DIR
    / "densenet121_binary_test_evaluation.csv"
)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 8


# =========================================================
# HEADER
# =========================================================

print("=" * 75)
print("MELONMA - DENSENET121 BINARY TEST EVALUATION")
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
        f"\nTest CSV not found:\n{TEST_CSV}"
    )

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"\nModel not found:\n{MODEL_PATH}"
    )


# =========================================================
# LOAD TEST DATA
# =========================================================

test_df = pd.read_csv(TEST_CSV)

print("\nTest images:", len(test_df))


# =========================================================
# CHECK REQUIRED COLUMNS
# =========================================================

required_columns = [
    "image_path",
    "binary_label",
]

missing_columns = [
    column
    for column in required_columns
    if column not in test_df.columns
]

if missing_columns:
    raise ValueError(
        f"\nMissing columns: {missing_columns}"
    )


# =========================================================
# VERIFY LABELS
# =========================================================

test_df["binary_label"] = (
    test_df["binary_label"]
    .astype(int)
)

unique_labels = sorted(
    test_df["binary_label"].unique()
)

print("\nLabels found:", unique_labels)

if not set(unique_labels).issubset({0, 1}):
    raise ValueError(
        "Binary labels must contain only 0 and 1."
    )


# =========================================================
# IMAGE PATH
# =========================================================

def make_absolute_path(path):

    path = Path(path)

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

missing_images = test_df[
    ~test_df["image_path"]
    .apply(lambda x: Path(x).exists())
]

print(
    "\nMissing test images:",
    len(missing_images)
)

if len(missing_images) > 0:

    print(
        missing_images[
            "image_path"
        ]
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
    test_df["binary_label"]
    .value_counts()
    .sort_index()
)

if "binary_diagnosis" in test_df.columns:

    print("\nDiagnosis distribution:")

    print(
        test_df["binary_diagnosis"]
        .value_counts()
    )


# =========================================================
# IMAGE LOADING
# =========================================================

def load_image(path, label):

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

    return image, label


# =========================================================
# DATASET
# =========================================================

test_paths = (
    test_df["image_path"]
    .values
)

test_labels = (
    test_df["binary_label"]
    .values
)


test_dataset = tf.data.Dataset.from_tensor_slices(
    (
        test_paths,
        test_labels
    )
)


test_dataset = test_dataset.map(
    load_image,
    num_parallel_calls=tf.data.AUTOTUNE
)


test_dataset = test_dataset.batch(
    BATCH_SIZE
)


test_dataset = test_dataset.prefetch(
    tf.data.AUTOTUNE
)


# =========================================================
# LOAD MODEL
# =========================================================

print("\n" + "=" * 75)
print("LOADING TRAINED DENSENET121")
print("=" * 75)

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("\nModel loaded successfully.")


# =========================================================
# PREDICTION
# =========================================================

print("\n" + "=" * 75)
print("RUNNING TEST PREDICTION")
print("=" * 75)

probabilities = model.predict(
    test_dataset,
    verbose=1
)


# =========================================================
# HANDLE OUTPUT SHAPE
# =========================================================

probabilities = np.asarray(
    probabilities
)

print(
    "\nPrediction shape:",
    probabilities.shape
)


# ---------------------------------------------------------
# Binary output handling
# ---------------------------------------------------------

if probabilities.ndim == 2 and probabilities.shape[1] == 1:

    melanoma_probability = (
        probabilities[:, 0]
    )

elif probabilities.ndim == 2 and probabilities.shape[1] == 2:

    melanoma_probability = (
        probabilities[:, 1]
    )

else:

    raise ValueError(
        "Unexpected model output shape."
    )


# =========================================================
# CLASS PREDICTION
# =========================================================

predicted_labels = (
    melanoma_probability >= 0.5
).astype(int)


true_labels = (
    test_labels.astype(int)
)


# =========================================================
# METRICS
# =========================================================

accuracy = accuracy_score(
    true_labels,
    predicted_labels
)

precision = precision_score(
    true_labels,
    predicted_labels,
    zero_division=0
)

recall = recall_score(
    true_labels,
    predicted_labels,
    zero_division=0
)

f1 = f1_score(
    true_labels,
    predicted_labels,
    zero_division=0
)

roc_auc = roc_auc_score(
    true_labels,
    melanoma_probability
)

pr_auc = average_precision_score(
    true_labels,
    melanoma_probability
)


# =========================================================
# CONFUSION MATRIX
# =========================================================

cm = confusion_matrix(
    true_labels,
    predicted_labels,
    labels=[0, 1]
)

tn, fp, fn, tp = cm.ravel()


# =========================================================
# SENSITIVITY / SPECIFICITY
# =========================================================

sensitivity = (
    tp / (tp + fn)
    if (tp + fn) > 0
    else 0.0
)

specificity = (
    tn / (tn + fp)
    if (tn + fp) > 0
    else 0.0
)


# =========================================================
# PRINT RESULTS
# =========================================================

print("\n" + "=" * 75)
print("DENSENET121 TEST RESULTS")
print("=" * 75)

print(
    f"\nAccuracy     : {accuracy:.4f}"
)

print(
    f"ROC-AUC      : {roc_auc:.4f}"
)

print(
    f"PR-AUC       : {pr_auc:.4f}"
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
    f"Actual Non-Mel   {tn:5d}      {fp:5d}"
)

print(
    f"Actual Melanoma  {fn:5d}      {tp:5d}"
)


# =========================================================
# CLASSIFICATION REPORT
# =========================================================

print("\n" + "=" * 75)
print("CLASSIFICATION REPORT")
print("=" * 75)

print(
    classification_report(
        true_labels,
        predicted_labels,
        target_names=[
            "Non-melanoma",
            "Melanoma"
        ],
        digits=4,
        zero_division=0
    )
)


# =========================================================
# SAVE REPORT
# =========================================================

results = pd.DataFrame([
    {
        "model": "DenseNet121",
        "dataset": "HAM10000 Binary Test",
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
])


results.to_csv(
    REPORT_PATH,
    index=False
)


# =========================================================
# FINAL STATUS
# =========================================================

print("\n" + "=" * 75)
print("TEST EVALUATION COMPLETED")
print("=" * 75)

print("\nResults saved:")
print(REPORT_PATH)

print("\nStatus:")
print("PASS - DenseNet121 test evaluation completed.")

print("=" * 75)