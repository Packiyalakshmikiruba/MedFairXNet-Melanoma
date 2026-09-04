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


# ============================================================
# CONFIGURATION
# ============================================================

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
    / "melanoma_densenet121_final_clahe.keras"
)

RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RESULT_PATH = (
    RESULTS_DIR
    / "densenet121_final_clahe_test_evaluation.csv"
)

# NEW: predictions CSV path (this is what was missing)
PREDICTIONS_PATH = (
    RESULTS_DIR
    / "densenet121_final_clahe_test_predictions.csv"
)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 8


# ============================================================
# HEADER
# ============================================================

print("=" * 75)
print("MELONMA - FINAL CLAHE DENSENET121 TEST EVALUATION")
print("=" * 75)

print("\nTest CSV:")
print(TEST_CSV)

print("\nModel:")
print(MODEL_PATH)


# ============================================================
# CHECK FILES
# ============================================================

if not TEST_CSV.exists():
    raise FileNotFoundError(
        f"Test CSV not found:\n{TEST_CSV}"
    )

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found:\n{MODEL_PATH}"
    )


# ============================================================
# LOAD TEST DATA
# ============================================================

test_df = pd.read_csv(TEST_CSV)

print("\nTest images:", len(test_df))


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = [
    "image_id",
    "image_path",
    "binary_label",
    "binary_diagnosis",
]

for column in required_columns:

    if column not in test_df.columns:
        raise ValueError(
            f"Required column missing: {column}"
        )


# ============================================================
# LABELS
# ============================================================

y_true = (
    test_df["binary_label"]
    .astype(int)
    .to_numpy()
)


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

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


# ============================================================
# IMAGE PATH
# ============================================================

def make_absolute_path(path):

    path = Path(str(path))

    if path.is_absolute():
        return str(path)

    return str(BASE_DIR / path)


test_df["image_path"] = (
    test_df["image_path"]
    .apply(make_absolute_path)
)


# ============================================================
# CHECK MISSING IMAGES
# ============================================================

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
        "Some test CLAHE images are missing."
    )


# ============================================================
# IMAGE LOADING
# ============================================================

def load_image(path):

    image = tf.io.read_file(path)

    image = tf.image.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    image = tf.image.resize(
        image,
        IMAGE_SIZE
    )

    image = tf.cast(
        image,
        tf.float32
    )

    # Same preprocessing used during training
    image = tf.keras.applications.densenet.preprocess_input(
        image
    )

    return image


# ============================================================
# CREATE TEST DATASET
# ============================================================

test_paths = test_df["image_path"].values

test_dataset = tf.data.Dataset.from_tensor_slices(
    test_paths
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


# ============================================================
# LOAD MODEL
# ============================================================

print("\n" + "=" * 75)
print("LOADING FINAL DENSENET121")
print("=" * 75)

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("\nModel loaded successfully.")


# ============================================================
# PREDICTION
# ============================================================

print("\n" + "=" * 75)
print("RUNNING FINAL TEST PREDICTION")
print("=" * 75)

probabilities = (
    model.predict(
        test_dataset,
        verbose=1
    )
    .reshape(-1)
)


print("\nPrediction count:", len(probabilities))


if len(probabilities) != len(y_true):

    raise ValueError(
        "Prediction count does not match test labels."
    )


# ============================================================
# DEFAULT THRESHOLD
# ============================================================

THRESHOLD = 0.50

y_pred = (
    probabilities >= THRESHOLD
).astype(int)


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

roc_auc = roc_auc_score(
    y_true,
    probabilities
)

pr_auc = average_precision_score(
    y_true,
    probabilities
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


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1]
)

tn, fp, fn, tp = cm.ravel()


specificity = (
    tn / (tn + fp)
    if (tn + fp) > 0
    else 0.0
)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 75)
print("FINAL CLAHE DENSENET121 TEST RESULTS")
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


# ============================================================
# CONFUSION MATRIX
# ============================================================

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


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 75)
print("CLASSIFICATION REPORT")
print("=" * 75)

report = classification_report(
    y_true,
    y_pred,
    labels=[0, 1],
    target_names=[
        "Non-melanoma",
        "Melanoma"
    ],
    zero_division=0
)

print(report)


# ============================================================
# SAVE RESULTS
# ============================================================

results = pd.DataFrame(
    [
        {
            "model": "DenseNet121",
            "dataset": "Final CLAHE Test",
            "test_images": len(test_df),
            "threshold": THRESHOLD,
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


# ============================================================
# NEW: SAVE PER-IMAGE PREDICTIONS
# (this is the piece that was missing -- needed for ROC/PR curves
#  and any future error analysis)
# ============================================================

image_ids = (
    test_df["image_id"].astype(str).to_numpy()
    if "image_id" in test_df.columns
    else np.arange(len(test_df))
)

predictions_df = pd.DataFrame({
    "image_id": image_ids,
    "true_label": y_true,
    "raw_prediction": probabilities,
    "predicted_label": y_pred,
    "true_class": ["Melanoma" if v == 1 else "Non-melanoma" for v in y_true],
    "predicted_class": ["Melanoma" if v == 1 else "Non-melanoma" for v in y_pred],
})

predictions_df.to_csv(
    PREDICTIONS_PATH,
    index=False
)

print("\nPredictions saved:")
print(PREDICTIONS_PATH)


# ============================================================
# FINAL STATUS
# ============================================================

print("\n" + "=" * 75)
print("FINAL TEST EVALUATION COMPLETED")
print("=" * 75)

print("\nResults saved:")
print(RESULT_PATH)

print("\n" + "=" * 75)
print("STATUS: PASS")
print(
    "Final CLAHE DenseNet121 test evaluation completed successfully."
)
print("=" * 75)