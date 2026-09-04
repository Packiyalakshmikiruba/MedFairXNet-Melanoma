from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
)


# =========================================================
# CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "melanoma_convnext_final_clahe.keras"
)

TEST_CSV = (
    BASE_DIR
    / "data"
    / "processed"
    / "final_binary_clahe_splits"
    / "test_final_clahe.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "results"
    / "convnext_final_error_analysis"
)

PREDICTIONS_PATH = (
    BASE_DIR
    / "results"
    / "convnext_final_clahe_test_predictions.csv"
)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 8
THRESHOLD = 0.50


# =========================================================
# IMAGENET NORMALIZATION
# =========================================================

MEAN = tf.constant(
    [0.485, 0.456, 0.406],
    dtype=tf.float32
)

STD = tf.constant(
    [0.229, 0.224, 0.225],
    dtype=tf.float32
)


# =========================================================
# DIRECTORIES
# =========================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

TP_DIR = OUTPUT_DIR / "true_positive"
TN_DIR = OUTPUT_DIR / "true_negative"
FP_DIR = OUTPUT_DIR / "false_positive"
FN_DIR = OUTPUT_DIR / "false_negative"

for directory in [
    TP_DIR,
    TN_DIR,
    FP_DIR,
    FN_DIR,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True
    )


# =========================================================
# HEADER
# =========================================================

print("=" * 90)
print("MELONMA - CONVNEXT FINAL ERROR ANALYSIS")
print("=" * 90)

print("\nModel:")
print(MODEL_PATH)

print("\nTest CSV:")
print(TEST_CSV)

print("\nThreshold:")
print(THRESHOLD)


# =========================================================
# CHECK FILES
# =========================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found:\n{MODEL_PATH}"
    )

if not TEST_CSV.exists():
    raise FileNotFoundError(
        f"Test CSV not found:\n{TEST_CSV}"
    )


# =========================================================
# LOAD TEST DATA
# =========================================================

print("\n" + "=" * 90)
print("LOADING TEST DATA")
print("=" * 90)

df = pd.read_csv(TEST_CSV)

print("\nTest records:", len(df))

if "image_path" not in df.columns:
    raise ValueError(
        "image_path column not found."
    )

if "binary_label" not in df.columns:
    raise ValueError(
        "binary_label column not found."
    )

print("\nClass distribution:")

print(
    df["binary_label"]
    .value_counts()
    .sort_index()
)


# =========================================================
# MAKE PATHS ABSOLUTE
# =========================================================

def make_absolute_path(path):

    path = Path(str(path))

    if path.is_absolute():
        return str(path)

    return str(BASE_DIR / path)


df["image_path"] = (
    df["image_path"]
    .apply(make_absolute_path)
)


# =========================================================
# CHECK IMAGES
# =========================================================

missing = df[
    ~df["image_path"]
    .apply(lambda x: Path(x).exists())
]

print(
    "\nMissing images:",
    len(missing)
)

if len(missing) > 0:

    print(
        missing["image_path"]
        .head(10)
        .to_string(index=False)
    )

    raise FileNotFoundError(
        "Missing test images."
    )


# =========================================================
# LOAD MODEL
# =========================================================

print("\n" + "=" * 90)
print("LOADING CONVNEXT MODEL")
print("=" * 90)

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("\nModel loaded successfully.")

print(
    "Input shape:",
    model.input_shape
)

print(
    "Output shape:",
    model.output_shape
)


# =========================================================
# IMAGE LOADER
# =========================================================

def load_image(path):

    image = tf.io.read_file(path)

    image = tf.image.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    image.set_shape(
        [None, None, 3]
    )

    image = tf.image.resize(
        image,
        IMAGE_SIZE,
        method=tf.image.ResizeMethod.BILINEAR
    )

    image = tf.cast(
        image,
        tf.float32
    )

    image = image / 255.0

    image = (
        image - MEAN
    ) / STD

    return image


# =========================================================
# BUILD TEST DATASET
# =========================================================

test_dataset = (
    tf.data.Dataset
    .from_tensor_slices(
        df["image_path"].values
    )
    .map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)


# =========================================================
# PREDICTION
# =========================================================

print("\n" + "=" * 90)
print("RUNNING CONVNEXT PREDICTIONS")
print("=" * 90)

raw_predictions = (
    model.predict(
        test_dataset,
        verbose=1
    )
    .reshape(-1)
)

y_true = (
    df["binary_label"]
    .astype(int)
    .to_numpy()
)

print(
    "\nPrediction count:",
    len(raw_predictions)
)

print(
    "Test count:",
    len(y_true)
)

if len(raw_predictions) != len(y_true):
    raise ValueError(
        "Prediction count does not match test count."
    )


# =========================================================
# SANITY CHECK
# =========================================================

print("\n" + "=" * 90)
print("PREDICTION SANITY CHECK")
print("=" * 90)

print(
    f"\nMinimum: {raw_predictions.min():.6f}"
)

print(
    f"Maximum: {raw_predictions.max():.6f}"
)

print(
    f"Mean:    {raw_predictions.mean():.6f}"
)

print(
    f"Median:  {np.median(raw_predictions):.6f}"
)


# =========================================================
# ROC / PR
# =========================================================

roc_auc = roc_auc_score(
    y_true,
    raw_predictions
)

pr_auc = average_precision_score(
    y_true,
    raw_predictions
)


# =========================================================
# BINARY PREDICTIONS
# =========================================================

y_pred = (
    raw_predictions >= THRESHOLD
).astype(int)


# =========================================================
# CONFUSION MATRIX
# =========================================================

tn, fp, fn, tp = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1]
).ravel()


accuracy = (
    (tp + tn)
    / len(y_true)
)

precision = precision_score(
    y_true,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    zero_division=0
)

specificity = (
    tn / (tn + fp)
    if (tn + fp) > 0
    else 0
)

f1 = f1_score(
    y_true,
    y_pred,
    zero_division=0
)


# =========================================================
# PRINT RESULTS
# =========================================================

print("\n" + "=" * 90)
print("CONVNEXT ERROR ANALYSIS RESULTS")
print("=" * 90)

print(
    f"\nThreshold   : {THRESHOLD:.2f}"
)

print(
    f"Accuracy    : {accuracy:.4f}"
)

print(
    f"ROC-AUC     : {roc_auc:.4f}"
)

print(
    f"PR-AUC      : {pr_auc:.4f}"
)

print(
    f"Precision   : {precision:.4f}"
)

print(
    f"Sensitivity : {recall:.4f}"
)

print(
    f"Specificity : {specificity:.4f}"
)

print(
    f"F1-score    : {f1:.4f}"
)

print("\nConfusion Matrix:")

print(
    f"TN = {tn}"
)

print(
    f"FP = {fp}"
)

print(
    f"FN = {fn}"
)

print(
    f"TP = {tp}"
)


# =========================================================
# SAVE ALL PREDICTIONS
# =========================================================

prediction_df = df.copy()

prediction_df[
    "prediction_probability"
] = raw_predictions

prediction_df[
    "predicted_label"
] = y_pred

prediction_df[
    "prediction_type"
] = np.select(
    [
        (y_true == 1) & (y_pred == 1),
        (y_true == 0) & (y_pred == 0),
        (y_true == 0) & (y_pred == 1),
        (y_true == 1) & (y_pred == 0),
    ],
    [
        "TP",
        "TN",
        "FP",
        "FN",
    ],
    default="UNKNOWN"
)

all_predictions_path = (
    OUTPUT_DIR
    / "convnext_predictions.csv"
)

prediction_df.to_csv(
    all_predictions_path,
    index=False
)


# =========================================================
# SAVE ERROR CSVs
# =========================================================

prediction_df[
    prediction_df["prediction_type"] == "TP"
].to_csv(
    OUTPUT_DIR / "true_positives.csv",
    index=False
)

prediction_df[
    prediction_df["prediction_type"] == "TN"
].to_csv(
    OUTPUT_DIR / "true_negatives.csv",
    index=False
)

prediction_df[
    prediction_df["prediction_type"] == "FP"
].to_csv(
    OUTPUT_DIR / "false_positives.csv",
    index=False
)

prediction_df[
    prediction_df["prediction_type"] == "FN"
].to_csv(
    OUTPUT_DIR / "false_negatives.csv",
    index=False
)


# =========================================================
# SAVE METRICS
# =========================================================

metrics_df = pd.DataFrame(
    [{
        "model": "ConvNeXtTiny",
        "dataset": "Final CLAHE Test",
        "threshold": THRESHOLD,
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "precision": precision,
        "sensitivity": recall,
        "specificity": specificity,
        "f1_score": f1,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
    }]
)

metrics_df.to_csv(
    OUTPUT_DIR / "error_analysis_metrics.csv",
    index=False
)


# =========================================================
# CLASSIFICATION REPORT
# =========================================================

report = classification_report(
    y_true,
    y_pred,
    target_names=[
        "Non-melanoma",
        "Melanoma"
    ],
    zero_division=0
)

print("\n" + "=" * 90)
print("CLASSIFICATION REPORT")
print("=" * 90)

print(report)

with open(
    OUTPUT_DIR / "classification_report.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write(report)


# =========================================================
# CONFUSION MATRIX PLOT
# =========================================================

cm = np.array([
    [tn, fp],
    [fn, tp]
])

plt.figure(
    figsize=(7, 6)
)

plt.imshow(cm)

plt.title(
    "ConvNeXt Final CLAHE - Confusion Matrix"
)

plt.xlabel(
    "Predicted Label"
)

plt.ylabel(
    "True Label"
)

plt.xticks(
    [0, 1],
    ["Non-melanoma", "Melanoma"]
)

plt.yticks(
    [0, 1],
    ["Non-melanoma", "Melanoma"]
)

for i in range(2):

    for j in range(2):

        plt.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center"
        )

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "confusion_matrix.png",
    dpi=300
)

plt.close()


# =========================================================
# FINAL STATUS
# =========================================================

print("\n" + "=" * 90)
print("ERROR ANALYSIS COMPLETED")
print("=" * 90)

print(
    "\nTP:",
    tp
)

print(
    "TN:",
    tn
)

print(
    "FP:",
    fp
)

print(
    "FN:",
    fn
)

print(
    "\nOutput directory:"
)

print(OUTPUT_DIR)

print("\nSTATUS: PASS")
print("ConvNeXt final error analysis completed.")
print("=" * 90)
