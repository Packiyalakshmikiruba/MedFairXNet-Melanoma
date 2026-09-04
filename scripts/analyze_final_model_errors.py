import os
import shutil
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    precision_recall_curve,
    auc
)

# ============================================================
# MELONMA - FINAL MODEL ERROR ANALYSIS
# EfficientNetV2 + PH2 + CLAHE
# ============================================================

PROJECT_ROOT = r"C:\Users\HP\Desktop\Melonma"

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "melanoma_efficientnetv2_ph2_clahe.keras"
)

TEST_CSV = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "PH2",
    "splits",
    "test_ph2_clahe.csv"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "final_model_error_analysis"
)

IMAGE_SIZE = (224, 224)

# Final model comparison used threshold 0.50
THRESHOLD = 0.50

BATCH_SIZE = 8

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Error image directories
TP_DIR = os.path.join(OUTPUT_DIR, "true_positive")
TN_DIR = os.path.join(OUTPUT_DIR, "true_negative")
FP_DIR = os.path.join(OUTPUT_DIR, "false_positive")
FN_DIR = os.path.join(OUTPUT_DIR, "false_negative")

for directory in [
    TP_DIR,
    TN_DIR,
    FP_DIR,
    FN_DIR
]:
    os.makedirs(directory, exist_ok=True)


# ============================================================
# HEADER
# ============================================================

print("=" * 100)
print("MELONMA - FINAL MODEL ERROR ANALYSIS")
print("=" * 100)

print("\nFinal model:")
print("EfficientNetV2 + PH2 + CLAHE")

print("\nModel:")
print(MODEL_PATH)

print("\nTest CSV:")
print(TEST_CSV)

print("\nThreshold:")
print(THRESHOLD)

print("\nOutput directory:")
print(OUTPUT_DIR)


# ============================================================
# CHECK FILES
# ============================================================

if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(
        f"\nModel not found:\n{MODEL_PATH}"
    )

if not os.path.isfile(TEST_CSV):
    raise FileNotFoundError(
        f"\nTest CSV not found:\n{TEST_CSV}"
    )


# ============================================================
# LOAD TEST DATA
# ============================================================

print("\n" + "=" * 100)
print("LOADING TEST DATA")
print("=" * 100)

df = pd.read_csv(TEST_CSV)

print(f"\nTest records: {len(df)}")

print("\nColumns:")
print(list(df.columns))


# ============================================================
# FIND IMAGE COLUMN
# ============================================================

if "clahe_image_path" in df.columns:
    image_column = "clahe_image_path"
elif "image_path" in df.columns:
    image_column = "image_path"
else:
    raise ValueError(
        "No clahe_image_path or image_path column found."
    )
if "binary_label" not in df.columns:
    raise ValueError(
        "binary_label column not found."
    )

print(f"\nImage column: {image_column}")
print("Label column: binary_label")


# ============================================================
# CHECK IMAGES
# ============================================================

print("\nChecking test images...")

missing = []

for path in df[image_column].astype(str):

    if not os.path.isfile(path):
        missing.append(path)

print(f"Missing images: {len(missing)}")

if missing:

    print("\nFirst missing images:")

    for path in missing[:10]:
        print(path)

    raise FileNotFoundError(
        "Some test images are missing."
    )


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\nTest class distribution:")

print(
    df["binary_label"].value_counts().sort_index()
)

print(
    f"\nNon-melanoma: "
    f"{int((df['binary_label'] == 0).sum())}"
)

print(
    f"Melanoma    : "
    f"{int((df['binary_label'] == 1).sum())}"
)


# ============================================================
# LOAD MODEL
# ============================================================

print("\n" + "=" * 100)
print("LOADING FINAL MODEL")
print("=" * 100)

model = keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("\nModel loaded successfully.")

print(
    f"Model output shape: {model.output_shape}"
)


# ============================================================
# IMAGE LOADING FUNCTION
# ============================================================

def load_image(path):

    image_bytes = tf.io.read_file(path)

    image = tf.image.decode_png(
        image_bytes,
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

    # Normalize to 0-1
    image = image / 255.0

    # ImageNet normalization
    mean = tf.constant(
        [0.485, 0.456, 0.406],
        dtype=tf.float32
    )

    std = tf.constant(
        [0.229, 0.224, 0.225],
        dtype=tf.float32
    )

    image = (
        image - mean
    ) / std

    return image

# ============================================================
# LOAD ALL IMAGES
# ============================================================

print("\n" + "=" * 100)
print("LOADING TEST IMAGES")
print("=" * 100)

images = []

for i, path in enumerate(
    df[image_column].astype(str)
):

    image = load_image(path)

    images.append(
        image.numpy()
    )

    if (
        (i + 1) % 10 == 0
        or i + 1 == len(df)
    ):

        print(
            f"Loaded: {i + 1}/{len(df)}"
        )


X_test = np.asarray(
    images,
    dtype=np.float32
)

y_true = df[
    "binary_label"
].astype(int).to_numpy()

print(
    f"\nImage array shape: {X_test.shape}"
)

print(
    f"Label array shape: {y_true.shape}"
)


# ============================================================
# MODEL PREDICTION
# ============================================================

print("\n" + "=" * 100)
print("RUNNING FINAL MODEL PREDICTION")
print("=" * 100)

predictions_raw = model.predict(
    X_test,
    batch_size=BATCH_SIZE,
    verbose=1
)

print(
    f"\nRaw prediction shape: "
    f"{predictions_raw.shape}"
)


# ============================================================
# EXTRACT PROBABILITY
# ============================================================

if predictions_raw.ndim == 2:

    if predictions_raw.shape[1] == 1:

        y_prob = predictions_raw[:, 0]

    elif predictions_raw.shape[1] == 2:

        y_prob = predictions_raw[:, 1]

    else:

        raise ValueError(
            "Unexpected model output shape."
        )

else:

    y_prob = predictions_raw.reshape(-1)


# Keep probabilities within valid range
y_prob = np.clip(
    y_prob,
    0.0,
    1.0
)


# ============================================================
# CLASSIFICATION
# ============================================================

y_pred = (
    y_prob >= THRESHOLD
).astype(int)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 100)
print("CONFUSION MATRIX")
print("=" * 100)

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1]
)

tn, fp, fn, tp = cm.ravel()

print("\n              Predicted")
print("              Non-Mel    Melanoma")
print(
    f"Actual Non-Mel   {tn:4d}       {fp:4d}"
)
print(
    f"Actual Melanoma  {fn:4d}       {tp:4d}"
)


# ============================================================
# METRICS
# ============================================================

accuracy = (
    (tp + tn) /
    (tp + tn + fp + fn)
)

precision = (
    tp / (tp + fp)
    if (tp + fp) > 0
    else 0
)

sensitivity = (
    tp / (tp + fn)
    if (tp + fn) > 0
    else 0
)

specificity = (
    tn / (tn + fp)
    if (tn + fp) > 0
    else 0
)

f1 = (
    2 * precision * sensitivity /
    (precision + sensitivity)
    if (precision + sensitivity) > 0
    else 0
)

roc_auc = roc_auc_score(
    y_true,
    y_prob
)

precision_curve, recall_curve, _ = (
    precision_recall_curve(
        y_true,
        y_prob
    )
)

pr_auc = auc(
    recall_curve,
    precision_curve
)


# ============================================================
# PRINT METRICS
# ============================================================

print("\n" + "=" * 100)
print("FINAL ERROR ANALYSIS METRICS")
print("=" * 100)

print(
    f"\nThreshold     : {THRESHOLD:.2f}"
)

print(
    f"Accuracy      : {accuracy:.4f}"
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
# ERROR COUNTS
# ============================================================

print("\n" + "=" * 100)
print("ERROR COUNTS")
print("=" * 100)

print(
    f"\nTrue Positives  : {tp}"
)

print(
    f"True Negatives  : {tn}"
)

print(
    f"False Positives : {fp}"
)

print(
    f"False Negatives : {fn}"
)


# ============================================================
# CREATE RESULT DATAFRAME
# ============================================================

results = []

for i in range(len(df)):

    true_label = int(
        y_true[i]
    )

    predicted_label = int(
        y_pred[i]
    )

    probability = float(
        y_prob[i]
    )

    if true_label == 1 and predicted_label == 1:

        error_type = "TP"

    elif true_label == 0 and predicted_label == 0:

        error_type = "TN"

    elif true_label == 0 and predicted_label == 1:

        error_type = "FP"

    elif true_label == 1 and predicted_label == 0:

        error_type = "FN"

    results.append({

        "image_id": (
            df.iloc[i]["image_id"]
            if "image_id" in df.columns
            else os.path.basename(
                str(df.iloc[i][image_column])
            )
        ),

        "image_path": str(
            df.iloc[i][image_column]
        ),

        "true_label": true_label,

        "true_class": (
            "Melanoma"
            if true_label == 1
            else "Non-melanoma"
        ),

        "prediction_probability": probability,

        "predicted_label": predicted_label,

        "predicted_class": (
            "Melanoma"
            if predicted_label == 1
            else "Non-melanoma"
        ),

        "threshold": THRESHOLD,

        "error_type": error_type

    })


results_df = pd.DataFrame(
    results
)


# ============================================================
# SAVE COMPLETE PREDICTIONS
# ============================================================

all_predictions_csv = os.path.join(
    OUTPUT_DIR,
    "final_model_predictions.csv"
)

results_df.to_csv(
    all_predictions_csv,
    index=False
)

print(
    f"\nAll predictions saved:\n"
    f"{all_predictions_csv}"
)


# ============================================================
# SAVE INDIVIDUAL ERROR CSVs
# ============================================================

tp_df = results_df[
    results_df["error_type"] == "TP"
].copy()

tn_df = results_df[
    results_df["error_type"] == "TN"
].copy()

fp_df = results_df[
    results_df["error_type"] == "FP"
].copy()

fn_df = results_df[
    results_df["error_type"] == "FN"
].copy()


tp_csv = os.path.join(
    OUTPUT_DIR,
    "true_positives.csv"
)

tn_csv = os.path.join(
    OUTPUT_DIR,
    "true_negatives.csv"
)

fp_csv = os.path.join(
    OUTPUT_DIR,
    "false_positives.csv"
)

fn_csv = os.path.join(
    OUTPUT_DIR,
    "false_negatives.csv"
)

tp_df.to_csv(
    tp_csv,
    index=False
)

tn_df.to_csv(
    tn_csv,
    index=False
)

fp_df.to_csv(
    fp_csv,
    index=False
)

fn_df.to_csv(
    fn_csv,
    index=False
)


# ============================================================
# SAVE METRICS CSV
# ============================================================

metrics_df = pd.DataFrame([{

    "model": "EfficientNetV2_PH2_CLAHE",

    "threshold": THRESHOLD,

    "accuracy": accuracy,

    "roc_auc": roc_auc,

    "pr_auc": pr_auc,

    "precision": precision,

    "sensitivity": sensitivity,

    "specificity": specificity,

    "f1_score": f1,

    "true_positive": tp,

    "true_negative": tn,

    "false_positive": fp,

    "false_negative": fn,

    "total_samples": len(y_true)

}])

metrics_csv = os.path.join(
    OUTPUT_DIR,
    "final_error_analysis_metrics.csv"
)

metrics_df.to_csv(
    metrics_csv,
    index=False
)


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

cm_df = pd.DataFrame(
    cm,
    index=[
        "Actual_NonMelanoma",
        "Actual_Melanoma"
    ],
    columns=[
        "Predicted_NonMelanoma",
        "Predicted_Melanoma"
    ]
)

cm_csv = os.path.join(
    OUTPUT_DIR,
    "confusion_matrix.csv"
)

cm_df.to_csv(
    cm_csv
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    y_true,
    y_pred,
    target_names=[
        "Non-melanoma",
        "Melanoma"
    ],
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(
    report
).transpose()

report_csv = os.path.join(
    OUTPUT_DIR,
    "classification_report.csv"
)

report_df.to_csv(
    report_csv
)


# ============================================================
# COPY ERROR IMAGES
# ============================================================

print("\n" + "=" * 100)
print("SAVING ERROR IMAGES")
print("=" * 100)


def copy_images(
    subset_df,
    destination,
    label
):

    count = 0

    for _, row in subset_df.iterrows():

        source = str(
            row["image_path"]
        )

        image_id = str(
            row["image_id"]
        )

        filename = os.path.basename(
            source
        )

        if not filename:
            filename = image_id

        destination_file = os.path.join(
            destination,
            filename
        )

        try:

            shutil.copy2(
                source,
                destination_file
            )

            count += 1

        except Exception as e:

            print(
                f"Could not copy "
                f"{filename}: {e}"
            )

    print(
        f"{label}: {count} images saved."
    )


copy_images(
    tp_df,
    TP_DIR,
    "True Positives"
)

copy_images(
    tn_df,
    TN_DIR,
    "True Negatives"
)

copy_images(
    fp_df,
    FP_DIR,
    "False Positives"
)

copy_images(
    fn_df,
    FN_DIR,
    "False Negatives"
)


# ============================================================
# CONFUSION MATRIX PLOT
# ============================================================

print("\n" + "=" * 100)
print("GENERATING CONFUSION MATRIX PLOT")
print("=" * 100)

plt.figure(
    figsize=(7, 6)
)

plt.imshow(
    cm,
    interpolation="nearest"
)

plt.title(
    "EfficientNetV2 PH2 CLAHE - Confusion Matrix"
)

plt.colorbar()

tick_marks = np.arange(2)

plt.xticks(
    tick_marks,
    ["Non-melanoma", "Melanoma"],
    rotation=30
)

plt.yticks(
    tick_marks,
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

plt.ylabel(
    "Actual Class"
)

plt.xlabel(
    "Predicted Class"
)

plt.tight_layout()

cm_plot = os.path.join(
    OUTPUT_DIR,
    "confusion_matrix.png"
)

plt.savefig(
    cm_plot,
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print(
    f"Confusion matrix plot saved:\n"
    f"{cm_plot}"
)


# ============================================================
# ROC CURVE
# ============================================================

from sklearn.metrics import roc_curve

fpr, tpr, _ = roc_curve(
    y_true,
    y_prob
)

plt.figure(
    figsize=(7, 6)
)

plt.plot(
    fpr,
    tpr,
    label=f"ROC-AUC = {roc_auc:.4f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "EfficientNetV2 PH2 CLAHE - ROC Curve"
)

plt.legend(
    loc="lower right"
)

plt.tight_layout()

roc_plot = os.path.join(
    OUTPUT_DIR,
    "roc_curve.png"
)

plt.savefig(
    roc_plot,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# PR CURVE
# ============================================================

plt.figure(
    figsize=(7, 6)
)

plt.plot(
    recall_curve,
    precision_curve,
    label=f"PR-AUC = {pr_auc:.4f}"
)

plt.xlabel(
    "Recall"
)

plt.ylabel(
    "Precision"
)

plt.title(
    "EfficientNetV2 PH2 CLAHE - Precision-Recall Curve"
)

plt.legend(
    loc="upper right"
)

plt.tight_layout()

pr_plot = os.path.join(
    OUTPUT_DIR,
    "precision_recall_curve.png"
)

plt.savefig(
    pr_plot,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# ERROR ANALYSIS SUMMARY
# ============================================================

print("\n" + "=" * 100)
print("ERROR ANALYSIS SUMMARY")
print("=" * 100)

print(
    f"\nTotal samples    : {len(results_df)}"
)

print(
    f"True Positives   : {tp}"
)

print(
    f"True Negatives   : {tn}"
)

print(
    f"False Positives  : {fp}"
)

print(
    f"False Negatives  : {fn}"
)

print(
    f"\nAccuracy         : {accuracy:.4f}"
)

print(
    f"ROC-AUC          : {roc_auc:.4f}"
)

print(
    f"PR-AUC           : {pr_auc:.4f}"
)

print(
    f"Precision        : {precision:.4f}"
)

print(
    f"Sensitivity      : {sensitivity:.4f}"
)

print(
    f"Specificity      : {specificity:.4f}"
)

print(
    f"F1-score         : {f1:.4f}"
)


# ============================================================
# FILES SAVED
# ============================================================

print("\n" + "=" * 100)
print("FILES SAVED")
print("=" * 100)

print(
    f"\nMetrics:\n{metrics_csv}"
)

print(
    f"\nConfusion matrix:\n{cm_csv}"
)

print(
    f"\nClassification report:\n{report_csv}"
)

print(
    f"\nAll predictions:\n{all_predictions_csv}"
)

print(
    f"\nTrue positives:\n{tp_csv}"
)

print(
    f"\nTrue negatives:\n{tn_csv}"
)

print(
    f"\nFalse positives:\n{fp_csv}"
)

print(
    f"\nFalse negatives:\n{fn_csv}"
)

print(
    f"\nOutput directory:\n{OUTPUT_DIR}"
)


# ============================================================
# FINAL STATUS
# ============================================================

print("\n" + "=" * 100)
print("STATUS: PASS")
print(
    "Final model error analysis completed successfully."
)
print("=" * 100)