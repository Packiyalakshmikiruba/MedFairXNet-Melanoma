
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)

# ============================================================
# MELONMA - DENSENET121 THRESHOLDED TEST EVALUATION
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

OUTPUT_DIR = (
    BASE_DIR
    / "results"
    / "densenet121_thresholded_test"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 8

# Threshold selected ONLY from validation set
THRESHOLD = 0.27


# ============================================================
# HEADER
# ============================================================

print("=" * 80)
print("MELONMA - DENSENET121 THRESHOLDED TEST EVALUATION")
print("=" * 80)

print("\nTest CSV:")
print(TEST_CSV)

print("\nModel:")
print(MODEL_PATH)

print("\nValidation-selected threshold:", THRESHOLD)


# ============================================================
# VALIDATE FILES
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
# LOAD TEST CSV
# ============================================================

df = pd.read_csv(TEST_CSV)

print("\n" + "=" * 80)
print("TEST DATA")
print("=" * 80)

print("Test records:", len(df))

if "image_path" not in df.columns:
    raise ValueError(
        f"'image_path' column not found.\n"
        f"Available columns: {list(df.columns)}"
    )

if "binary_label" not in df.columns:
    raise ValueError(
        "'binary_label' column not found."
    )


# ============================================================
# RESOLVE IMAGE PATHS
# ============================================================

def resolve_image_path(value):

    path = Path(str(value))

    if path.is_absolute():
        return str(path.resolve())

    return str(
        (BASE_DIR / path).resolve()
    )


df["resolved_image_path"] = (
    df["image_path"]
    .apply(resolve_image_path)
)


# ============================================================
# CHECK MISSING IMAGES
# ============================================================

missing = [
    path
    for path in df["resolved_image_path"]
    if not Path(path).exists()
]

print("Missing test images:", len(missing))

if missing:

    print("\nFirst missing images:")

    for path in missing[:10]:
        print(path)

    raise FileNotFoundError(
        f"{len(missing)} test images are missing."
    )


# ============================================================
# LABELS
# ============================================================

y_true = (
    df["binary_label"]
    .astype(int)
    .to_numpy()
)

print("\nTest class distribution:")

print(
    df["binary_label"]
    .value_counts()
    .sort_index()
)

print(
    "\nNon-melanoma:",
    int((y_true == 0).sum())
)

print(
    "Melanoma    :",
    int((y_true == 1).sum())
)


# ============================================================
# IMAGE LOADER
# ============================================================

def load_image(path):

    image_bytes = tf.io.read_file(
        str(path)
    )

    image = tf.io.decode_image(
        image_bytes,
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
    ) / 255.0

    return image.numpy()


# ============================================================
# LOAD TEST IMAGES
# ============================================================

print("\n" + "=" * 80)
print("LOADING TEST IMAGES")
print("=" * 80)

images = []

paths = (
    df["resolved_image_path"]
    .tolist()
)

for index, path in enumerate(
    paths,
    start=1
):

    try:

        image = load_image(path)

        images.append(image)

    except Exception as error:

        raise RuntimeError(
            f"\nFailed to load:\n{path}\n"
            f"Error: {error}"
        ) from error

    if (
        index % 100 == 0
        or index == len(paths)
    ):

        print(
            f"Loaded: {index}/{len(paths)}"
        )


X_test = np.asarray(
    images,
    dtype=np.float32
)

print(
    "\nTest image array shape:",
    X_test.shape
)

print(
    "Test label array shape:",
    y_true.shape
)


# ============================================================
# LOAD MODEL
# ============================================================

print("\n" + "=" * 80)
print("LOADING DENSENET121")
print("=" * 80)

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("Model loaded successfully.")

print(
    "Model output shape:",
    model.output_shape
)


# ============================================================
# PREDICTION
# ============================================================

print("\n" + "=" * 80)
print("RUNNING TEST PREDICTION")
print("=" * 80)

raw_predictions = model.predict(
    X_test,
    batch_size=BATCH_SIZE,
    verbose=1
)

print(
    "\nRaw prediction shape:",
    raw_predictions.shape
)


# ============================================================
# EXTRACT MELANOMA PROBABILITY
# ============================================================

if (
    raw_predictions.ndim == 2
    and raw_predictions.shape[1] == 1
):

    probabilities = (
        raw_predictions[:, 0]
    )

elif (
    raw_predictions.ndim == 2
    and raw_predictions.shape[1] == 2
):

    probabilities = (
        raw_predictions[:, 1]
    )

elif raw_predictions.ndim == 1:

    probabilities = raw_predictions

else:

    raise ValueError(
        f"Unsupported prediction shape: "
        f"{raw_predictions.shape}"
    )


probabilities = np.asarray(
    probabilities,
    dtype=np.float64
)


# ============================================================
# APPLY VALIDATION THRESHOLD
# ============================================================

y_pred = (
    probabilities >= THRESHOLD
).astype(int)


print(
    "\nPrediction count:",
    len(y_pred)
)

if len(y_pred) != len(y_true):

    raise ValueError(
        "Prediction count does not match test labels."
    )


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred
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

roc_auc = roc_auc_score(
    y_true,
    probabilities
)

pr_auc = average_precision_score(
    y_true,
    probabilities
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

tn, fp, fn, tp = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1]
).ravel()


specificity = (
    tn / (tn + fp)
    if (tn + fp) > 0
    else 0.0
)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 80)
print("FINAL DENSENET121 THRESHOLDED TEST RESULTS")
print("=" * 80)

print(
    f"\nThreshold      : {THRESHOLD:.2f}"
)

print(
    f"Accuracy       : {accuracy:.4f}"
)

print(
    f"ROC-AUC        : {roc_auc:.4f}"
)

print(
    f"PR-AUC         : {pr_auc:.4f}"
)

print(
    f"Precision      : {precision:.4f}"
)

print(
    f"Sensitivity    : {sensitivity:.4f}"
)

print(
    f"Specificity    : {specificity:.4f}"
)

print(
    f"F1-score       : {f1:.4f}"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 80)
print("CONFUSION MATRIX")
print("=" * 80)

print(
    "\n              Predicted"
)

print(
    "              Non-Mel    Melanoma"
)

print(
    f"Actual Non-Mel     {tn:4d}       {fp:4d}"
)

print(
    f"Actual Melanoma    {fn:4d}       {tp:4d}"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 80)
print("CLASSIFICATION REPORT")
print("=" * 80)

report = classification_report(
    y_true,
    y_pred,
    target_names=[
        "Non-melanoma",
        "Melanoma"
    ],
    zero_division=0
)

print(report)


# ============================================================
# SAVE METRICS
# ============================================================

metrics_df = pd.DataFrame([
    {
        "model": "DenseNet121",
        "threshold": THRESHOLD,
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "precision": precision,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1_score": f1,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "test_records": len(y_true),
    }
])

metrics_path = (
    OUTPUT_DIR
    / "thresholded_test_metrics.csv"
)

metrics_df.to_csv(
    metrics_path,
    index=False
)


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

cm_df = pd.DataFrame(
    [
        [tn, fp],
        [fn, tp]
    ],
    index=[
        "Actual_NonMelanoma",
        "Actual_Melanoma"
    ],
    columns=[
        "Predicted_NonMelanoma",
        "Predicted_Melanoma"
    ]
)

cm_path = (
    OUTPUT_DIR
    / "confusion_matrix.csv"
)

cm_df.to_csv(
    cm_path
)


# ============================================================
# SAVE CLASSIFICATION REPORT
# ============================================================

report_dict = classification_report(
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
    report_dict
).transpose()

report_path = (
    OUTPUT_DIR
    / "classification_report.csv"
)

report_df.to_csv(
    report_path
)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

prediction_df = df[
    [
        "image_path",
        "binary_label"
    ]
].copy()

prediction_df[
    "melanoma_probability"
] = probabilities

prediction_df[
    "predicted_label"
] = y_pred

prediction_df[
    "threshold"
] = THRESHOLD

prediction_df[
    "predicted_diagnosis"
] = np.where(
    y_pred == 1,
    "Melanoma",
    "Non-melanoma"
)

prediction_path = (
    OUTPUT_DIR
    / "predictions.csv"
)

prediction_df.to_csv(
    prediction_path,
    index=False
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 80)
print("FILES SAVED")
print("=" * 80)

print(
    "\nMetrics:"
)

print(metrics_path)

print(
    "\nConfusion Matrix:"
)

print(cm_path)

print(
    "\nClassification Report:"
)

print(report_path)

print(
    "\nPredictions:"
)

print(prediction_path)

print("\n" + "=" * 80)
print("STATUS: PASS")
print(
    "DenseNet121 thresholded test evaluation completed successfully."
)
print("=" * 80)

