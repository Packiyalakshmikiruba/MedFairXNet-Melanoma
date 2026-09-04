from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
)


# ============================================================
# MELANOMA - FINAL MODEL VISUALIZATION
# EfficientNetV2 + PH2 + CLAHE
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\HP\Desktop\Melonma"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "melanoma_efficientnetv2_ph2_clahe.keras"
)

TEST_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "PH2"
    / "splits"
    / "test_ph2_clahe.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "final_model_visualizations"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

IMAGE_SIZE = (224, 224)

# IMPORTANT:
# Model output is sigmoid probability.
# 0.50 = default binary classification threshold.
THRESHOLD = 0.50


# ============================================================
# HEADER
# ============================================================

print()
print("=" * 100)
print("MELANOMA - FINAL MODEL VISUALIZATION")
print("=" * 100)

print()
print("Final model:")
print("EfficientNetV2 + PH2 + CLAHE")

print()
print("Project root:")
print(PROJECT_ROOT)

print()
print("Model:")
print(MODEL_PATH)

print()
print("Test CSV:")
print(TEST_CSV)

print()
print("Output directory:")
print(OUTPUT_DIR)

print()
print("Image size:")
print(IMAGE_SIZE)

print()
print("Classification threshold:")
print(THRESHOLD)


# ============================================================
# VALIDATE FILES
# ============================================================

print()
print("=" * 100)
print("VALIDATING INPUT FILES")
print("=" * 100)

if not PROJECT_ROOT.exists():

    raise FileNotFoundError(
        f"Project root not found:\n{PROJECT_ROOT}"
    )

if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"Model not found:\n{MODEL_PATH}"
    )

if not TEST_CSV.exists():

    raise FileNotFoundError(
        f"Test CSV not found:\n{TEST_CSV}"
    )

print()
print("Project root: OK")
print("Model file  : OK")
print("Test CSV    : OK")


# ============================================================
# LOAD TEST DATA
# ============================================================

print()
print("=" * 100)
print("LOADING TEST DATA")
print("=" * 100)

df = pd.read_csv(
    TEST_CSV
)

print()
print(
    f"Test records: {len(df)}"
)


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = [
    "image_id",
    "binary_label",
    "image_path",
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

IMAGE_COLUMN = "image_path"
LABEL_COLUMN = "binary_label"

print()
print("Required columns found.")

print()
print("Image column:")
print(IMAGE_COLUMN)

print()
print("Label column:")
print(LABEL_COLUMN)


# ============================================================
# CLEAN / VALIDATE LABELS
# ============================================================

print()
print("=" * 100)
print("VALIDATING LABELS")
print("=" * 100)

df[LABEL_COLUMN] = pd.to_numeric(
    df[LABEL_COLUMN],
    errors="coerce"
)

if df[LABEL_COLUMN].isna().any():

    bad_rows = df[
        df[LABEL_COLUMN].isna()
    ]

    print(
        bad_rows.head(10).to_string()
    )

    raise ValueError(
        "Invalid binary labels found in test CSV."
    )

df[LABEL_COLUMN] = (
    df[LABEL_COLUMN]
    .astype(int)
)

invalid_labels = sorted(
    set(df[LABEL_COLUMN].unique())
    - {0, 1}
)

if invalid_labels:

    raise ValueError(
        f"Invalid labels found: {invalid_labels}. "
        "Expected only 0 and 1."
    )

print()
print("Labels are valid.")
print("Allowed labels: 0 = Non-melanoma, 1 = Melanoma")


# ============================================================
# CHECK IMAGE PATHS
# ============================================================

print()
print("=" * 100)
print("CHECKING TEST IMAGE PATHS")
print("=" * 100)

missing_images = []

for _, row in df.iterrows():

    image_path = Path(
        str(row[IMAGE_COLUMN])
    )

    # --------------------------------------------------------
    # HANDLE RELATIVE PATHS
    # --------------------------------------------------------

    if not image_path.is_absolute():

        image_path = (
            PROJECT_ROOT
            / image_path
        )

    if not image_path.exists():

        missing_images.append(
            {
                "image_id": row["image_id"],
                "image_path": str(image_path),
            }
        )


print()
print(
    f"Missing images: {len(missing_images)}"
)

if missing_images:

    missing_df = pd.DataFrame(
        missing_images
    )

    missing_path = (
        OUTPUT_DIR
        / "missing_test_images.csv"
    )

    missing_df.to_csv(
        missing_path,
        index=False
    )

    print()
    print(
        "Missing-image report saved:"
    )

    print(
        missing_path
    )

    raise FileNotFoundError(
        "Missing test images detected."
    )

print()
print("All test images exist.")


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print()
print("=" * 100)
print("TEST CLASS DISTRIBUTION")
print("=" * 100)

class_counts = (
    df[LABEL_COLUMN]
    .value_counts()
    .sort_index()
)

non_melanoma_count = int(
    class_counts.get(0, 0)
)

melanoma_count = int(
    class_counts.get(1, 0)
)

print()
print(
    "Non-melanoma:",
    non_melanoma_count
)

print(
    "Melanoma    :",
    melanoma_count
)

print()
print(
    "Total       :",
    len(df)
)


# ============================================================
# IMAGENET NORMALIZATION
# ============================================================

print()
print("=" * 100)
print("SETTING IMAGENET NORMALIZATION")
print("=" * 100)

IMAGENET_MEAN = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32
)

IMAGENET_STD = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32
)

print()
print(
    "Mean:",
    IMAGENET_MEAN
)

print(
    "Std :",
    IMAGENET_STD
)


# ============================================================
# LOAD TEST IMAGES
# ============================================================

print()
print("=" * 100)
print("LOADING TEST IMAGES")
print("=" * 100)

images = []
labels = []
image_ids = []
resolved_image_paths = []


for index, row in df.iterrows():

    # --------------------------------------------------------
    # IMAGE PATH
    # --------------------------------------------------------

    image_path = Path(
        str(row[IMAGE_COLUMN])
    )

    if not image_path.is_absolute():

        image_path = (
            PROJECT_ROOT
            / image_path
        )

    image_path = image_path.resolve()

    # --------------------------------------------------------
    # LOAD IMAGE
    # --------------------------------------------------------

    image = tf.keras.utils.load_img(
        image_path,
        target_size=IMAGE_SIZE,
        color_mode="rgb"
    )

    # --------------------------------------------------------
    # IMAGE -> NUMPY
    # --------------------------------------------------------

    image_array = (
        tf.keras.utils.img_to_array(
            image
        )
    )

    # --------------------------------------------------------
    # FLOAT32
    # --------------------------------------------------------

    image_array = image_array.astype(
        np.float32
    )

    # --------------------------------------------------------
    # PIXEL SCALING
    # 0 - 255 -> 0 - 1
    # --------------------------------------------------------

    image_array = (
        image_array / 255.0
    )

    # --------------------------------------------------------
    # IMAGENET NORMALIZATION
    # --------------------------------------------------------

    image_array = (
        image_array - IMAGENET_MEAN
    ) / IMAGENET_STD

    # --------------------------------------------------------
    # STORE IMAGE
    # --------------------------------------------------------

    images.append(
        image_array
    )

    # --------------------------------------------------------
    # STORE LABEL
    # --------------------------------------------------------

    labels.append(
        int(row[LABEL_COLUMN])
    )

    # --------------------------------------------------------
    # STORE IMAGE ID
    # --------------------------------------------------------

    image_ids.append(
        str(row["image_id"])
    )

    # --------------------------------------------------------
    # STORE RESOLVED PATH
    # --------------------------------------------------------

    resolved_image_paths.append(
        str(image_path)
    )

    # --------------------------------------------------------
    # PROGRESS
    # --------------------------------------------------------

    if (
        (index + 1) % 10 == 0
        or index + 1 == len(df)
    ):

        print(
            f"Loaded: "
            f"{index + 1}/{len(df)}"
        )


# ============================================================
# CREATE TEST ARRAY
# ============================================================

X_test = np.asarray(
    images,
    dtype=np.float32
)

y_test = np.asarray(
    labels,
    dtype=np.int32
)


# ============================================================
# BASIC SHAPES
# ============================================================

print()
print("=" * 100)
print("TEST ARRAY INFORMATION")
print("=" * 100)

print()
print(
    "X_test shape:",
    X_test.shape
)

print(
    "y_test shape:",
    y_test.shape
)

print(
    "X_test dtype:",
    X_test.dtype
)

print(
    "y_test dtype:",
    y_test.dtype
)


# ============================================================
# SANITY CHECK
# ============================================================

print()
print("=" * 100)
print("PREPROCESSING SANITY CHECK")
print("=" * 100)

print()
print(
    f"Minimum pixel value after normalization: "
    f"{X_test.min():.6f}"
)

print(
    f"Maximum pixel value after normalization: "
    f"{X_test.max():.6f}"
)

print(
    f"Mean after normalization: "
    f"{X_test.mean():.6f}"
)

print(
    f"Std after normalization: "
    f"{X_test.std():.6f}"
)

print()
print("Preprocessing:")
print("1. CLAHE image")
print("2. RGB loading")
print("3. Resize to 224 x 224")
print("4. Pixel scaling /255.0")
print("5. ImageNet normalization")


# ============================================================
# LOAD MODEL
# ============================================================

print()
print("=" * 100)
print("LOADING FINAL MODEL")
print("=" * 100)

model = tf.keras.models.load_model(
    MODEL_PATH
)

print()
print("Model loaded successfully.")

print()
print(
    "Model output shape:",
    model.output_shape
)


# ============================================================
# VERIFY MODEL OUTPUT
# ============================================================

print()
print("=" * 100)
print("VERIFYING MODEL OUTPUT")
print("=" * 100)

if model.output_shape[-1] != 1:

    raise ValueError(
        "Expected binary model output with shape "
        "(None, 1). "
        f"Found: {model.output_shape}"
    )

last_layer = model.layers[-1]

last_activation = getattr(
    last_layer,
    "activation",
    None
)

print()
print(
    "Last layer:",
    last_layer.name
)

print(
    "Last layer type:",
    type(last_layer).__name__
)

print(
    "Last activation:",
    last_activation
)

print()
print(
    "Binary sigmoid output confirmed."
)


# ============================================================
# MODEL SUMMARY
# ============================================================

print()
print("=" * 100)
print("MODEL SUMMARY")
print("=" * 100)

print()

model.summary()


# ============================================================
# RUN PREDICTION
# ============================================================

print()
print("=" * 100)
print("RUNNING FINAL MODEL PREDICTION")
print("=" * 100)

raw_predictions = model.predict(
    X_test,
    verbose=1
)

# ------------------------------------------------------------
# CONVERT TO 1D
# ------------------------------------------------------------

raw_predictions = np.asarray(
    raw_predictions,
    dtype=np.float32
).reshape(-1)


# ============================================================
# PREDICTION VALIDATION
# ============================================================

print()
print(
    "Prediction count:",
    len(raw_predictions)
)

if len(raw_predictions) != len(y_test):

    raise RuntimeError(
        "Prediction count does not match "
        "test labels."
    )

if not np.all(
    np.isfinite(raw_predictions)
):

    raise RuntimeError(
        "NaN or Inf values detected in predictions."
    )


# ============================================================
# SIGMOID PROBABILITY CHECK
# ============================================================

print()
print(
    "Prediction minimum:",
    f"{raw_predictions.min():.6f}"
)

print(
    "Prediction maximum:",
    f"{raw_predictions.max():.6f}"
)

print(
    "Prediction mean:",
    f"{raw_predictions.mean():.6f}"
)

if (
    raw_predictions.min() < 0
    or raw_predictions.max() > 1
):

    raise RuntimeError(
        "Model predictions are outside "
        "the expected sigmoid probability range [0, 1]."
    )

print()
print(
    "Sigmoid probability output confirmed."
)


# ============================================================
# BINARY PREDICTIONS
# ============================================================

y_pred = (
    raw_predictions >= THRESHOLD
).astype(
    np.int32
)

print()
print(
    "Threshold:",
    THRESHOLD
)

print(
    "Predicted Non-melanoma:",
    int(np.sum(y_pred == 0))
)

print(
    "Predicted Melanoma:",
    int(np.sum(y_pred == 1))
)


# ============================================================
# METRICS
# ============================================================

print()
print("=" * 100)
print("CALCULATING FINAL METRICS")
print("=" * 100)


# ------------------------------------------------------------
# ACCURACY
# ------------------------------------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)


# ------------------------------------------------------------
# PRECISION
# ------------------------------------------------------------

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)


# ------------------------------------------------------------
# SENSITIVITY / RECALL
# ------------------------------------------------------------

sensitivity = recall_score(
    y_test,
    y_pred,
    zero_division=0
)


# ------------------------------------------------------------
# SPECIFICITY
# ------------------------------------------------------------

specificity = recall_score(
    y_test,
    y_pred,
    pos_label=0,
    zero_division=0
)


# ------------------------------------------------------------
# F1
# ------------------------------------------------------------

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)


# ------------------------------------------------------------
# ROC AUC
# ------------------------------------------------------------

if len(np.unique(y_test)) == 2:

    roc_auc = roc_auc_score(
        y_test,
        raw_predictions
    )

else:

    roc_auc = np.nan


# ------------------------------------------------------------
# PR AUC / AVERAGE PRECISION
# ------------------------------------------------------------

if len(np.unique(y_test)) == 2:

    pr_auc = average_precision_score(
        y_test,
        raw_predictions
    )

else:

    pr_auc = np.nan


# ------------------------------------------------------------
# BALANCED SCORE
# ------------------------------------------------------------

balanced_score = (
    sensitivity + specificity
) / 2


# ============================================================
# PRINT METRICS
# ============================================================

print()
print(
    f"Threshold    : {THRESHOLD:.4f}"
)

print(
    f"Accuracy     : {accuracy:.4f}"
)

print(
    f"ROC-AUC      : {roc_auc:.4f}"
)

print(
    f"PR-AUC       : {pr_auc:.4f}"
)

print(
    f"Precision    : {precision:.4f}"
)

print(
    f"Sensitivity  : {sensitivity:.4f}"
)

print(
    f"Specificity  : {specificity:.4f}"
)

print(
    f"F1-score     : {f1:.4f}"
)

print(
    f"Balanced     : {balanced_score:.4f}"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print()
print("=" * 100)
print("CONFUSION MATRIX")
print("=" * 100)

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1]
)

tn = int(cm[0, 0])
fp = int(cm[0, 1])
fn = int(cm[1, 0])
tp = int(cm[1, 1])

print()
print(
    "True Negative :",
    tn
)

print(
    "False Positive:",
    fp
)

print(
    "False Negative:",
    fn
)

print(
    "True Positive :",
    tp
)

print()
print(
    "              Predicted"
)

print(
    "              Non-Mel    Melanoma"
)

print(
    f"Actual Non-Mel   "
    f"{tn:>5}      "
    f"{fp:>5}"
)

print(
    f"Actual Melanoma  "
    f"{fn:>5}      "
    f"{tp:>5}"
)


# ============================================================
# SAVE CONFUSION MATRIX CSV
# ============================================================

cm_df = pd.DataFrame(
    cm,
    index=[
        "Actual_Non_Melanoma",
        "Actual_Melanoma"
    ],
    columns=[
        "Predicted_Non_Melanoma",
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

print()
print(
    "Confusion matrix CSV saved:"
)

print(
    cm_path
)


# ============================================================
# CONFUSION MATRIX PLOT
# ============================================================

print()
print("=" * 100)
print("GENERATING CONFUSION MATRIX PLOT")
print("=" * 100)

plt.figure(
    figsize=(7, 6)
)

plt.imshow(
    cm
)

plt.title(
    "EfficientNetV2 + PH2 + CLAHE\n"
    "Confusion Matrix"
)

plt.xlabel(
    "Predicted Label"
)

plt.ylabel(
    "True Label"
)

plt.xticks(
    [0, 1],
    [
        "Non-melanoma",
        "Melanoma"
    ]
)

plt.yticks(
    [0, 1],
    [
        "Non-melanoma",
        "Melanoma"
    ]
)

for i in range(2):

    for j in range(2):

        plt.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center",
            fontsize=16
        )

plt.colorbar()

plt.tight_layout()

cm_plot_path = (
    OUTPUT_DIR
    / "final_confusion_matrix.png"
)

plt.savefig(
    cm_plot_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print()
print(
    "Confusion matrix plot saved:"
)

print(
    cm_plot_path
)


# ============================================================
# ROC CURVE
# ============================================================

print()
print("=" * 100)
print("GENERATING ROC CURVE")
print("=" * 100)

if len(np.unique(y_test)) == 2:

    fpr, tpr, roc_thresholds = (
        roc_curve(
            y_test,
            raw_predictions
        )
    )

    plt.figure(
        figsize=(8, 6)
    )

    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        label=(
            "EfficientNetV2 "
            f"(AUC = {roc_auc:.4f})"
        )
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1
    )

    plt.xlabel(
        "False Positive Rate"
    )

    plt.ylabel(
        "True Positive Rate"
    )

    plt.title(
        "ROC Curve - "
        "EfficientNetV2 + PH2 + CLAHE"
    )

    plt.legend(
        loc="lower right"
    )

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    roc_path = (
        OUTPUT_DIR
        / "roc_curve.png"
    )

    plt.savefig(
        roc_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print()
    print(
        "ROC curve saved:"
    )

    print(
        roc_path
    )

else:

    print(
        "ROC curve skipped: "
        "only one class present."
    )


# ============================================================
# PRECISION-RECALL CURVE
# ============================================================

print()
print("=" * 100)
print("GENERATING PRECISION-RECALL CURVE")
print("=" * 100)

if len(np.unique(y_test)) == 2:

    precision_curve, recall_curve, pr_thresholds = (
        precision_recall_curve(
            y_test,
            raw_predictions
        )
    )

    plt.figure(
        figsize=(8, 6)
    )

    plt.plot(
        recall_curve,
        precision_curve,
        linewidth=2,
        label=(
            "EfficientNetV2 "
            f"(AP = {pr_auc:.4f})"
        )
    )

    plt.xlabel(
        "Recall / Sensitivity"
    )

    plt.ylabel(
        "Precision"
    )

    plt.title(
        "Precision-Recall Curve - "
        "EfficientNetV2 + PH2 + CLAHE"
    )

    plt.legend(
        loc="upper right"
    )

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    pr_path = (
        OUTPUT_DIR
        / "precision_recall_curve.png"
    )

    plt.savefig(
        pr_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print()
    print(
        "Precision-Recall curve saved:"
    )

    print(
        pr_path
    )

else:

    print(
        "PR curve skipped: "
        "only one class present."
    )


# ============================================================
# PREDICTION SCORE DISTRIBUTION
# ============================================================

print()
print("=" * 100)
print("GENERATING PREDICTION SCORE DISTRIBUTION")
print("=" * 100)

melanoma_scores = (
    raw_predictions[
        y_test == 1
    ]
)

non_melanoma_scores = (
    raw_predictions[
        y_test == 0
    ]
)

plt.figure(
    figsize=(9, 6)
)

if len(non_melanoma_scores) > 0:

    plt.hist(
        non_melanoma_scores,
        bins=15,
        alpha=0.6,
        label="True Non-melanoma"
    )

if len(melanoma_scores) > 0:

    plt.hist(
        melanoma_scores,
        bins=15,
        alpha=0.6,
        label="True Melanoma"
    )

plt.axvline(
    THRESHOLD,
    linestyle="--",
    linewidth=2,
    label=(
        f"Threshold = {THRESHOLD:.2f}"
    )
)

plt.xlabel(
    "Predicted Melanoma Probability"
)

plt.ylabel(
    "Number of Images"
)

plt.title(
    "Prediction Score Distribution"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()

distribution_path = (
    OUTPUT_DIR
    / "prediction_score_distribution.png"
)

plt.savefig(
    distribution_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print()
print(
    "Prediction distribution saved:"
)

print(
    distribution_path
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print()
print("=" * 100)
print("CLASSIFICATION REPORT")
print("=" * 100)

report = classification_report(
    y_test,
    y_pred,
    labels=[0, 1],
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

report_path = (
    OUTPUT_DIR
    / "classification_report.csv"
)

report_df.to_csv(
    report_path
)

print()

print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=[
            "Non-melanoma",
            "Melanoma"
        ],
        zero_division=0
    )
)

print()
print(
    "Classification report saved:"
)

print(
    report_path
)


# ============================================================
# SAVE ALL PREDICTIONS
# ============================================================

print()
print("=" * 100)
print("SAVING ALL PREDICTIONS")
print("=" * 100)

predictions_df = pd.DataFrame(
    {
        "image_id": image_ids,

        "image_path": resolved_image_paths,

        "true_label": y_test,

        "raw_prediction": raw_predictions,

        "predicted_label": y_pred,

        "true_class": [
            (
                "Melanoma"
                if x == 1
                else "Non-melanoma"
            )
            for x in y_test
        ],

        "predicted_class": [
            (
                "Melanoma"
                if x == 1
                else "Non-melanoma"
            )
            for x in y_pred
        ],
    }
)

predictions_path = (
    OUTPUT_DIR
    / "final_model_predictions.csv"
)

predictions_df.to_csv(
    predictions_path,
    index=False
)

print()
print(
    "Predictions saved:"
)

print(
    predictions_path
)


# ============================================================
# SAVE FINAL METRICS
# ============================================================

print()
print("=" * 100)
print("SAVING FINAL METRICS")
print("=" * 100)

metrics_df = pd.DataFrame(
    [
        {
            "model":
                "EfficientNetV2_PH2_CLAHE",

            "dataset":
                "PH2",

            "preprocessing":
                "CLAHE",

            "threshold":
                THRESHOLD,

            "accuracy":
                accuracy,

            "roc_auc":
                roc_auc,

            "pr_auc":
                pr_auc,

            "precision":
                precision,

            "sensitivity":
                sensitivity,

            "specificity":
                specificity,

            "f1_score":
                f1,

            "balanced_score":
                balanced_score,

            "test_samples":
                len(y_test),

            "non_melanoma_samples":
                non_melanoma_count,

            "melanoma_samples":
                melanoma_count,

            "true_positives":
                tp,

            "true_negatives":
                tn,

            "false_positives":
                fp,

            "false_negatives":
                fn,
        }
    ]
)

metrics_path = (
    OUTPUT_DIR
    / "final_model_metrics.csv"
)

metrics_df.to_csv(
    metrics_path,
    index=False
)

print()
print(
    "Metrics saved:"
)

print(
    metrics_path
)


# ============================================================
# FINAL SUMMARY
# ============================================================

summary = pd.DataFrame(
    [
        {
            "model":
                "EfficientNetV2_PH2_CLAHE",

            "dataset":
                "PH2",

            "preprocessing":
                "CLAHE",

            "test_samples":
                len(y_test),

            "threshold":
                THRESHOLD,

            "accuracy":
                accuracy,

            "roc_auc":
                roc_auc,

            "pr_auc":
                pr_auc,

            "precision":
                precision,

            "sensitivity":
                sensitivity,

            "specificity":
                specificity,

            "f1_score":
                f1,

            "balanced_score":
                balanced_score,

            "true_positives":
                tp,

            "true_negatives":
                tn,

            "false_positives":
                fp,

            "false_negatives":
                fn,
        }
    ]
)

summary_path = (
    OUTPUT_DIR
    / "final_visualization_summary.csv"
)

summary.to_csv(
    summary_path,
    index=False
)

print()
print(
    "Summary saved:"
)

print(
    summary_path
)


# ============================================================
# FINAL STATUS
# ============================================================

print()
print("=" * 100)
print("FINAL MODEL VISUALIZATION SUMMARY")
print("=" * 100)

print()

print(
    "Model          : "
    "EfficientNetV2 + PH2 + CLAHE"
)

print(
    f"Test samples   : {len(y_test)}"
)

print(
    f"Threshold      : {THRESHOLD:.4f}"
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

print(
    f"Balanced score : {balanced_score:.4f}"
)

print()

print(
    f"TN             : {tn}"
)

print(
    f"FP             : {fp}"
)

print(
    f"FN             : {fn}"
)

print(
    f"TP             : {tp}"
)

print()
print(
    "Output directory:"
)

print(
    OUTPUT_DIR
)


# ============================================================
# GENERATED FILES
# ============================================================

print()
print("=" * 100)
print("GENERATED FILES")
print("=" * 100)

generated_files = sorted(
    OUTPUT_DIR.iterdir()
)

for file_path in generated_files:

    if file_path.is_file():

        print(
            f"  - {file_path.name}"
        )


# ============================================================
# FINAL PASS
# ============================================================

print()
print("=" * 100)
print("STATUS: PASS")
print("=" * 100)

print()

print(
    "Final EfficientNetV2 PH2 CLAHE "
    "evaluation and visualization "
    "completed successfully."
)

print()

print("=" * 100)