from pathlib import Path
import time

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
    roc_curve,
    precision_recall_curve,
)

import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent


# ============================================================
# FINAL CLAHE TEST CSV
# ============================================================

TEST_CSV = (
    BASE_DIR
    / "data"
    / "processed"
    / "final_binary_clahe_splits"
    / "test_final_clahe.csv"
)


# ============================================================
# TRAINED EFFICIENTNETV2 MODEL
# ============================================================

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "melanoma_efficientnetv2_final_clahe.keras"
)


# ============================================================
# RESULTS DIRECTORY
# ============================================================

RESULTS_DIR = (
    BASE_DIR
    / "results"
    / "efficientnetv2_final_clahe"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# OUTPUT FILES
# ============================================================

METRICS_PATH = (
    RESULTS_DIR
    / "efficientnetv2_final_clahe_metrics.csv"
)

PREDICTIONS_PATH = (
    RESULTS_DIR
    / "efficientnetv2_final_clahe_predictions.csv"
)

REPORT_PATH = (
    RESULTS_DIR
    / "efficientnetv2_final_clahe_classification_report.txt"
)

ROC_CURVE_PATH = (
    RESULTS_DIR
    / "efficientnetv2_final_clahe_roc_curve.png"
)

PR_CURVE_PATH = (
    RESULTS_DIR
    / "efficientnetv2_final_clahe_pr_curve.png"
)

CONFUSION_MATRIX_PATH = (
    RESULTS_DIR
    / "efficientnetv2_final_clahe_confusion_matrix.png"
)


# ============================================================
# MODEL / DATA CONFIGURATION
# ============================================================

IMAGE_SIZE = (
    224,
    224
)

BATCH_SIZE = 8

THRESHOLD = 0.5

RANDOM_SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

tf.random.set_seed(
    RANDOM_SEED
)

np.random.seed(
    RANDOM_SEED
)


# ============================================================
# HEADER
# ============================================================

print(
    "=" * 90
)

print(
    "MELONMA - EFFICIENTNETV2 "
    "FINAL CLAHE TEST EVALUATION"
)

print(
    "=" * 90
)


print(
    "\nTensorFlow version:"
)

print(
    tf.__version__
)


print(
    "\nTest CSV:"
)

print(
    TEST_CSV
)


print(
    "\nModel:"
)

print(
    MODEL_PATH
)


print(
    "\nImage size:"
)

print(
    f"{IMAGE_SIZE[0]} x {IMAGE_SIZE[1]}"
)


print(
    "\nBatch size:"
)

print(
    BATCH_SIZE
)


print(
    "\nThreshold:"
)

print(
    THRESHOLD
)


print(
    "\nEvaluation preprocessing:"
)

print(
    "CLAHE images"
)


# ============================================================
# FILE CHECK
# ============================================================

if not TEST_CSV.exists():

    raise FileNotFoundError(
        f"\nTest CSV not found:\n"
        f"{TEST_CSV}"
    )


if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"\nEfficientNetV2 model not found:\n"
        f"{MODEL_PATH}"
    )


# ============================================================
# LOAD TEST CSV
# ============================================================

test_df = pd.read_csv(
    TEST_CSV
)


print(
    "\n" + "=" * 90
)

print(
    "TEST DATA INFORMATION"
)

print(
    "=" * 90
)


print(
    "\nTest images:",
    len(test_df)
)


print(
    "\nColumns:"
)

print(
    list(test_df.columns)
)


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = [
    "image_path",
    "binary_label",
]


for column in required_columns:

    if column not in test_df.columns:

        raise ValueError(
            f"\nRequired column missing: "
            f"{column}"
        )


# ============================================================
# LABEL VALIDATION
# ============================================================

test_df[
    "binary_label"
] = (
    pd.to_numeric(
        test_df["binary_label"],
        errors="raise"
    )
    .astype(int)
)


unique_labels = set(
    test_df[
        "binary_label"
    ].unique()
)


if not unique_labels.issubset(
    {0, 1}
):

    raise ValueError(
        "\nInvalid binary labels detected: "
        f"{unique_labels}"
    )


# ============================================================
# IMAGE PATH RESOLUTION
# ============================================================

def make_absolute_path(
    path
):

    path = Path(
        str(path)
    )

    if path.is_absolute():

        return str(path)

    return str(
        BASE_DIR / path
    )


test_df[
    "resolved_image_path"
] = (
    test_df[
        "image_path"
    ]
    .apply(
        make_absolute_path
    )
)


# ============================================================
# IMAGE CHECK
# ============================================================

missing_images = test_df[
    ~test_df[
        "resolved_image_path"
    ]
    .apply(
        lambda x: Path(x).exists()
    )
]


print(
    "\nMissing test images:",
    len(missing_images)
)


if len(missing_images) > 0:

    print(
        "\nFirst missing images:"
    )

    print(
        missing_images[
            "resolved_image_path"
        ]
        .head(10)
        .to_string(
            index=False
        )
    )

    raise FileNotFoundError(
        "\nSome test images are missing."
    )


print(
    "All test images are available."
)


# ============================================================
# TEST CLASS DISTRIBUTION
# ============================================================

print(
    "\n" + "=" * 90
)

print(
    "TEST CLASS DISTRIBUTION"
)

print(
    "=" * 90
)


label_counts = (
    test_df[
        "binary_label"
    ]
    .value_counts()
    .sort_index()
)


print(
    "\nBinary label distribution:"
)

print(
    label_counts
)


print(
    "\nNon-melanoma (0):",
    int(
        label_counts.get(
            0,
            0
        )
    )
)


print(
    "Melanoma (1):",
    int(
        label_counts.get(
            1,
            0
        )
    )
)


# ============================================================
# DATASET
# ============================================================

print(
    "\n" + "=" * 90
)

print(
    "TEST DATALOADER"
)

print(
    "=" * 90
)


test_paths = (
    test_df[
        "resolved_image_path"
    ]
    .astype(str)
    .values
)


# ============================================================
# IMAGE LOADING
# ============================================================

def load_image(
    path
):

    image = tf.io.read_file(
        path
    )

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

    return image


# ============================================================
# CREATE TF DATASET
# ============================================================

test_dataset = (
    tf.data.Dataset
    .from_tensor_slices(
        test_paths
    )
    .map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    .batch(
        BATCH_SIZE
    )
    .prefetch(
        tf.data.AUTOTUNE
    )
)


print(
    "\nTest samples:",
    len(test_df)
)


print(
    "Test batches:",
    int(
        np.ceil(
            len(test_df)
            /
            BATCH_SIZE
        )
    )
)


# ============================================================
# LOAD MODEL
# ============================================================

print(
    "\n" + "=" * 90
)

print(
    "LOADING FINAL EFFICIENTNETV2 MODEL"
)

print(
    "=" * 90
)


print(
    "\nLoading checkpoint..."
)


model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)


print(
    "\nModel loaded successfully."
)


# ============================================================
# MODEL INFORMATION
# ============================================================

print(
    "\n" + "=" * 90
)

print(
    "MODEL INFORMATION"
)

print(
    "=" * 90
)


print(
    "\nModel name:"
)

print(
    "EfficientNetV2"
)


print(
    "\nInput shape:"
)

print(
    model.input_shape
)


print(
    "\nOutput shape:"
)

print(
    model.output_shape
)


print(
    "\nTotal parameters:"
)

print(
    model.count_params()
)


# ============================================================
# OUTPUT SHAPE SAFETY CHECK
# ============================================================

output_shape = model.output_shape


if (
    len(output_shape) != 2
    or
    output_shape[-1] != 1
):

    raise RuntimeError(
        "\nUnexpected model output shape: "
        f"{output_shape}\n"
        "Expected binary output shape: "
        "(None, 1)"
    )


print(
    "\nBinary classification output verified."
)


# ============================================================
# PREDICTION
# ============================================================

print(
    "\n" + "=" * 90
)

print(
    "RUNNING FINAL EFFICIENTNETV2 TEST PREDICTION"
)

print(
    "=" * 90
)


start_time = (
    time.time()
)


predictions = model.predict(
    test_dataset,
    verbose=1
)


evaluation_time = (
    time.time()
    -
    start_time
)


predictions = np.asarray(
    predictions
).reshape(-1)


# ============================================================
# PREDICTION COUNT CHECK
# ============================================================

print(
    "\nPrediction count:",
    len(predictions)
)


print(
    "Test image count:",
    len(test_df)
)


if len(predictions) != len(
    test_df
):

    raise RuntimeError(
        "\nPrediction count does not "
        "match test image count."
    )


# ============================================================
# PREDICTION SCORE SANITY CHECK
# ============================================================

print(
    "\n" + "=" * 90
)

print(
    "PREDICTION SCORE SANITY CHECK"
)

print(
    "=" * 90
)


print(
    "\nMinimum prediction:",
    f"{predictions.min():.6f}"
)


print(
    "Maximum prediction:",
    f"{predictions.max():.6f}"
)


print(
    "Mean prediction:",
    f"{predictions.mean():.6f}"
)


print(
    "Median prediction:",
    f"{np.median(predictions):.6f}"
)


if np.any(
    np.isnan(predictions)
):

    raise RuntimeError(
        "\nNaN predictions detected."
    )


if np.any(
    predictions < 0
) or np.any(
    predictions > 1
):

    raise RuntimeError(
        "\nPredictions are outside "
        "the valid probability range [0, 1]."
    )


# ============================================================
# TRUE LABELS
# ============================================================

y_true = (
    test_df[
        "binary_label"
    ]
    .astype(int)
    .values
)


# ============================================================
# BINARY PREDICTIONS
# ============================================================

y_pred = (
    predictions
    >=
    THRESHOLD
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


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=[
        0,
        1
    ]
)


tn, fp, fn, tp = (
    cm.ravel()
)


# ============================================================
# SPECIFICITY
# ============================================================

if (
    tn + fp
) > 0:

    specificity = (
        tn
        /
        (
            tn + fp
        )
    )

else:

    specificity = 0.0


# ============================================================
# FINAL RESULTS
# ============================================================

print(
    "\n" + "=" * 90
)

print(
    "FINAL EFFICIENTNETV2 TEST RESULTS"
)

print(
    "=" * 90
)


print(
    "\nTest images   :",
    len(y_true)
)


print(
    "Threshold     :",
    THRESHOLD
)


print(
    "Accuracy      :",
    f"{accuracy:.6f}"
)


print(
    "ROC-AUC       :",
    f"{roc_auc:.6f}"
)


print(
    "PR-AUC        :",
    f"{pr_auc:.6f}"
)


print(
    "Precision      :",
    f"{precision:.6f}"
)


print(
    "Sensitivity    :",
    f"{sensitivity:.6f}"
)


print(
    "Specificity    :",
    f"{specificity:.6f}"
)


print(
    "F1-score       :",
    f"{f1:.6f}"
)


print(
    "\nTrue Negative :",
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
    "True Positive  :",
    tp
)


print(
    "\nEvaluation time:",
    f"{evaluation_time:.2f} seconds"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

classification_report_text = (
    classification_report(
        y_true,
        y_pred,
        labels=[
            0,
            1
        ],
        target_names=[
            "Non-melanoma",
            "Melanoma"
        ],
        digits=4,
        zero_division=0
    )
)


print(
    "\n" + "=" * 90
)

print(
    "CLASSIFICATION REPORT"
)

print(
    "=" * 90
)


print(
    classification_report_text
)


# ============================================================
# SAVE CLASSIFICATION REPORT
# ============================================================

with open(
    REPORT_PATH,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "EFFICIENTNETV2 - FINAL CLAHE "
        "TEST CLASSIFICATION REPORT\n"
    )

    file.write(
        "=" * 80
        +
        "\n\n"
    )

    file.write(
        classification_report_text
    )

    file.write(
        "\n\nAdditional Metrics\n"
    )

    file.write(
        "=" * 80
        +
        "\n"
    )

    file.write(
        f"Test samples: "
        f"{len(y_true)}\n"
    )

    file.write(
        f"Threshold: "
        f"{THRESHOLD:.6f}\n"
    )

    file.write(
        f"Accuracy: "
        f"{accuracy:.6f}\n"
    )

    file.write(
        f"ROC-AUC: "
        f"{roc_auc:.6f}\n"
    )

    file.write(
        f"PR-AUC: "
        f"{pr_auc:.6f}\n"
    )

    file.write(
        f"Precision: "
        f"{precision:.6f}\n"
    )

    file.write(
        f"Sensitivity: "
        f"{sensitivity:.6f}\n"
    )

    file.write(
        f"Specificity: "
        f"{specificity:.6f}\n"
    )

    file.write(
        f"F1-score: "
        f"{f1:.6f}\n"
    )

    file.write(
        f"TN: {tn}\n"
    )

    file.write(
        f"FP: {fp}\n"
    )

    file.write(
        f"FN: {fn}\n"
    )

    file.write(
        f"TP: {tp}\n"
    )

    file.write(
        f"Evaluation time seconds: "
        f"{evaluation_time:.4f}\n"
    )


# ============================================================
# SAVE METRICS CSV
# ============================================================

metrics_df = pd.DataFrame(
    [
        {
            "model": "EfficientNetV2",
            "dataset": "Final CLAHE Test",
            "test_samples": len(y_true),
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
            "evaluation_time_seconds":
                evaluation_time,
        }
    ]
)


metrics_df.to_csv(
    METRICS_PATH,
    index=False
)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

prediction_df = (
    test_df.copy()
)


prediction_df[
    "true_label"
] = y_true


prediction_df[
    "melanoma_probability"
] = predictions


prediction_df[
    "predicted_label"
] = y_pred


prediction_df[
    "prediction"
] = np.where(
    y_pred == 1,
    "Melanoma",
    "Non-melanoma"
)


prediction_df[
    "correct"
] = (
    y_true
    ==
    y_pred
)


prediction_df.to_csv(
    PREDICTIONS_PATH,
    index=False
)


# ============================================================
# ROC CURVE
# ============================================================

fpr, tpr, _ = (
    roc_curve(
        y_true,
        predictions
    )
)


plt.figure(
    figsize=(7, 6)
)


plt.plot(
    fpr,
    tpr,
    label=(
        f"EfficientNetV2 "
        f"(AUC = {roc_auc:.4f})"
    )
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
    "EfficientNetV2 ROC Curve - "
    "Final CLAHE Test"
)


plt.legend(
    loc="lower right"
)


plt.grid(
    alpha=0.3
)


plt.tight_layout()


plt.savefig(
    ROC_CURVE_PATH,
    dpi=300
)


plt.close()


# ============================================================
# PRECISION-RECALL CURVE
# ============================================================

precision_curve, recall_curve, _ = (
    precision_recall_curve(
        y_true,
        predictions
    )
)


plt.figure(
    figsize=(7, 6)
)


plt.plot(
    recall_curve,
    precision_curve,
    label=(
        f"PR-AUC = "
        f"{pr_auc:.4f}"
    )
)


plt.xlabel(
    "Recall"
)


plt.ylabel(
    "Precision"
)


plt.title(
    "EfficientNetV2 Precision-Recall "
    "Curve - Final CLAHE Test"
)


plt.legend()


plt.grid(
    alpha=0.3
)


plt.tight_layout()


plt.savefig(
    PR_CURVE_PATH,
    dpi=300
)


plt.close()


# ============================================================
# CONFUSION MATRIX FIGURE
# ============================================================

plt.figure(
    figsize=(7, 6)
)


plt.imshow(
    cm,
    interpolation="nearest"
)


plt.title(
    "EfficientNetV2 Confusion Matrix - "
    "Final CLAHE Test"
)


plt.colorbar()


tick_marks = np.arange(
    2
)


plt.xticks(
    tick_marks,
    [
        "Non-melanoma",
        "Melanoma"
    ],
    rotation=20
)


plt.yticks(
    tick_marks,
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
            str(
                cm[i, j]
            ),
            ha="center",
            va="center",
            fontsize=14
        )


plt.ylabel(
    "True Label"
)


plt.xlabel(
    "Predicted Label"
)


plt.tight_layout()


plt.savefig(
    CONFUSION_MATRIX_PATH,
    dpi=300
)


plt.close()


# ============================================================
# FINAL OUTPUT
# ============================================================

print(
    "\n" + "=" * 90
)

print(
    "EFFICIENTNETV2 EVALUATION COMPLETED"
)

print(
    "=" * 90
)


print(
    "\nMetrics saved:"
)

print(
    METRICS_PATH
)


print(
    "\nPredictions saved:"
)

print(
    PREDICTIONS_PATH
)


print(
    "\nClassification report saved:"
)

print(
    REPORT_PATH
)


print(
    "\nROC curve saved:"
)

print(
    ROC_CURVE_PATH
)


print(
    "\nPR curve saved:"
)

print(
    PR_CURVE_PATH
)


print(
    "\nConfusion matrix saved:"
)

print(
    CONFUSION_MATRIX_PATH
)


print(
    "\n" + "=" * 90
)

print(
    "STATUS: PASS"
)

print(
    "EfficientNetV2 final CLAHE "
    "test evaluation completed successfully."
)

print(
    "=" * 90
)
