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
)


# ============================================================
# CONFIGURATION
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
    / "melanoma_efficientnetv2_ph2_clahe.keras"
)

EXPECTED_EVALUATION_CSV = (
    BASE_DIR
    / "results"
    / "efficientnetv2_ph2_clahe_test_evaluation.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "results"
    / "ph2_prediction_distribution"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 8
EVALUATION_THRESHOLD = 0.50


# ============================================================
# IMAGE NORMALIZATION
# IMPORTANT:
# This MUST remain identical to
# evaluate_efficientnetv2_ph2_clahe.py
# ============================================================

MEAN = tf.constant(
    [0.485, 0.456, 0.406],
    dtype=tf.float32
)

STD = tf.constant(
    [0.229, 0.224, 0.225],
    dtype=tf.float32
)


# ============================================================
# HEADER
# ============================================================

print("=" * 100)
print("MELONMA - PH2 PREDICTION DISTRIBUTION ANALYSIS")
print("=" * 100)

print("\nModel:")
print(MODEL_PATH)

print("\nTest CSV:")
print(TEST_CSV)

print("\nExpected evaluation CSV:")
print(EXPECTED_EVALUATION_CSV)

print("\nOutput directory:")
print(OUTPUT_DIR)

print("\nImage size:")
print(IMAGE_SIZE)

print("\nBatch size:")
print(BATCH_SIZE)

print("\nEvaluation threshold:")
print(EVALUATION_THRESHOLD)

print("\nPreprocessing:")
print("1. CLAHE image path")
print("2. RGB PNG decoding")
print("3. Resize to 224 x 224")
print("4. Pixel scaling: /255.0")
print("5. ImageNet normalization")
print("   Mean = [0.485, 0.456, 0.406]")
print("   Std  = [0.229, 0.224, 0.225]")


# ============================================================
# FILE CHECKS
# ============================================================

if not TEST_CSV.exists():

    raise FileNotFoundError(
        f"\nTest CSV not found:\n{TEST_CSV}"
    )


if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"\nModel not found:\n{MODEL_PATH}"
    )


if not EXPECTED_EVALUATION_CSV.exists():

    raise FileNotFoundError(
        f"\nExpected evaluation CSV not found:\n"
        f"{EXPECTED_EVALUATION_CSV}"
    )


# ============================================================
# LOAD TEST DATA
# ============================================================

print("\n" + "=" * 100)
print("LOADING TEST DATA")
print("=" * 100)

test_df = pd.read_csv(TEST_CSV)

print("\nTest records:", len(test_df))


required_columns = [
    "image_id",
    "clahe_image_path",
    "binary_label",
]


for column in required_columns:

    if column not in test_df.columns:

        raise ValueError(
            f"Missing required column: {column}"
        )


# ============================================================
# MAKE CLAHE PATH ABSOLUTE
# SAME LOGIC AS FINAL EVALUATION SCRIPT
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
# CLASS DISTRIBUTION
# ============================================================

print("\nClass distribution:")

print(
    test_df["binary_label"]
    .value_counts()
    .sort_index()
)


expected_nonmelanoma = int(
    np.sum(test_df["binary_label"].values == 0)
)

expected_melanoma = int(
    np.sum(test_df["binary_label"].values == 1)
)


print("\nExpected test distribution:")
print(
    f"Non-melanoma : {expected_nonmelanoma}"
)
print(
    f"Melanoma     : {expected_melanoma}"
)


# ============================================================
# CLAHE IMAGE PATH VALIDATION
# ============================================================

print("\n" + "=" * 100)
print("CLAHE IMAGE PATH VALIDATION")
print("=" * 100)


missing = test_df[
    ~test_df["clahe_image_path"]
    .apply(lambda x: Path(x).exists())
]


print(
    "\nMissing CLAHE images:",
    len(missing)
)


if len(missing) > 0:

    print(
        missing["clahe_image_path"]
        .head(10)
        .to_string(index=False)
    )

    raise FileNotFoundError(
        "Some PH2 CLAHE test images are missing."
    )


print("\nAll CLAHE test images are available.")


# ============================================================
# IMAGE LOADER
#
# THIS IS COPIED LOGIC FROM
# evaluate_efficientnetv2_ph2_clahe.py
#
# DO NOT CHANGE WITHOUT ALSO CHANGING
# THE FINAL EVALUATION SCRIPT.
# ============================================================

def load_image(path):

    image = tf.io.read_file(path)

    image = tf.image.decode_png(
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

    image = image / 255.0

    image = (
        image - MEAN
    ) / STD

    return image


# ============================================================
# CREATE TEST DATASET
#
# IDENTICAL TO FINAL EVALUATION
# ============================================================

print("\n" + "=" * 100)
print("LOADING CLAHE TEST IMAGES")
print("=" * 100)


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
# PREPROCESSING SANITY CHECK
# ============================================================

print("\n" + "=" * 100)
print("PREPROCESSING SANITY CHECK")
print("=" * 100)


# Take the same dataset and inspect all images.
sanity_images = []

for batch in test_dataset:

    sanity_images.append(
        batch.numpy()
    )


sanity_array = np.concatenate(
    sanity_images,
    axis=0
)


print(
    f"\nMinimum pixel value after normalization: "
    f"{sanity_array.min():.6f}"
)

print(
    f"Maximum pixel value after normalization: "
    f"{sanity_array.max():.6f}"
)

print(
    f"Mean pixel value after normalization: "
    f"{sanity_array.mean():.6f}"
)

print(
    f"Std pixel value after normalization: "
    f"{sanity_array.std():.6f}"
)


if len(sanity_array) != len(test_df):

    raise ValueError(
        "Preprocessed image count does not match "
        "test records."
    )


# ============================================================
# RECREATE DATASET FOR PREDICTION
# ============================================================

# A fresh dataset is created so prediction is completely
# independent of the sanity-check iteration.

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
# LOAD FINAL MODEL
# ============================================================

print("\n" + "=" * 100)
print("LOADING FINAL MODEL")
print("=" * 100)


model = tf.keras.models.load_model(
    MODEL_PATH
)


print("\nModel loaded successfully.")

print(
    "Model output shape:",
    model.output_shape
)


# ============================================================
# RUN PREDICTIONS
# ============================================================

print("\n" + "=" * 100)
print("RUNNING PREDICTIONS")
print("=" * 100)


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


# ============================================================
# HANDLE MODEL OUTPUT SHAPE
# SAME LOGIC AS FINAL EVALUATION
# ============================================================

if raw_predictions.ndim == 2:

    if raw_predictions.shape[1] == 1:

        predictions = (
            raw_predictions[:, 0]
        )

    elif raw_predictions.shape[1] == 2:

        predictions = (
            raw_predictions[:, 1]
        )

    else:

        raise ValueError(
            f"Unexpected model output shape: "
            f"{raw_predictions.shape}"
        )

else:

    predictions = (
        raw_predictions.reshape(-1)
    )


predictions = np.asarray(
    predictions
).reshape(-1)


print(
    "\nFinal prediction count:",
    len(predictions)
)


if len(predictions) != len(test_df):

    raise ValueError(
        "Prediction count does not match "
        "test image count."
    )


print(
    "Prediction count matches test images."
)


# ============================================================
# PREDICTION SCORE SANITY CHECK
# ============================================================

print("\n" + "=" * 100)
print("PREDICTION SCORE SANITY CHECK")
print("=" * 100)


print(
    f"\nMinimum prediction: "
    f"{predictions.min():.6f}"
)

print(
    f"Maximum prediction: "
    f"{predictions.max():.6f}"
)

print(
    f"Mean prediction: "
    f"{predictions.mean():.6f}"
)

print(
    f"Median prediction: "
    f"{np.median(predictions):.6f}"
)


# ============================================================
# TRUE LABELS
# ============================================================

y_true = (
    test_df["binary_label"]
    .astype(int)
    .values
)


# ============================================================
# PREDICTIONS AT 0.50
# EXACTLY SAME AS FINAL EVALUATION
# ============================================================

y_pred = (
    predictions >= EVALUATION_THRESHOLD
).astype(int)


# ============================================================
# SAVE RAW PREDICTIONS
# ============================================================

results = pd.DataFrame({

    "image_id":
        test_df["image_id"].astype(str).values,

    "true_label":
        y_true,

    "true_class": [
        "Melanoma"
        if x == 1
        else "Non-melanoma"
        for x in y_true
    ],

    "prediction_score":
        predictions,

    "prediction_at_0_50":
        y_pred,

    "predicted_class_at_0_50": [
        "Melanoma"
        if x == 1
        else "Non-melanoma"
        for x in y_pred
    ]
})


raw_prediction_path = (
    OUTPUT_DIR
    / "ph2_raw_predictions.csv"
)


results.to_csv(
    raw_prediction_path,
    index=False
)


print("\nRaw predictions saved:")
print(raw_prediction_path)


# ============================================================
# SCORE DISTRIBUTION
# ============================================================

print("\n" + "=" * 100)
print("PREDICTION SCORE DISTRIBUTION")
print("=" * 100)


mel_scores = predictions[
    y_true == 1
]

nonmel_scores = predictions[
    y_true == 0
]


print("\nMelanoma scores:")

print(
    f"Count  : {len(mel_scores)}"
)

print(
    f"Min    : {mel_scores.min():.6f}"
)

print(
    f"Max    : {mel_scores.max():.6f}"
)

print(
    f"Mean   : {mel_scores.mean():.6f}"
)

print(
    f"Median : {np.median(mel_scores):.6f}"
)


print("\nNon-melanoma scores:")

print(
    f"Count  : {len(nonmel_scores)}"
)

print(
    f"Min    : {nonmel_scores.min():.6f}"
)

print(
    f"Max    : {nonmel_scores.max():.6f}"
)

print(
    f"Mean   : {nonmel_scores.mean():.6f}"
)

print(
    f"Median : {np.median(nonmel_scores):.6f}"
)


# ============================================================
# THRESHOLD ANALYSIS
# ============================================================

print("\n" + "=" * 100)
print("THRESHOLD ANALYSIS")
print("=" * 100)


threshold_rows = []


thresholds = np.arange(
    0.05,
    0.96,
    0.01
)


for threshold in thresholds:

    y_threshold_pred = (
        predictions >= threshold
    ).astype(int)


    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_threshold_pred,
        labels=[0, 1]
    ).ravel()


    accuracy = accuracy_score(
        y_true,
        y_threshold_pred
    )


    precision = precision_score(
        y_true,
        y_threshold_pred,
        zero_division=0
    )


    sensitivity = recall_score(
        y_true,
        y_threshold_pred,
        zero_division=0
    )


    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )


    f1 = f1_score(
        y_true,
        y_threshold_pred,
        zero_division=0
    )


    balanced_score = (
        sensitivity + specificity
    ) / 2


    threshold_rows.append({

        "threshold":
            round(float(threshold), 2),

        "accuracy":
            accuracy,

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

        "true_positive":
            tp,

        "true_negative":
            tn,

        "false_positive":
            fp,

        "false_negative":
            fn
    })


threshold_df = pd.DataFrame(
    threshold_rows
)


threshold_path = (
    OUTPUT_DIR
    / "ph2_threshold_analysis.csv"
)


threshold_df.to_csv(
    threshold_path,
    index=False
)


print("\nThreshold analysis saved:")
print(threshold_path)


# ============================================================
# 0.50 THRESHOLD VALIDATION
# ============================================================

print("\n" + "=" * 100)
print("THRESHOLD 0.50 VALIDATION")
print("=" * 100)


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


balanced_score = (
    sensitivity + specificity
) / 2


print(
    f"\nThreshold   : "
    f"{EVALUATION_THRESHOLD:.2f}"
)

print(
    f"Accuracy    : "
    f"{accuracy:.4f}"
)

print(
    f"Precision   : "
    f"{precision:.4f}"
)

print(
    f"Sensitivity : "
    f"{sensitivity:.4f}"
)

print(
    f"Specificity : "
    f"{specificity:.4f}"
)

print(
    f"F1-score    : "
    f"{f1:.4f}"
)

print(
    f"Balanced    : "
    f"{balanced_score:.4f}"
)


print("\nConfusion matrix:")

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


# ============================================================
# BEST THRESHOLDS
# ============================================================

print("\n" + "=" * 100)
print("BEST THRESHOLDS")
print("=" * 100)


best_f1 = threshold_df.loc[
    threshold_df["f1_score"].idxmax()
]


best_balanced = threshold_df.loc[
    threshold_df["balanced_score"].idxmax()
]


best_accuracy = threshold_df.loc[
    threshold_df["accuracy"].idxmax()
]


print("\nBest F1 threshold:")

print(
    f"Threshold   : "
    f"{best_f1['threshold']:.2f}"
)

print(
    f"Accuracy    : "
    f"{best_f1['accuracy']:.4f}"
)

print(
    f"Precision   : "
    f"{best_f1['precision']:.4f}"
)

print(
    f"Sensitivity : "
    f"{best_f1['sensitivity']:.4f}"
)

print(
    f"Specificity : "
    f"{best_f1['specificity']:.4f}"
)

print(
    f"F1-score    : "
    f"{best_f1['f1_score']:.4f}"
)


print("\nBest balanced threshold:")

print(
    f"Threshold   : "
    f"{best_balanced['threshold']:.2f}"
)

print(
    f"Accuracy    : "
    f"{best_balanced['accuracy']:.4f}"
)

print(
    f"Precision   : "
    f"{best_balanced['precision']:.4f}"
)

print(
    f"Sensitivity : "
    f"{best_balanced['sensitivity']:.4f}"
)

print(
    f"Specificity : "
    f"{best_balanced['specificity']:.4f}"
)

print(
    f"Balanced    : "
    f"{best_balanced['balanced_score']:.4f}"
)


print("\nBest accuracy threshold:")

print(
    f"Threshold   : "
    f"{best_accuracy['threshold']:.2f}"
)

print(
    f"Accuracy    : "
    f"{best_accuracy['accuracy']:.4f}"
)


# ============================================================
# ROC-AUC / PR-AUC
# ============================================================

print("\n" + "=" * 100)
print("RANKING METRICS")
print("=" * 100)


try:

    roc_auc = roc_auc_score(
        y_true,
        predictions
    )

except Exception:

    roc_auc = np.nan


try:

    pr_auc = average_precision_score(
        y_true,
        predictions
    )

except Exception:

    pr_auc = np.nan


print(
    f"\nROC-AUC : "
    f"{roc_auc:.4f}"
)


print(
    f"PR-AUC  : "
    f"{pr_auc:.4f}"
)


# ============================================================
# LOAD EXPECTED FINAL EVALUATION
# ============================================================

print("\n" + "=" * 100)
print("EXPECTED FINAL EVALUATION COMPARISON")
print("=" * 100)


expected_df = pd.read_csv(
    EXPECTED_EVALUATION_CSV
)


if len(expected_df) != 1:

    raise ValueError(
        "Expected evaluation CSV should contain "
        "exactly one result row."
    )


expected = expected_df.iloc[0]


metrics_to_compare = [

    "accuracy",
    "roc_auc",
    "pr_auc",
    "precision",
    "sensitivity",
    "specificity",
    "f1_score",
    "true_negative",
    "false_positive",
    "false_negative",
    "true_positive",
]


actual_values = {

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

    "true_negative":
        tn,

    "false_positive":
        fp,

    "false_negative":
        fn,

    "true_positive":
        tp,
}


comparison_rows = []


all_match = True


for metric in metrics_to_compare:

    expected_value = float(
        expected[metric]
    )

    actual_value = float(
        actual_values[metric]
    )


    # --------------------------------------------------------
    # Integer confusion-matrix values
    # --------------------------------------------------------

    if metric in [
        "true_negative",
        "false_positive",
        "false_negative",
        "true_positive",
    ]:

        match = (
            int(round(actual_value))
            == int(round(expected_value))
        )

    else:

        match = np.isclose(
            actual_value,
            expected_value,
            atol=1e-6,
            rtol=1e-6
        )


    if not match:

        all_match = False


    comparison_rows.append({

        "metric":
            metric,

        "actual":
            actual_value,

        "expected":
            expected_value,

        "match":
            match
    })


    print(
        f"\n{metric:<20}: "
        f"{actual_value:.10f}"
    )

    print(
        f"{'':20}  "
        f"expected = {expected_value:.10f} "
        f"-> "
        f"{'MATCH' if match else 'MISMATCH'}"
    )


# ============================================================
# SAVE COMPARISON
# ============================================================

comparison_df = pd.DataFrame(
    comparison_rows
)


comparison_path = (
    OUTPUT_DIR
    / "ph2_evaluation_consistency_check.csv"
)


comparison_df.to_csv(
    comparison_path,
    index=False
)


# ============================================================
# SAVE SUMMARY
# ============================================================

summary = pd.DataFrame([{

    "model":
        "EfficientNetV2_PH2_CLAHE",

    "test_samples":
        len(y_true),

    "non_melanoma_samples":
        int(np.sum(y_true == 0)),

    "melanoma_samples":
        int(np.sum(y_true == 1)),

    "evaluation_threshold":
        EVALUATION_THRESHOLD,

    "roc_auc":
        roc_auc,

    "pr_auc":
        pr_auc,

    "accuracy":
        accuracy,

    "precision":
        precision,

    "sensitivity":
        sensitivity,

    "specificity":
        specificity,

    "f1_score":
        f1,

    "true_negative":
        tn,

    "false_positive":
        fp,

    "false_negative":
        fn,

    "true_positive":
        tp,

    "best_f1_threshold":
        best_f1["threshold"],

    "best_f1":
        best_f1["f1_score"],

    "best_balanced_threshold":
        best_balanced["threshold"],

    "best_balanced_score":
        best_balanced["balanced_score"],

    "best_accuracy_threshold":
        best_accuracy["threshold"],

    "best_accuracy":
        best_accuracy["accuracy"],

    "matches_final_evaluation":
        all_match
}])


summary_path = (
    OUTPUT_DIR
    / "ph2_prediction_distribution_summary.csv"
)


summary.to_csv(
    summary_path,
    index=False
)


# ============================================================
# FILES SAVED
# ============================================================

print("\n" + "=" * 100)
print("FILES SAVED")
print("=" * 100)


print("\nRaw predictions:")
print(raw_prediction_path)


print("\nThreshold analysis:")
print(threshold_path)


print("\nConsistency check:")
print(comparison_path)


print("\nSummary:")
print(summary_path)


# ============================================================
# FINAL STATUS
# ============================================================

print("\n" + "=" * 100)


if all_match:

    print("STATUS: PASS")

    print(
        "PH2 prediction distribution analysis "
        "matches the final PH2 evaluation."
    )

else:

    print("STATUS: REVIEW REQUIRED")

    print(
        "PH2 prediction distribution analysis "
        "does not match the final PH2 evaluation."
    )


print("=" * 100)
