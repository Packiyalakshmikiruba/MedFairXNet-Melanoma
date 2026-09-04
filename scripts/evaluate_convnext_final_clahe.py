from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow import keras

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


DATA_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "final_binary_clahe_splits"
)


TEST_CSV = (
    DATA_DIR
    / "test_final_clahe.csv"
)


MODEL_PATH = (
    BASE_DIR
    / "models"
    / "melanoma_convnext_final_clahe.keras"
)


RESULTS_DIR = (
    BASE_DIR
    / "results"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


METRICS_PATH = (
    RESULTS_DIR
    / "convnext_final_clahe_test_evaluation.csv"
)


CONFUSION_PATH = (
    RESULTS_DIR
    / "convnext_final_clahe_confusion_matrix.csv"
)


REPORT_PATH = (
    RESULTS_DIR
    / "convnext_final_clahe_classification_report.csv"
)


PREDICTIONS_PATH = (
    RESULTS_DIR
    / "convnext_final_clahe_test_predictions.csv"
)


IMAGE_SIZE = (
    224,
    224
)


BATCH_SIZE = 8


THRESHOLD = 0.50


# ============================================================
# IMAGENET NORMALIZATION
# ============================================================

# IMPORTANT:
# This MUST match the training script.

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

print("=" * 90)

print(
    "MELONMA - CONVNEXT FINAL CLAHE TEST EVALUATION"
)

print("=" * 90)


print("\nTest CSV:")
print(TEST_CSV)


print("\nModel:")
print(MODEL_PATH)


print("\nImage preprocessing:")

print(
    "Resize       : 224 x 224"
)

print(
    "Pixel range  : 0 - 255 -> 0 - 1"
)

print(
    "Normalization: ImageNet mean/std"
)

print(
    "Mean         : [0.485, 0.456, 0.406]"
)

print(
    "Std          : [0.229, 0.224, 0.225]"
)


print("\nThreshold:")
print(THRESHOLD)


# ============================================================
# FILE CHECK
# ============================================================

if not TEST_CSV.exists():

    raise FileNotFoundError(
        f"\nTest CSV not found:\n{TEST_CSV}"
    )


if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"\nModel not found:\n{MODEL_PATH}"
    )


# ============================================================
# LOAD TEST METADATA
# ============================================================

print("\n" + "=" * 90)

print("LOADING TEST DATA")

print("=" * 90)


test_df = pd.read_csv(
    TEST_CSV
)


print(
    "\nTest samples:",
    len(test_df)
)


print(
    "\nTest columns:"
)

print(
    test_df.columns.tolist()
)


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = [

    "image_path",

    "binary_label",

    "binary_diagnosis",

    "image_id",

]


for column in required_columns:

    if column not in test_df.columns:

        raise ValueError(
            f"\nRequired column missing: {column}"
        )


# ============================================================
# COLUMN DEFINITIONS
# ============================================================

PATH_COLUMN = (
    "image_path"
)


LABEL_COLUMN = (
    "binary_label"
)


# ============================================================
# IMAGE PATH CONVERSION
# ============================================================

def make_absolute_path(path):

    path = Path(
        str(path)
    )

    if path.is_absolute():

        return str(path)

    return str(
        BASE_DIR / path
    )


test_df[PATH_COLUMN] = (

    test_df[PATH_COLUMN]
    .apply(make_absolute_path)

)


# ============================================================
# CHECK LABELS
# ============================================================

valid_labels = {
    0,
    1
}


labels_found = set(
    test_df[LABEL_COLUMN]
    .unique()
)


if not labels_found.issubset(
    valid_labels
):

    raise ValueError(
        f"Invalid labels found: {labels_found}"
    )


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\n" + "=" * 90)

print("CLASS DISTRIBUTION")

print("=" * 90)


non_melanoma_count = int(
    (
        test_df[LABEL_COLUMN] == 0
    ).sum()
)


melanoma_count = int(
    (
        test_df[LABEL_COLUMN] == 1
    ).sum()
)


print(
    "\nNon-melanoma:",
    non_melanoma_count
)


print(
    "Melanoma:",
    melanoma_count
)


# ============================================================
# CHECK TEST IMAGES
# ============================================================

print("\n" + "=" * 90)

print("CHECKING TEST IMAGES")

print("=" * 90)


missing_images = test_df[
    ~test_df[PATH_COLUMN]
    .apply(
        lambda x: Path(x).exists()
    )
]


print(
    "\nMissing images:",
    len(missing_images)
)


if len(missing_images) > 0:

    print(
        "\nFirst missing images:"
    )

    print(
        missing_images[
            PATH_COLUMN
        ]
        .head(10)
        .to_string(
            index=False
        )
    )

    raise FileNotFoundError(
        "\nTest images are missing."
    )


print(
    "All test images are available."
)


# ============================================================
# LOAD MODEL
# ============================================================

print("\n" + "=" * 90)

print("LOADING CONVNEXT MODEL")

print("=" * 90)


model = keras.models.load_model(
    MODEL_PATH,
    compile=False
)


print(
    "\nModel loaded successfully."
)


print(
    "\nModel input shape:"
)

print(
    model.input_shape
)


print(
    "\nModel output shape:"
)

print(
    model.output_shape
)


# ============================================================
# IMAGE LOADER
# ============================================================

def load_image(path):

    # --------------------------------------------------------
    # Read image
    # --------------------------------------------------------

    image = tf.io.read_file(
        path
    )


    # --------------------------------------------------------
    # Decode PNG/JPEG
    # --------------------------------------------------------

    image = tf.image.decode_image(

        image,

        channels=3,

        expand_animations=False

    )


    # --------------------------------------------------------
    # Static shape
    # --------------------------------------------------------

    image.set_shape(
        [None, None, 3]
    )


    # --------------------------------------------------------
    # Resize
    # --------------------------------------------------------

    image = tf.image.resize(

        image,

        IMAGE_SIZE,

        method=tf.image.ResizeMethod.BILINEAR

    )


    # --------------------------------------------------------
    # Convert to float32
    # --------------------------------------------------------

    image = tf.cast(

        image,

        tf.float32

    )


    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Training uses:
    #
    # 0-255 -> 0-1
    # --------------------------------------------------------

    image = (
        image / 255.0
    )


    # --------------------------------------------------------
    # ImageNet normalization
    #
    # Same as training.
    # --------------------------------------------------------

    image = (

        image - MEAN

    ) / STD


    return image


# ============================================================
# BUILD TEST DATASET
# ============================================================

print("\n" + "=" * 90)

print("BUILDING TEST DATASET")

print("=" * 90)


test_paths = (

    test_df[
        PATH_COLUMN
    ]
    .astype(str)
    .to_numpy()

)


test_labels = (

    test_df[
        LABEL_COLUMN
    ]
    .astype(int)
    .to_numpy()

)


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
    "\nTest dataset created."
)


print(
    "Number of test images:",
    len(test_paths)
)


# ============================================================
# PREPROCESSING SANITY CHECK
# ============================================================

print("\n" + "=" * 90)

print("PREPROCESSING SANITY CHECK")

print("=" * 90)


sample_image = load_image(
    test_paths[0]
)


sample_min = float(
    tf.reduce_min(
        sample_image
    ).numpy()
)


sample_max = float(
    tf.reduce_max(
        sample_image
    ).numpy()
)


sample_mean = float(
    tf.reduce_mean(
        sample_image
    ).numpy()
)


print(
    "\nFirst test image after preprocessing:"
)


print(
    "Shape:",
    sample_image.shape
)


print(
    "Minimum:",
    f"{sample_min:.6f}"
)


print(
    "Maximum:",
    f"{sample_max:.6f}"
)


print(
    "Mean:",
    f"{sample_mean:.6f}"
)


# ============================================================
# RUN PREDICTIONS
# ============================================================

print("\n" + "=" * 90)

print("RUNNING CONVNEXT PREDICTIONS")

print("=" * 90)


raw_predictions = (

    model.predict(

        test_dataset,

        verbose=1

    )

)


# ============================================================
# FLATTEN PREDICTIONS
# ============================================================

raw_predictions = np.asarray(
    raw_predictions
).reshape(-1)


print(
    "\nPrediction count:",
    len(raw_predictions)
)


print(
    "Test image count:",
    len(test_labels)
)


if len(raw_predictions) != len(
    test_labels
):

    raise ValueError(
        "\nPrediction count does not "
        "match test image count."
    )


print(
    "Prediction count matches "
    "test images."
)


# ============================================================
# PREDICTION SCORE SANITY CHECK
# ============================================================

print("\n" + "=" * 90)

print("PREDICTION SCORE SANITY CHECK")

print("=" * 90)


print(
    "\nMinimum prediction:",
    f"{raw_predictions.min():.6f}"
)


print(
    "Maximum prediction:",
    f"{raw_predictions.max():.6f}"
)


print(
    "Mean prediction:",
    f"{raw_predictions.mean():.6f}"
)


print(
    "Median prediction:",
    f"{np.median(raw_predictions):.6f}"
)


# ============================================================
# BINARY PREDICTIONS
# ============================================================

y_true = test_labels


y_pred = (
    raw_predictions >= THRESHOLD
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
    raw_predictions
)


pr_auc = average_precision_score(
    y_true,
    raw_predictions
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


tn, fp, fn, tp = (
    cm.ravel()
)


specificity = (

    tn / (tn + fp)

    if (tn + fp) > 0

    else 0.0

)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 90)

print("CONVNEXT FINAL CLAHE TEST RESULTS")

print("=" * 90)


print(
    "\nTest images :",
    len(y_true)
)


print(
    "Threshold   :",
    f"{THRESHOLD:.2f}"
)


print(
    "Accuracy    :",
    f"{accuracy:.6f}"
)


print(
    "ROC-AUC     :",
    f"{roc_auc:.6f}"
)


print(
    "PR-AUC      :",
    f"{pr_auc:.6f}"
)


print(
    "Precision   :",
    f"{precision:.6f}"
)


print(
    "Sensitivity :",
    f"{sensitivity:.6f}"
)


print(
    "Specificity :",
    f"{specificity:.6f}"
)


print(
    "F1 Score    :",
    f"{f1:.6f}"
)


# ============================================================
# CONFUSION MATRIX PRINT
# ============================================================

print(
    "\nConfusion Matrix:"
)


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
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 90)

print("CLASSIFICATION REPORT")

print("=" * 90)


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


print(
    classification_report(

        y_true,

        y_pred,

        target_names=[
            "Non-melanoma",
            "Melanoma"
        ],

        zero_division=0

    )
)


# ============================================================
# SAVE METRICS
# ============================================================

metrics_df = pd.DataFrame({

    "model": [
        "ConvNeXtTiny"
    ],

    "dataset": [
        "Final CLAHE Test"
    ],

    "test_samples": [
        len(y_true)
    ],

    "threshold": [
        THRESHOLD
    ],

    "accuracy": [
        accuracy
    ],

    "roc_auc": [
        roc_auc
    ],

    "pr_auc": [
        pr_auc
    ],

    "precision": [
        precision
    ],

    "sensitivity": [
        sensitivity
    ],

    "specificity": [
        specificity
    ],

    "f1_score": [
        f1
    ],

    "true_negative": [
        int(tn)
    ],

    "false_positive": [
        int(fp)
    ],

    "false_negative": [
        int(fn)
    ],

    "true_positive": [
        int(tp)
    ]

})


metrics_df.to_csv(

    METRICS_PATH,

    index=False

)


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

confusion_df = pd.DataFrame(

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


confusion_df.to_csv(

    CONFUSION_PATH

)


# ============================================================
# SAVE CLASSIFICATION REPORT
# ============================================================

report_df.to_csv(

    REPORT_PATH

)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

prediction_df = test_df.copy()


prediction_df[
    "true_label"
] = y_true


prediction_df[
    "melanoma_probability"
] = raw_predictions


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
    "error_type"
] = "CORRECT"


prediction_df.loc[

    (y_true == 0) &
    (y_pred == 1),

    "error_type"

] = "FALSE_POSITIVE"


prediction_df.loc[

    (y_true == 1) &
    (y_pred == 0),

    "error_type"

] = "FALSE_NEGATIVE"


prediction_df.loc[

    (y_true == 0) &
    (y_pred == 0),

    "error_type"

] = "TRUE_NEGATIVE"


prediction_df.loc[

    (y_true == 1) &
    (y_pred == 1),

    "error_type"

] = "TRUE_POSITIVE"


prediction_df.to_csv(

    PREDICTIONS_PATH,

    index=False

)


# ============================================================
# FILES SAVED
# ============================================================

print("\n" + "=" * 90)

print("FILES SAVED")

print("=" * 90)


print(
    "\nMetrics:"
)

print(
    METRICS_PATH
)


print(
    "\nConfusion matrix:"
)

print(
    CONFUSION_PATH
)


print(
    "\nClassification report:"
)

print(
    REPORT_PATH
)


print(
    "\nPredictions:"
)

print(
    PREDICTIONS_PATH
)


# ============================================================
# FINAL STATUS
# ============================================================

print("\n" + "=" * 90)

print("STATUS: PASS")

print(
    "ConvNeXt Final CLAHE test evaluation completed."
)

print("=" * 90)