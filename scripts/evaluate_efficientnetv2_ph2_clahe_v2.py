# ============================================================
# MELONMA
# PH2 CLAHE - EFFICIENTNETV2B0 V2 TEST EVALUATION
# ============================================================

import os

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

IMAGE_SIZE = (224, 224)

BATCH_SIZE = 8

PROJECT_ROOT = Path(
    r"C:\Users\HP\Desktop\Melonma"
)


TEST_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "PH2"
    / "splits"
    / "test_ph2_clahe.csv"
)


MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "melanoma_efficientnetv2_ph2_clahe_v2.keras"
)


RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
)


RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


RESULT_PATH = (
    RESULTS_DIR
    / "efficientnetv2_ph2_clahe_test_evaluation.csv"
)


# ============================================================
# HEADER
# ============================================================

print()

print("=" * 75)

print(
    "MELONMA - PH2 CLAHE EFFICIENTNETV2 TEST EVALUATION"
)

print("=" * 75)


# ============================================================
# CHECK FILES
# ============================================================

print()

print("Test CSV:")

print(TEST_CSV)

print()

print("Model:")

print(MODEL_PATH)


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

test_df = pd.read_csv(
    TEST_CSV
)


print()

print(
    "Test images:",
    len(test_df)
)


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = [

    "clahe_image_path",

    "binary_label",

    "binary_diagnosis",

]


missing_columns = [

    column

    for column in required_columns

    if column not in test_df.columns

]


if missing_columns:

    raise ValueError(

        "Missing required columns: "

        + str(missing_columns)

    )


# ============================================================
# CHECK IMAGE PATHS
# ============================================================

missing_images = []


for path in test_df["clahe_image_path"]:

    image_path = Path(
        str(path)
    )

    if not image_path.exists():

        missing_images.append(
            str(path)
        )


print()

print(
    "Missing test images:",
    len(missing_images)
)


if missing_images:

    print()

    print(
        "First missing images:"
    )

    for path in missing_images[:10]:

        print(path)

    raise FileNotFoundError(
        "Some PH2 test images are missing."
    )


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print()

print("=" * 75)

print(
    "PH2 TEST CLASS DISTRIBUTION"
)

print("=" * 75)


print(
    test_df[
        "binary_diagnosis"
    ].value_counts()
)


print()

print(
    "Binary labels:"
)


print(
    test_df[
        "binary_label"
    ].value_counts()
)


# ============================================================
# LOAD IMAGE
# ============================================================
#
# IMPORTANT:
#
# This preprocessing MUST match training.
#
# Training used:
#
#     image = tf.cast(
#         image,
#         tf.float32
#     )
#
# Training did NOT use:
#
#     image / 255.0
#
# Training did NOT use:
#
#     ImageNet mean/std normalization.
#
# The training model used:
#
#     EfficientNetV2B0(
#         include_preprocessing=True
#     )
#
# Therefore the input remains in the
# 0-255 pixel-value range.
#
# ============================================================

def load_image(path):

    # --------------------------------------------------------
    # READ IMAGE
    # --------------------------------------------------------

    image = tf.io.read_file(
        path
    )


    # --------------------------------------------------------
    # DECODE PNG
    # --------------------------------------------------------

    image = tf.image.decode_png(
        image,
        channels=3
    )


    # --------------------------------------------------------
    # RESIZE
    # --------------------------------------------------------

    image = tf.image.resize(
        image,
        IMAGE_SIZE,
        method="bilinear"
    )


    # --------------------------------------------------------
    # FLOAT32
    # --------------------------------------------------------

    image = tf.cast(
        image,
        tf.float32
    )


    # --------------------------------------------------------
    # IMPORTANT
    # --------------------------------------------------------
    #
    # DO NOT:
    #
    # image = image / 255.0
    #
    # DO NOT:
    #
    # ImageNet mean/std normalization
    #
    # EfficientNetV2B0 itself performs
    # the required preprocessing because
    # include_preprocessing=True.
    #
    # --------------------------------------------------------

    return image


# ============================================================
# CREATE TEST DATASET
# ============================================================

print()

print("=" * 75)

print(
    "CREATING PH2 TEST DATASET"
)

print("=" * 75)


test_paths = (

    test_df[
        "clahe_image_path"
    ]

    .astype(str)

    .values

)


test_labels = (

    test_df[
        "binary_label"
    ]

    .astype(np.float32)

    .values

)


test_dataset = (

    tf.data.Dataset.from_tensor_slices(

        (
            test_paths,
            test_labels
        )

    )

    .map(

        lambda path, label: (

            load_image(path),

            label

        ),

        num_parallel_calls=tf.data.AUTOTUNE

    )

    .batch(
        BATCH_SIZE
    )

    .prefetch(
        tf.data.AUTOTUNE
    )

)


print()

print(
    "Test dataset created successfully."
)


# ============================================================
# LOAD MODEL
# ============================================================

print()

print("=" * 75)

print(
    "LOADING PH2 EFFICIENTNETV2"
)

print("=" * 75)


model = tf.keras.models.load_model(
    MODEL_PATH
)


print()

print(
    "Model loaded successfully."
)


print()

print(
    "Model output shape:",
    model.output_shape
)


# ============================================================
# MODEL OUTPUT VALIDATION
# ============================================================

if model.output_shape[-1] != 1:

    raise ValueError(

        "Expected binary classification output "
        "with shape (None, 1), but got: "

        + str(model.output_shape)

    )


# ============================================================
# RUN PREDICTIONS
# ============================================================

print()

print("=" * 75)

print(
    "RUNNING PH2 TEST PREDICTION"
)

print("=" * 75)


raw_predictions = model.predict(

    test_dataset,

    verbose=1

)


print()

print(
    "Raw prediction shape:",
    raw_predictions.shape
)


# ============================================================
# FLATTEN PREDICTIONS
# ============================================================

y_true = np.asarray(
    test_labels
).astype(int)


y_probability = (

    raw_predictions

    .reshape(-1)

)


print()

print(
    "Final prediction count:",
    len(y_probability)
)


if len(y_probability) != len(y_true):

    raise ValueError(

        "Prediction count does not match "
        "test image count."

    )


print(
    "Prediction count matches test images."
)


# ============================================================
# CHECK PROBABILITY RANGE
# ============================================================

if np.any(
    y_probability < 0
) or np.any(
    y_probability > 1
):

    raise ValueError(
        "Model predictions are outside [0, 1]."
    )


# ============================================================
# THRESHOLD
# ============================================================

THRESHOLD = 0.5


y_pred = (

    y_probability >= THRESHOLD

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

    y_probability

)


pr_auc = average_precision_score(

    y_true,

    y_probability

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


true_negative = int(

    cm[0, 0]

)


false_positive = int(

    cm[0, 1]

)


false_negative = int(

    cm[1, 0]

)


true_positive = int(

    cm[1, 1]

)


# ============================================================
# SPECIFICITY
# ============================================================

if (

    true_negative
    + false_positive

) > 0:

    specificity = (

        true_negative

        /

        (

            true_negative
            + false_positive

        )

    )

else:

    specificity = 0.0


# ============================================================
# FINAL RESULTS
# ============================================================

print()

print("=" * 75)

print(
    "FINAL PH2 EFFICIENTNETV2 TEST RESULTS"
)

print("=" * 75)


print()

print(
    "Accuracy      :",
    f"{accuracy:.4f}"
)


print(
    "ROC-AUC       :",
    f"{roc_auc:.4f}"
)


print(
    "PR-AUC        :",
    f"{pr_auc:.4f}"
)


print(
    "Precision     :",
    f"{precision:.4f}"
)


print(
    "Sensitivity   :",
    f"{sensitivity:.4f}"
)


print(
    "Specificity   :",
    f"{specificity:.4f}"
)


print(
    "F1-score      :",
    f"{f1:.4f}"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print()

print("=" * 75)

print(
    "CONFUSION MATRIX"
)

print("=" * 75)


print()

print(
    "              Predicted"
)


print(
    "              Non-Mel   Melanoma"
)


print(

    f"Actual Non-Mel    "
    f"{true_negative:>3}       "
    f"{false_positive:>3}"

)


print(

    f"Actual Melanoma   "
    f"{false_negative:>3}       "
    f"{true_positive:>3}"

)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print()

print("=" * 75)

print(
    "CLASSIFICATION REPORT"
)

print("=" * 75)


print()


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

results = pd.DataFrame({

    "model": [

        "EfficientNetV2B0"

    ],

    "dataset": [

        "PH2 CLAHE Test"

    ],

    "test_images": [

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

        true_negative

    ],

    "false_positive": [

        false_positive

    ],

    "false_negative": [

        false_negative

    ],

    "true_positive": [

        true_positive

    ],

})


results.to_csv(

    RESULT_PATH,

    index=False

)


# ============================================================
# FINAL STATUS
# ============================================================

print()

print("=" * 75)

print(
    "FINAL PH2 TEST EVALUATION COMPLETED"
)

print("=" * 75)


print()

print(
    "Results saved:"
)

print(
    RESULT_PATH
)


print()

print("=" * 75)

print(
    "STATUS: PASS"
)

print(
    "PH2 EfficientNetV2 CLAHE test evaluation "
    "completed successfully."
)

print("=" * 75)