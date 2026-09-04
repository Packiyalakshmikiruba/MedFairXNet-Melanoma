from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)


# =========================================================
# CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

TEST_CSV = (
    BASE_DIR
    / "data"
    / "processed"
    / "splits"
    / "test.csv"
)

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "skin_lesion_efficientnet.keras"
)

RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 7


# =========================================================
# LABEL MAPPING
# =========================================================

label_mapping = {
    "akiec": 0,
    "bcc": 1,
    "bkl": 2,
    "df": 3,
    "mel": 4,
    "nv": 5,
    "vasc": 6,
}

class_names = [
    "akiec",
    "bcc",
    "bkl",
    "df",
    "mel",
    "nv",
    "vasc",
]


# =========================================================
# START
# =========================================================

print("=" * 70)
print("HAM10000 EFFICIENTNETB0 TEST EVALUATION")
print("=" * 70)


# =========================================================
# CHECK MODEL
# =========================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found:\n{MODEL_PATH}"
    )

print("\nModel path:")
print(MODEL_PATH)


# =========================================================
# LOAD TEST CSV
# =========================================================

if not TEST_CSV.exists():
    raise FileNotFoundError(
        f"Test CSV not found:\n{TEST_CSV}"
    )

test_df = pd.read_csv(TEST_CSV)

print("\nTest records:", len(test_df))


# =========================================================
# LABEL CREATION
# =========================================================

test_df["label"] = test_df["dx"].map(label_mapping)

if test_df["label"].isna().any():
    raise ValueError(
        "Unknown diagnosis found in test dataset."
    )

test_df["label"] = test_df["label"].astype(int)


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

missing_test = test_df[
    ~test_df["image_path"].apply(
        lambda x: Path(x).exists()
    )
]

print(
    "Missing test images:",
    len(missing_test)
)

if len(missing_test) > 0:

    print("\nFirst missing images:")

    print(
        missing_test["image_path"]
        .head(10)
        .to_string(index=False)
    )

    raise FileNotFoundError(
        "Some test images are missing."
    )


# =========================================================
# DATASET
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


def create_test_dataset(df):

    paths = df["image_path"].values

    labels = df["label"].values

    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.batch(
        BATCH_SIZE
    )

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset


test_dataset = create_test_dataset(
    test_df
)


# =========================================================
# LOAD MODEL
# =========================================================

print("\nLoading trained model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully.")


# =========================================================
# MODEL EVALUATION
# =========================================================

print("\n" + "=" * 70)
print("MODEL EVALUATION")
print("=" * 70)

test_loss, test_accuracy = model.evaluate(
    test_dataset,
    verbose=1
)

print("\nTest Loss     :", round(test_loss, 4))
print("Test Accuracy :", round(test_accuracy, 4))
print(
    "Test Accuracy :",
    f"{test_accuracy * 100:.2f}%"
)


# =========================================================
# PREDICTIONS
# =========================================================

print("\nGenerating predictions...")

probabilities = model.predict(
    test_dataset,
    verbose=1
)

predicted_labels = np.argmax(
    probabilities,
    axis=1
)

true_labels = test_df["label"].values


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
    average="weighted",
    zero_division=0
)

recall = recall_score(
    true_labels,
    predicted_labels,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    true_labels,
    predicted_labels,
    average="weighted",
    zero_division=0
)


# =========================================================
# ROC-AUC
# =========================================================

try:

    roc_auc = roc_auc_score(
        true_labels,
        probabilities,
        multi_class="ovr",
        average="weighted"
    )

except ValueError:

    roc_auc = None


# =========================================================
# PRINT METRICS
# =========================================================

print("\n" + "=" * 70)
print("TEST METRICS")
print("=" * 70)

print(f"\nAccuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

if roc_auc is not None:

    print(f"ROC-AUC   : {roc_auc:.4f}")

else:

    print("ROC-AUC   : Could not be calculated")


# =========================================================
# CLASSIFICATION REPORT
# =========================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

report = classification_report(
    true_labels,
    predicted_labels,
    target_names=class_names,
    zero_division=0
)

print("\n")
print(report)


# =========================================================
# CONFUSION MATRIX
# =========================================================

cm = confusion_matrix(
    true_labels,
    predicted_labels
)

cm_df = pd.DataFrame(
    cm,
    index=class_names,
    columns=class_names
)

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print("\n")
print(cm_df)


# =========================================================
# SAVE CONFUSION MATRIX
# =========================================================

cm_path = RESULTS_DIR / "confusion_matrix.csv"

cm_df.to_csv(
    cm_path
)


# =========================================================
# SAVE METRICS
# =========================================================

metrics = {
    "accuracy": accuracy,
    "precision_weighted": precision,
    "recall_weighted": recall,
    "f1_weighted": f1,
    "test_loss": test_loss,
}

if roc_auc is not None:
    metrics["roc_auc_weighted_ovr"] = roc_auc

metrics_df = pd.DataFrame(
    [metrics]
)

metrics_path = RESULTS_DIR / "evaluation_metrics.csv"

metrics_df.to_csv(
    metrics_path,
    index=False
)


# =========================================================
# SAVE CLASSIFICATION REPORT
# =========================================================

report_dict = classification_report(
    true_labels,
    predicted_labels,
    target_names=class_names,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(
    report_dict
).transpose()

report_path = (
    RESULTS_DIR
    / "classification_report.csv"
)

report_df.to_csv(
    report_path
)


# =========================================================
# FINAL SUMMARY
# =========================================================

print("\n" + "=" * 70)
print("EVALUATION COMPLETED")
print("=" * 70)

print("\nResults saved:")

print(
    "\nMetrics:",
    metrics_path
)

print(
    "Confusion Matrix:",
    cm_path
)

print(
    "Classification Report:",
    report_path
)

print("\nEvaluation completed successfully.")