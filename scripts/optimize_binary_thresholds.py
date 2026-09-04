from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# ============================================================
# MELONMA - BINARY THRESHOLD OPTIMIZATION
# FIXED VERSION FOR WINDOWS PATH / TENSORFLOW READFILE ERROR
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

VAL_CSV = (
    BASE_DIR
    / "data"
    / "processed"
    / "final_binary_clahe_splits"
    / "validation_final_clahe.csv"
)

MODEL_PATHS = {
    "ResNet101": BASE_DIR / "models" / "melanoma_resnet101_final_clahe.keras",
    "DenseNet121": BASE_DIR / "models" / "melanoma_densenet121_final_clahe.keras",
    "EfficientNetV2": BASE_DIR / "models" / "melanoma_efficientnetv2_final_clahe.keras",
}

OUTPUT_DIR = BASE_DIR / "results" / "threshold_optimization"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 8


# ============================================================
# HEADER
# ============================================================

print("=" * 80)
print("MELONMA - BINARY THRESHOLD OPTIMIZATION")
print("=" * 80)

print("\nValidation CSV:")
print(VAL_CSV)

if not VAL_CSV.exists():
    raise FileNotFoundError(f"Validation CSV not found:\n{VAL_CSV}")


# ============================================================
# LOAD VALIDATION DATA
# ============================================================

df = pd.read_csv(VAL_CSV)

print("\n" + "=" * 80)
print("VALIDATION DATA")
print("=" * 80)

print("Records:", len(df))

if "image_path" not in df.columns:
    raise ValueError(
        "Required column 'image_path' not found in validation CSV.\n"
        f"Available columns: {list(df.columns)}"
    )

if "binary_label" not in df.columns:
    raise ValueError(
        "Required column 'binary_label' not found in validation CSV."
    )

# ------------------------------------------------------------
# Convert paths to absolute Windows paths
# ------------------------------------------------------------

def resolve_image_path(value):
    """
    Convert CSV image path into an absolute path.

    Handles:
    - absolute Windows paths
    - relative paths
    - paths relative to project root
    """

    p = Path(str(value))

    if p.is_absolute():
        return str(p.resolve())

    return str((BASE_DIR / p).resolve())


df["resolved_image_path"] = df["image_path"].apply(resolve_image_path)

# Check paths BEFORE TensorFlow starts
missing = [
    p for p in df["resolved_image_path"]
    if not Path(p).exists()
]

print("Missing validation images:", len(missing))

if missing:
    print("\nFirst missing files:")
    for p in missing[:10]:
        print(p)

    raise FileNotFoundError(
        f"{len(missing)} validation image files are missing."
    )

y_true = df["binary_label"].astype(int).to_numpy()

print("\nClass distribution:")
print(
    df["binary_label"]
    .value_counts()
    .sort_index()
)

print("\nValidation labels:")
print("Non-melanoma:", int((y_true == 0).sum()))
print("Melanoma    :", int((y_true == 1).sum()))


# ============================================================
# IMAGE LOADER
# ============================================================

def load_image(path):
    """
    Load one image using Python path string first.
    This avoids the Tensor('args_0') path problem.
    """

    path = str(path)

    image_bytes = tf.io.read_file(path)

    image = tf.io.decode_image(
        image_bytes,
        channels=3,
        expand_animations=False,
    )

    image.set_shape([None, None, 3])

    image = tf.image.resize(
        image,
        IMAGE_SIZE,
        method=tf.image.ResizeMethod.BILINEAR,
    )

    image = tf.cast(image, tf.float32) / 255.0

    return image


# ============================================================
# SAFE PYTHON IMAGE LOADING
# ============================================================

def create_numpy_dataset(paths):
    """
    Loads images into NumPy arrays first.

    This is intentionally used instead of:
        Dataset.map(load_image)

    because the previous implementation caused TensorFlow
    to pass symbolic tensors as Windows filenames.
    """

    images = []

    total = len(paths)

    print("\nLoading validation images...")

    for i, path in enumerate(paths, start=1):

        try:
            image = load_image(path)

            # Execute immediately as NumPy
            image = image.numpy()

            images.append(image)

        except Exception as e:
            raise RuntimeError(
                f"\nFailed to load image:\n{path}\n"
                f"Error: {e}"
            ) from e

        if i % 100 == 0 or i == total:
            print(f"Loaded: {i}/{total}")

    return np.asarray(images, dtype=np.float32)


# ============================================================
# LOAD ALL VALIDATION IMAGES
# ============================================================

X_val = create_numpy_dataset(
    df["resolved_image_path"].tolist()
)

print("\nImage array shape:", X_val.shape)
print("Label array shape:", y_true.shape)

if len(X_val) != len(y_true):
    raise ValueError(
        "Image count and label count do not match."
    )


# ============================================================
# MODEL PREDICTION
# ============================================================

def get_melanoma_probability(model, predictions):

    predictions = np.asarray(predictions)

    print("Raw prediction shape:", predictions.shape)

    # --------------------------------------------------------
    # Binary sigmoid output
    # --------------------------------------------------------

    if predictions.ndim == 1:
        return predictions.astype(float)

    if predictions.ndim == 2 and predictions.shape[1] == 1:
        return predictions[:, 0].astype(float)

    # --------------------------------------------------------
    # Two-class softmax output
    # --------------------------------------------------------

    if predictions.ndim == 2 and predictions.shape[1] == 2:
        return predictions[:, 1].astype(float)

    raise ValueError(
        f"Unsupported model output shape: {predictions.shape}"
    )


# ============================================================
# THRESHOLD METRICS
# ============================================================

def calculate_metrics(y_true, probabilities, threshold):

    y_pred = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    sensitivity = recall

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "f1_score": float(f1),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


# ============================================================
# OPTIMIZATION
# ============================================================

all_results = []

thresholds = np.arange(
    0.10,
    0.91,
    0.01,
)


# ============================================================
# PROCESS EACH MODEL
# ============================================================

for model_name, model_path in MODEL_PATHS.items():

    print("\n" + "=" * 80)
    print(f"PROCESSING MODEL: {model_name}")
    print("=" * 80)

    print("\nModel path:")
    print(model_path)

    if not model_path.exists():
        print("\nWARNING: Model not found. Skipping.")
        continue

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print("\nLoading model...")

    model = tf.keras.models.load_model(
        model_path,
        compile=False,
    )

    print("Model loaded successfully.")
    print("Model output shape:", model.output_shape)

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    print("\nRunning validation prediction...")

    predictions = model.predict(
        X_val,
        batch_size=BATCH_SIZE,
        verbose=1,
    )

    probabilities = get_melanoma_probability(
        model,
        predictions,
    )

    if len(probabilities) != len(y_true):
        raise ValueError(
            "Prediction count does not match validation labels."
        )

    # --------------------------------------------------------
    # Global ranking metrics
    # --------------------------------------------------------

    roc_auc = roc_auc_score(
        y_true,
        probabilities,
    )

    pr_auc = average_precision_score(
        y_true,
        probabilities,
    )

    print("\nROC-AUC:", round(roc_auc, 4))
    print("PR-AUC :", round(pr_auc, 4))

    # --------------------------------------------------------
    # Search thresholds
    # --------------------------------------------------------

    model_results = []

    for threshold in thresholds:

        metrics = calculate_metrics(
            y_true,
            probabilities,
            threshold,
        )

        metrics["model"] = model_name
        metrics["roc_auc"] = float(roc_auc)
        metrics["pr_auc"] = float(pr_auc)

        model_results.append(metrics)
        all_results.append(metrics)

    results_df = pd.DataFrame(model_results)

    # --------------------------------------------------------
    # Best threshold by F1
    # --------------------------------------------------------

    best_f1_row = results_df.loc[
        results_df["f1_score"].idxmax()
    ]

    # --------------------------------------------------------
    # Best threshold by balanced sensitivity/specificity
    # --------------------------------------------------------

    results_df["youden_j"] = (
        results_df["sensitivity"]
        + results_df["specificity"]
        - 1.0
    )

    best_youden_row = results_df.loc[
        results_df["youden_j"].idxmax()
    ]

    # --------------------------------------------------------
    # Best threshold by accuracy
    # --------------------------------------------------------

    best_accuracy_row = results_df.loc[
        results_df["accuracy"].idxmax()
    ]

    print("\n" + "-" * 80)
    print(f"{model_name} - THRESHOLD RESULTS")
    print("-" * 80)

    print("\nBest F1 threshold:")
    print(
        f"Threshold    : {best_f1_row['threshold']:.2f}"
    )
    print(
        f"Accuracy     : {best_f1_row['accuracy']:.4f}"
    )
    print(
        f"Precision    : {best_f1_row['precision']:.4f}"
    )
    print(
        f"Sensitivity  : {best_f1_row['sensitivity']:.4f}"
    )
    print(
        f"Specificity  : {best_f1_row['specificity']:.4f}"
    )
    print(
        f"F1-score     : {best_f1_row['f1_score']:.4f}"
    )

    print("\nBest Youden-J threshold:")
    print(
        f"Threshold    : {best_youden_row['threshold']:.2f}"
    )
    print(
        f"Sensitivity  : {best_youden_row['sensitivity']:.4f}"
    )
    print(
        f"Specificity  : {best_youden_row['specificity']:.4f}"
    )

    print("\nBest Accuracy threshold:")
    print(
        f"Threshold    : {best_accuracy_row['threshold']:.2f}"
    )
    print(
        f"Accuracy     : {best_accuracy_row['accuracy']:.4f}"
    )

    # --------------------------------------------------------
    # Save model-specific results
    # --------------------------------------------------------

    model_output = (
        OUTPUT_DIR
        / f"{model_name.lower()}_thresholds.csv"
    )

    results_df.to_csv(
        model_output,
        index=False,
    )

    print("\nThreshold table saved:")
    print(model_output)

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    summary = pd.DataFrame([
        {
            "model": model_name,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "best_f1_threshold": best_f1_row["threshold"],
            "best_f1": best_f1_row["f1_score"],
            "best_f1_accuracy": best_f1_row["accuracy"],
            "best_f1_precision": best_f1_row["precision"],
            "best_f1_sensitivity": best_f1_row["sensitivity"],
            "best_f1_specificity": best_f1_row["specificity"],
            "best_youden_threshold": best_youden_row["threshold"],
            "best_accuracy_threshold": best_accuracy_row["threshold"],
        }
    ])

    summary_output = (
        OUTPUT_DIR
        / f"{model_name.lower()}_threshold_summary.csv"
    )

    summary.to_csv(
        summary_output,
        index=False,
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

if not all_results:
    raise RuntimeError(
        "No model was successfully evaluated."
    )

all_results_df = pd.DataFrame(
    all_results
)

all_output = (
    OUTPUT_DIR
    / "all_model_threshold_results.csv"
)

all_results_df.to_csv(
    all_output,
    index=False,
)

print("\n" + "=" * 80)
print("FINAL THRESHOLD OPTIMIZATION SUMMARY")
print("=" * 80)

summary_rows = []

for model_name in MODEL_PATHS.keys():

    model_df = all_results_df[
        all_results_df["model"] == model_name
    ]

    if model_df.empty:
        continue

    best = model_df.loc[
        model_df["f1_score"].idxmax()
    ]

    summary_rows.append({
        "model": model_name,
        "roc_auc": best["roc_auc"],
        "pr_auc": best["pr_auc"],
        "optimal_threshold": best["threshold"],
        "accuracy": best["accuracy"],
        "precision": best["precision"],
        "sensitivity": best["sensitivity"],
        "specificity": best["specificity"],
        "f1_score": best["f1_score"],
    })

final_summary = pd.DataFrame(
    summary_rows
)

print(
    final_summary.to_string(
        index=False
    )
)

final_summary_path = (
    OUTPUT_DIR
    / "final_threshold_summary.csv"
)

final_summary.to_csv(
    final_summary_path,
    index=False,
)

print("\nAll threshold results saved:")
print(all_output)

print("\nFinal summary saved:")
print(final_summary_path)

print("\n" + "=" * 80)
print("STATUS: PASS")
print("Binary threshold optimization completed successfully.")
print("=" * 80)
