"""
MELONMA - RESNET101 TEST PREDICTIONS (per-sample, for McNemar's test)
========================================================================
Runs the trained ResNet101 model on the main 999-image test set
(test_final_clahe.csv - the SAME test set used for MedFairXNet/
Swin-Tiny/ConvNeXtTiny) and saves a per-sample predictions CSV with
the same schema as medfairxnet_final_clahe_test_predictions.csv, so
mcnemar_significance_test.py can load it directly.

No MC-Dropout here (ResNet101 is a standard deterministic classifier,
not the uncertainty-aware proposed model) - just one forward pass per
image at the standard 0.50 threshold.

Handles both possible output shapes:
  - single sigmoid unit -> probability directly
  - 2-unit softmax -> column 1 = melanoma probability

Outputs (results/):
    resnet101_final_clahe_test_predictions.csv

Run from the project's scripts/ folder:
    python scripts\\generate_resnet101_test_predictions.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score, roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score, confusion_matrix,
)

# Reuse IMAGE_SIZE from the shared preprocessing config so ResNet101 gets
# the exact same 224x224x3 ImageNet-normalized input as every other model.
from medfairxnet_model import IMAGE_SIZE

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

TEST_CSV = (
    BASE_DIR / "data" / "processed" / "final_binary_clahe_splits"
    / "test_final_clahe.csv"
)
MODEL_PATH = BASE_DIR / "models" / "melanoma_resnet101_final_clahe.keras"

RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PREDICTIONS_PATH = RESULTS_DIR / "resnet101_final_clahe_test_predictions.csv"
METRICS_PATH = RESULTS_DIR / "resnet101_final_clahe_test_predictions_metrics.csv"

BATCH_SIZE = 8
THRESHOLD = 0.50


def make_absolute_path(path):
    p = Path(str(path))
    return str(p) if p.is_absolute() else str((BASE_DIR / p).resolve())


def load_image(path):
    # Matches evaluate_resnet101_final_clahe.py exactly: decode + resize +
    # cast to float32 only - NO /255 scaling and NO ImageNet mean/std
    # normalization. (ResNet101's preprocessing, if any, is baked into the
    # model itself - unlike MedFairXNet/EfficientNetV2 which expect
    # externally-normalized input.)
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32)
    return image


def main():
    print("=" * 90)
    print("MELONMA - RESNET101 TEST PREDICTIONS (for McNemar's test)")
    print("=" * 90)
    print("\nTest CSV:", TEST_CSV)
    print("Model:", MODEL_PATH)

    if not TEST_CSV.exists():
        raise FileNotFoundError(f"Test CSV not found:\n{TEST_CSV}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"ResNet101 model not found:\n{MODEL_PATH}")

    test_df = pd.read_csv(TEST_CSV)

    path_column = next(
        (c for c in ["image_path", "clahe_image_path", "filepath", "file_path", "path"]
         if c in test_df.columns), None,
    )
    label_column = next(
        (c for c in ["binary_label", "label", "target", "y"] if c in test_df.columns), None,
    )
    if path_column is None or label_column is None:
        raise ValueError(f"Could not find path/label columns. Columns: {test_df.columns.tolist()}")

    test_df[path_column] = test_df[path_column].apply(make_absolute_path)
    missing = test_df[~test_df[path_column].apply(lambda x: Path(x).exists())]
    if len(missing) > 0:
        raise FileNotFoundError(f"{len(missing)} test images are missing.")

    y_true = test_df[label_column].astype(int).to_numpy()
    print(f"\nTest images: {len(test_df)}")
    print("Class distribution:\n", test_df[label_column].value_counts())

    print("\nLoading ResNet101 checkpoint...")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False, safe_mode=False)
    print(f"Model loaded. Total params: {model.count_params():,}")

    print("\nRunning prediction...")
    paths = test_df[path_column].tolist()
    all_probs = []

    for i in range(0, len(paths), BATCH_SIZE):
        batch_paths = paths[i:i + BATCH_SIZE]
        batch_images = tf.stack([load_image(p) for p in batch_paths])
        raw = model.predict(batch_images, verbose=0)
        raw = np.asarray(raw, dtype=float)

        if raw.ndim == 2 and raw.shape[1] == 2:
            probs = raw[:, 1]           # 2-class softmax -> melanoma column
        elif raw.ndim == 2 and raw.shape[1] == 1:
            probs = raw[:, 0]
        else:
            probs = raw.reshape(-1)

        all_probs.extend(probs.tolist())

        if (i // BATCH_SIZE + 1) % 10 == 0:
            print(f"Batch {i // BATCH_SIZE + 1}/{(len(paths) - 1) // BATCH_SIZE + 1}")

    y_prob = np.asarray(all_probs, dtype=float)
    y_pred = (y_prob >= THRESHOLD).astype(int)

    # ------------------------------------------------------------------
    # Sanity-check metrics against your existing evaluation CSV
    # (should match resnet101_final_clahe_test_evaluation.csv closely)
    # ------------------------------------------------------------------
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    sensitivity = recall_score(y_true, y_pred, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        roc_auc = np.nan
    try:
        pr_auc = average_precision_score(y_true, y_prob)
    except ValueError:
        pr_auc = np.nan

    print("\n" + "=" * 90)
    print("SANITY CHECK - should match resnet101_final_clahe_test_evaluation.csv")
    print("=" * 90)
    print(f"Accuracy: {accuracy:.4f}  ROC-AUC: {roc_auc:.4f}  PR-AUC: {pr_auc:.4f}")
    print(f"Sensitivity: {sensitivity:.4f}  Specificity: {specificity:.4f}  F1: {f1:.4f}")
    print(f"TP={tp} FP={fp} TN={tn} FN={fn}")
    print("(Expected from master comparison: accuracy=0.8088, roc_auc=0.8186, "
          "sensitivity=0.5980, specificity=0.8328, f1=0.3898)")

    pd.DataFrame([{
        "model": "ResNet101", "dataset": "Final CLAHE Test", "test_images": len(y_true),
        "threshold": THRESHOLD, "accuracy": accuracy, "roc_auc": roc_auc, "pr_auc": pr_auc,
        "precision": precision, "sensitivity": sensitivity, "specificity": specificity,
        "f1_score": f1, "true_negative": tn, "false_positive": fp,
        "false_negative": fn, "true_positive": tp,
    }]).to_csv(METRICS_PATH, index=False)

    # ------------------------------------------------------------------
    # Save predictions CSV (same schema as medfairxnet_..._test_predictions.csv)
    # ------------------------------------------------------------------
    image_ids = (
        test_df["image_id"].astype(str).to_numpy()
        if "image_id" in test_df.columns else np.arange(len(test_df))
    )
    predictions_df = pd.DataFrame({
        "image_id": image_ids,
        "true_label": y_true,
        "raw_prediction": y_prob,
        "predicted_label": y_pred,
        "true_class": ["Melanoma" if x == 1 else "Non-melanoma" for x in y_true],
        "predicted_class": ["Melanoma" if x == 1 else "Non-melanoma" for x in y_pred],
    })
    predictions_df.to_csv(PREDICTIONS_PATH, index=False)

    print("\n" + "=" * 90)
    print("FILES SAVED")
    print("=" * 90)
    print("Predictions:", PREDICTIONS_PATH)
    print("Metrics (sanity check):", METRICS_PATH)
    print("\nSTATUS: PASS")
    print("\nIf the sanity-check numbers above don't closely match the expected")
    print("values, the model's output format may differ from what this script")
    print("assumes - paste the mismatch here and it'll get fixed.")
    print("\nNext: add this file's path to PRED_FILES in mcnemar_significance_test.py")
    print('  "ResNet101": RESULTS_DIR / "resnet101_final_clahe_test_predictions.csv",')


if __name__ == "__main__":
    main()