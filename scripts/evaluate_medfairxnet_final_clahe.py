"""
MELONMA - MEDFAIRXNET FINAL CLAHE TEST EVALUATION
============================================================
Mirrors evaluate_efficientnetv2_final_clahe.py / evaluate_swin_final_clahe.py
so the output slots straight into your existing results/ folder and into
final_model_comparison.py.

Outputs (results/):
    medfairxnet_final_clahe_test_evaluation.csv   <- standard metrics
    medfairxnet_final_clahe_confusion_matrix.csv
    medfairxnet_final_clahe_classification_report.txt
    medfairxnet_final_clahe_test_predictions.csv  <- includes MC-Dropout
                                                      mean/std/entropy per image
    medfairxnet_final_clahe_fairness_by_subgroup.csv  <- only if a
                                                      dataset-source column
                                                      exists in the test CSV

Run from the project's scripts/ folder:
    python evaluate_medfairxnet_final_clahe.py
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score, roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score, confusion_matrix,
    classification_report,
)

from medfairxnet_model import (
    IMAGE_SIZE, mc_dropout_predict, MCDropout,
    _spatial_avg_pool, _spatial_max_pool, _spatial_pool_output_shape,
)


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

TEST_CSV = (
    BASE_DIR / "data" / "processed" / "final_binary_clahe_splits"
    / "test_final_clahe.csv"
)

MODEL_PATH = BASE_DIR / "models" / "melanoma_medfairxnet_final_clahe.keras"

RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

METRICS_PATH = RESULTS_DIR / "medfairxnet_final_clahe_test_evaluation.csv"
CONFUSION_PATH = RESULTS_DIR / "medfairxnet_final_clahe_confusion_matrix.csv"
REPORT_PATH = RESULTS_DIR / "medfairxnet_final_clahe_classification_report.txt"
PREDICTIONS_PATH = RESULTS_DIR / "medfairxnet_final_clahe_test_predictions.csv"
FAIRNESS_PATH = RESULTS_DIR / "medfairxnet_final_clahe_fairness_by_subgroup.csv"

BATCH_SIZE = 8
THRESHOLD = 0.5
MC_SAMPLES = 20  # forward passes per image for uncertainty estimate

SUBGROUP_CANDIDATES = ["dataset_source", "source", "dataset"]


# ============================================================
# HEADER / CHECKS
# ============================================================

print("=" * 90)
print("MELONMA - MEDFAIRXNET FINAL CLAHE TEST EVALUATION")
print("=" * 90)
print("\nTest CSV:", TEST_CSV)
print("Model:", MODEL_PATH)

if not TEST_CSV.exists():
    raise FileNotFoundError(f"Test CSV not found:\n{TEST_CSV}")
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"MedFairXNet model not found:\n{MODEL_PATH}\n"
        "Run train_medfairxnet_final_clahe.py first."
    )


# ============================================================
# LOAD TEST DATA
# ============================================================

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


def make_absolute_path(path):
    p = Path(str(path))
    return str(p) if p.is_absolute() else str((BASE_DIR / p).resolve())


test_df[path_column] = test_df[path_column].apply(make_absolute_path)

missing_images = test_df[~test_df[path_column].apply(lambda x: Path(x).exists())]
if len(missing_images) > 0:
    raise FileNotFoundError(f"{len(missing_images)} test images are missing.")

y_true = test_df[label_column].astype(int).to_numpy()
print("\nTest images:", len(test_df))
print("Class distribution:\n", test_df[label_column].value_counts())


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading MedFairXNet checkpoint...")
model = tf.keras.models.load_model(
    MODEL_PATH, compile=False, safe_mode=False,
    custom_objects={
        "MCDropout": MCDropout,
        "_spatial_avg_pool": _spatial_avg_pool,
        "_spatial_max_pool": _spatial_max_pool,
        "_spatial_pool_output_shape": _spatial_pool_output_shape,
    },
)
print("Model loaded. Total params:", f"{model.count_params():,}")


# ============================================================
# PREPROCESS + PREDICT (standard prediction + MC-Dropout uncertainty)
# ============================================================

def load_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    mean = tf.constant([0.485, 0.456, 0.406])
    std = tf.constant([0.229, 0.224, 0.225])
    return (image - mean) / std


print("\n" + "=" * 90)
print("RUNNING PREDICTION (MC-DROPOUT, N =", MC_SAMPLES, "PASSES PER IMAGE)")
print("=" * 90)

start = time.time()

paths = test_df[path_column].tolist()
mean_probs, std_probs, entropies = [], [], []

for i in range(0, len(paths), BATCH_SIZE):
    batch_paths = paths[i:i + BATCH_SIZE]
    batch_images = tf.stack([load_image(p) for p in batch_paths])

    mean_p, std_p, ent = mc_dropout_predict(model, batch_images, n_samples=MC_SAMPLES)

    mean_probs.extend(mean_p.tolist())
    std_probs.extend(std_p.tolist())
    entropies.extend(ent.tolist())

    if (i // BATCH_SIZE + 1) % 25 == 0:
        print(f"Batch {i // BATCH_SIZE + 1}/{(len(paths) - 1) // BATCH_SIZE + 1}")

evaluation_time = time.time() - start

y_prob = np.asarray(mean_probs, dtype=float)
y_std = np.asarray(std_probs, dtype=float)
y_entropy = np.asarray(entropies, dtype=float)
y_pred = (y_prob >= THRESHOLD).astype(int)


# ============================================================
# METRICS
# ============================================================

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

# calibration proxy: mean predictive entropy, split by correct/incorrect
mean_entropy_correct = float(y_entropy[y_pred == y_true].mean()) if (y_pred == y_true).any() else np.nan
mean_entropy_incorrect = float(y_entropy[y_pred != y_true].mean()) if (y_pred != y_true).any() else np.nan

print("\n" + "=" * 90)
print("FINAL MEDFAIRXNET TEST RESULTS")
print("=" * 90)
print(f"Test images   : {len(y_true)}")
print(f"Accuracy      : {accuracy:.6f}")
print(f"ROC-AUC       : {roc_auc:.6f}")
print(f"PR-AUC        : {pr_auc:.6f}")
print(f"Precision     : {precision:.6f}")
print(f"Sensitivity   : {sensitivity:.6f}")
print(f"Specificity   : {specificity:.6f}")
print(f"F1-score      : {f1:.6f}")
print(f"Mean predictive entropy (correct)   : {mean_entropy_correct:.6f}")
print(f"Mean predictive entropy (incorrect) : {mean_entropy_incorrect:.6f}")
print("  (higher entropy on incorrect predictions = model 'knows what it doesn't know')")

report = classification_report(
    y_true, y_pred, target_names=["Non-melanoma", "Melanoma"], zero_division=0,
)
print("\n" + report)


# ============================================================
# SAVE STANDARD OUTPUTS (same schema as the other 5 models,
# so final_model_comparison.py can read this file unchanged)
# ============================================================

metrics_df = pd.DataFrame([{
    "model": "MedFairXNet",
    "dataset": "Final CLAHE Test",
    "test_images": len(y_true),
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
    "mean_entropy_correct": mean_entropy_correct,
    "mean_entropy_incorrect": mean_entropy_incorrect,
    "mc_dropout_samples": MC_SAMPLES,
    "evaluation_time_seconds": evaluation_time,
}])
metrics_df.to_csv(METRICS_PATH, index=False)

cm_df = pd.DataFrame(
    [[tn, fp], [fn, tp]],
    columns=["Predicted_Non_Melanoma", "Predicted_Melanoma"],
    index=["Actual_Non_Melanoma", "Actual_Melanoma"],
)
cm_df.to_csv(CONFUSION_PATH)

with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write(report)

image_ids = (
    test_df["image_id"].astype(str).to_numpy()
    if "image_id" in test_df.columns else np.arange(len(test_df))
)

predictions_df = pd.DataFrame({
    "image_id": image_ids,
    "true_label": y_true,
    "raw_prediction": y_prob,
    "mc_dropout_std": y_std,
    "predictive_entropy": y_entropy,
    "predicted_label": y_pred,
    "true_class": ["Melanoma" if x == 1 else "Non-melanoma" for x in y_true],
    "predicted_class": ["Melanoma" if x == 1 else "Non-melanoma" for x in y_pred],
})
predictions_df.to_csv(PREDICTIONS_PATH, index=False)


# ============================================================
# OPTIONAL: FAIRNESS BREAKDOWN BY DATASET-SOURCE SUBGROUP
# (only runs if such a column genuinely exists in the test CSV —
#  no assumption is made about your data if it does not)
# ============================================================

subgroup_col = next((c for c in SUBGROUP_CANDIDATES if c in test_df.columns), None)

if subgroup_col is not None:
    print(f"\nFairness subgroup column detected: '{subgroup_col}'")
    rows = []
    for group_value, group_df in test_df.groupby(subgroup_col):
        idx = group_df.index
        g_true = y_true[idx]
        g_pred = y_pred[idx]
        g_prob = y_prob[idx]

        if len(np.unique(g_true)) < 2:
            g_auc = np.nan
        else:
            try:
                g_auc = roc_auc_score(g_true, g_prob)
            except ValueError:
                g_auc = np.nan

        g_tn, g_fp, g_fn, g_tp = confusion_matrix(g_true, g_pred, labels=[0, 1]).ravel()
        g_tpr = g_tp / (g_tp + g_fn) if (g_tp + g_fn) > 0 else np.nan
        g_fpr = g_fp / (g_fp + g_tn) if (g_fp + g_tn) > 0 else np.nan

        rows.append({
            "subgroup": group_value,
            "n_images": len(group_df),
            "accuracy": accuracy_score(g_true, g_pred),
            "roc_auc": g_auc,
            "tpr_sensitivity": g_tpr,
            "fpr": g_fpr,
        })

    fairness_df = pd.DataFrame(rows)
    fairness_df["tpr_gap_from_max"] = fairness_df["tpr_sensitivity"].max() - fairness_df["tpr_sensitivity"]
    fairness_df["fpr_gap_from_max"] = fairness_df["fpr"].max() - fairness_df["fpr"]
    fairness_df.to_csv(FAIRNESS_PATH, index=False)

    print("\nPer-subgroup fairness (equal-opportunity view):")
    print(fairness_df.to_string(index=False))
    print(f"\nSaved: {FAIRNESS_PATH}")
else:
    print(
        f"\nNo dataset-source column found in test CSV (looked for {SUBGROUP_CANDIDATES}). "
        "Skipping fairness-by-subgroup breakdown — add a 'dataset_source' column "
        "(ISIC2018 / HAM10000 / PH2) to the CSV to enable this."
    )


print("\n" + "=" * 90)
print("FILES SAVED")
print("=" * 90)
print("Metrics:", METRICS_PATH)
print("Confusion matrix:", CONFUSION_PATH)
print("Classification report:", REPORT_PATH)
print("Predictions (with uncertainty):", PREDICTIONS_PATH)
print(f"\nEvaluation time: {evaluation_time:.2f} seconds")
print("\nSTATUS: PASS")