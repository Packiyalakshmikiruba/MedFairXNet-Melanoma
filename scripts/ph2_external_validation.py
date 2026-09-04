import sys
import glob
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score, roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score, confusion_matrix,
)

from medfairxnet_model import (
    IMAGE_SIZE, mc_dropout_predict, MCDropout,
    _spatial_avg_pool, _spatial_max_pool, _spatial_pool_output_shape,
)

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PH2_SPLITS_DIR = BASE_DIR / "data" / "processed" / "PH2" / "splits"

# Set this manually if auto-detection below picks the wrong file, e.g.:
# PH2_CSV_OVERRIDE = PH2_SPLITS_DIR / "validation_ph2_clahe.csv"
PH2_CSV_OVERRIDE = PH2_SPLITS_DIR / "test_ph2_clahe.csv"

THRESHOLD = 0.50  # same threshold applied to both models for a fair comparison
MC_SAMPLES = 20   # MedFairXNet MC-Dropout passes per image

MODELS = {
    "ResNet101": {
        "path": BASE_DIR / "models" / "melanoma_resnet101_final_clahe.keras",
        "mc_dropout": False,
    },

    "ConvNeXt": {
        "path": BASE_DIR / "models" / "melanoma_convnext_final_clahe.keras",
        "mc_dropout": False,
    },

    "MedFairXNet": {
        "path": BASE_DIR / "models" / "melanoma_medfairxnet_final_clahe.keras",
        "mc_dropout": True,
    },
}

# ============================================================
# LOCATE PH2 CSV
# ============================================================

def locate_ph2_csv():
    if PH2_CSV_OVERRIDE is not None:
        return Path(PH2_CSV_OVERRIDE)

    if not PH2_SPLITS_DIR.exists():
        print(f"!! PH2 splits directory not found: {PH2_SPLITS_DIR}")
        sys.exit(1)

    candidates = sorted(PH2_SPLITS_DIR.glob("*.csv"))
    if not candidates:
        print(f"!! No CSV files found in {PH2_SPLITS_DIR}")
        sys.exit(1)

    print("PH2 CSV candidates found:")
    for c in candidates:
        try:
            n_rows = len(pd.read_csv(c))
        except Exception:
            n_rows = "?"
        print(f"  - {c.name}  ({n_rows} rows)")

    # Prefer a CLAHE-processed file (matches the CLAHE preprocessing used
    # by every other final_clahe model), then prefer one closest to 30 rows.
    clahe_candidates = [c for c in candidates if "clahe" in c.name.lower()]
    pool = clahe_candidates if clahe_candidates else candidates

    def row_count(path):
        try:
            return len(pd.read_csv(path))
        except Exception:
            return 10**9

    chosen = min(pool, key=lambda c: abs(row_count(c) - 30))
    print(f"\nAuto-selected: {chosen.name}")
    print("(If this is wrong, set PH2_CSV_OVERRIDE at the top of this script.)\n")
    return chosen


# ============================================================
# IMAGE LOADING (same protocol as evaluate_medfairxnet_final_clahe.py -
# 224x224x3, ImageNet mean/std normalization, per the project's shared
# preprocessing pipeline used for all six models)
# ============================================================

def load_image(path):
    image = tf.io.read_file(path)
    # PH2 images are BMP (not JPEG like HAM10000/ISIC2018), so use the
    # generic decoder which handles BMP/PNG/JPEG/GIF automatically.
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    mean = tf.constant([0.485, 0.456, 0.406])
    std = tf.constant([0.229, 0.224, 0.225])
    return (image - mean) / std


def make_absolute_path(path):
    p = Path(str(path))
    return str(p) if p.is_absolute() else str((BASE_DIR / p).resolve())


# ============================================================
# LOAD PH2 DATA
# ============================================================

ph2_csv = locate_ph2_csv()
ph2_df = pd.read_csv(ph2_csv)

path_column = next(
    (c for c in ["image_path", "clahe_image_path", "filepath", "file_path", "path"]
     if c in ph2_df.columns), None,
)
label_column = next(
    (c for c in ["binary_label", "label", "target", "y"] if c in ph2_df.columns), None,
)

if path_column is None or label_column is None:
    print(f"!! Could not find path/label columns. Columns: {ph2_df.columns.tolist()}")
    sys.exit(1)

ph2_df[path_column] = ph2_df[path_column].apply(make_absolute_path)

missing = ph2_df[~ph2_df[path_column].apply(lambda x: Path(x).exists())]
if len(missing) > 0:
    print(f"!! {len(missing)} PH2 images are missing on disk. First few:")
    print(missing[path_column].head())
    sys.exit(1)

y_true = ph2_df[label_column].astype(int).to_numpy()
paths = ph2_df[path_column].tolist()

print("=" * 90)
print("MELONMA - PH2 INDEPENDENT EXTERNAL VALIDATION")
print("=" * 90)
print(f"PH2 CSV: {ph2_csv}")
print(f"PH2 images: {len(ph2_df)}")
print(f"Class distribution:\n{ph2_df[label_column].value_counts()}")
print("=" * 90)


# ============================================================
# EVALUATE ONE MODEL
# ============================================================

def evaluate_model(name, cfg):
    print(f"\n{'-' * 90}\nEvaluating: {name}\n{'-' * 90}")
    model_path = cfg["path"]
    if not model_path.exists():
        print(f"!! Model not found: {model_path}  -- skipping {name}")
        return None

    if cfg["mc_dropout"]:
        model = tf.keras.models.load_model(
            model_path, compile=False, safe_mode=False,
            custom_objects={
                "MCDropout": MCDropout,
                "_spatial_avg_pool": _spatial_avg_pool,
                "_spatial_max_pool": _spatial_max_pool,
                "_spatial_pool_output_shape": _spatial_pool_output_shape,
            },
        )
    else:
        model = tf.keras.models.load_model(model_path, compile=False, safe_mode=False)

    print(f"Model loaded. Total params: {model.count_params():,}")

    start = time.time()
    images = tf.stack([load_image(p) for p in paths])

    if cfg["mc_dropout"]:
        mean_p, std_p, ent = mc_dropout_predict(model, images, n_samples=MC_SAMPLES)
        y_prob = np.asarray(mean_p, dtype=float)
        y_std = np.asarray(std_p, dtype=float)
        y_entropy = np.asarray(ent, dtype=float)
    else:
        raw = model.predict(images, verbose=0)
        raw = np.asarray(raw, dtype=float)
        if raw.ndim == 2 and raw.shape[1] == 2:
            # 2-class softmax output: column 1 = melanoma probability
            y_prob = raw[:, 1]
        elif raw.ndim == 2 and raw.shape[1] == 1:
            y_prob = raw[:, 0]
        else:
            y_prob = raw.reshape(-1)
        y_std = np.full_like(y_prob, np.nan)
        y_entropy = np.full_like(y_prob, np.nan)

    elapsed = time.time() - start
    y_pred = (y_prob >= THRESHOLD).astype(int)

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

    print(f"Accuracy: {accuracy:.4f}  ROC-AUC: {roc_auc:.4f}  PR-AUC: {pr_auc:.4f}")
    print(f"Sensitivity: {sensitivity:.4f}  Specificity: {specificity:.4f}  F1: {f1:.4f}")
    print(f"TP={tp} FP={fp} TN={tn} FN={fn}  (time: {elapsed:.1f}s)")

    image_ids = (
        ph2_df["image_id"].astype(str).to_numpy()
        if "image_id" in ph2_df.columns else np.arange(len(ph2_df))
    )
    preds_df = pd.DataFrame({
        "image_id": image_ids,
        "true_label": y_true,
        "raw_prediction": y_prob,
        "mc_dropout_std": y_std,
        "predictive_entropy": y_entropy,
        "predicted_label": y_pred,
    })
    preds_out = RESULTS_DIR / f"ph2_external_validation_predictions_{name.lower()}.csv"
    preds_df.to_csv(preds_out, index=False)
    print(f"Saved predictions: {preds_out}")

    return {
        "model": name,
        "dataset": "PH2 (independent external validation)",
        "n_images": len(y_true),
        "threshold": THRESHOLD,
        "accuracy": round(accuracy, 4),
        "roc_auc": round(roc_auc, 4) if roc_auc == roc_auc else np.nan,
        "pr_auc": round(pr_auc, 4) if pr_auc == pr_auc else np.nan,
        "precision": round(precision, 4),
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "f1_score": round(f1, 4),
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
    }


# ============================================================
# RUN BOTH MODELS
# ============================================================

rows = []
for name, cfg in MODELS.items():
    result = evaluate_model(name, cfg)
    if result is not None:
        rows.append(result)

if not rows:
    print("\n!! No models evaluated - check model paths.")
    sys.exit(1)

comparison_df = pd.DataFrame(rows)

print("\n" + "=" * 90)
print("PH2 EXTERNAL VALIDATION - COMPARISON TABLE")
print("=" * 90)
print(comparison_df.to_string(index=False))

comparison_out = RESULTS_DIR / "ph2_external_validation_comparison.csv"
comparison_df.to_csv(comparison_out, index=False)

print("\n" + "=" * 90)
print("FILES SAVED")
print("=" * 90)
print("Comparison table:", comparison_out)
print("\nSTATUS: PASS")
print("\nNOTE: Report this table separately from the 999-image benchmark -")
print("label it 'Independent External Validation (PH2, n=30)' in the paper.")
