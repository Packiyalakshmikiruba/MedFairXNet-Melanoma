"""
MELONMA - ROC & PR CURVE COMPARISON (Steps 6 + 7)
============================================================
Draws ROC and PR curves for every model that has a saved
*_test_predictions.csv with raw probabilities. Any model missing
that file is skipped and clearly reported -- nothing is invented.

Run from the project's scripts/ folder:
    python generate_roc_pr_curves.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"

ROC_FIG_PATH = RESULTS_DIR / "roc_curves_comparison.png"
PR_FIG_PATH = RESULTS_DIR / "pr_curves_comparison.png"

# filename -> model display name. Add more here once other models'
# prediction CSVs exist.
PREDICTION_FILES = {
    "ResNet101": "resnet101_final_clahe_test_predictions.csv",
    "Swin-Tiny": "swin_final_clahe_test_predictions.csv",
    "DenseNet121": "densenet121_final_clahe_test_predictions.csv",
    "EfficientNetV2": "efficientnetv2_final_clahe/efficientnetv2_final_clahe_predictions.csv",
    "ConvNeXtTiny": "convnext_final_clahe_test_predictions.csv",
    "MedFairXNet": "medfairxnet_final_clahe_test_predictions.csv",
}

LABEL_COL_CANDIDATES = ["true_label", "binary_label", "label", "y_true", "true_class"]
PROB_COL_CANDIDATES = [
    "raw_prediction", "melanoma_probability", "probability", "y_prob", "prob",
    "prediction_score", "predicted_probability", "score", "prediction",
]

print("=" * 80)
print("MELONMA - ROC & PR CURVE COMPARISON")
print("=" * 80)

loaded = {}
skipped = []

for model_name, filename in PREDICTION_FILES.items():
    path = RESULTS_DIR / filename
    print(f"\n{model_name}: {path}")

    if not path.exists():
        print("  STATUS: NOT FOUND -- skipping (no saved predictions for this model)")
        skipped.append(model_name)
        continue

    df = pd.read_csv(path)

    label_col = next((c for c in LABEL_COL_CANDIDATES if c in df.columns), None)

    # Pick the first probability-column candidate that is actually numeric
    # and inside [0, 1] (skips columns that look like probabilities by name
    # but are really text/class labels, e.g. some models' "prediction" column).
    prob_col = None
    for candidate in PROB_COL_CANDIDATES:
        if candidate not in df.columns:
            continue
        coerced = pd.to_numeric(df[candidate], errors="coerce")
        valid_fraction = coerced.notna().mean()
        in_range = coerced.dropna().between(0, 1).mean() if coerced.notna().any() else 0
        if valid_fraction > 0.99 and in_range > 0.99:
            prob_col = candidate
            break

    if label_col is None or prob_col is None:
        print(f"  STATUS: COULD NOT FIND usable label/probability columns. "
              f"Columns present: {df.columns.tolist()}")
        skipped.append(model_name)
        continue

    y_true = df[label_col]
    if y_true.dtype == object:
        y_true = y_true.map({"Melanoma": 1, "Non-melanoma": 0}).fillna(y_true)
    y_true = pd.to_numeric(y_true, errors="coerce")
    y_prob = pd.to_numeric(df[prob_col], errors="coerce")

    valid_mask = y_true.notna() & y_prob.notna()
    dropped = (~valid_mask).sum()
    if dropped > 0:
        print(f"  NOTE: dropping {dropped} row(s) with non-numeric label/probability")

    y_true = y_true[valid_mask].astype(int).to_numpy()
    y_prob = y_prob[valid_mask].to_numpy()

    loaded[model_name] = (y_true, y_prob)
    print(f"  STATUS: LOADED  n={len(y_true)}  (label_col='{label_col}', prob_col='{prob_col}')")

if not loaded:
    raise RuntimeError("No model prediction files could be loaded -- nothing to plot.")

if skipped:
    print("\n" + "=" * 80)
    print("MODELS SKIPPED (no usable prediction file found)")
    print("=" * 80)
    for m in skipped:
        print(f"  - {m}: needs a *_test_predictions.csv with per-image true label + probability")

# ------------------------------------------------------------------
# ROC curves
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 6))
for model_name, (y_true, y_prob) in loaded.items():
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.3f})", linewidth=2)

ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves — Final CLAHE Test Set")
ax.legend(loc="lower right", fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(ROC_FIG_PATH, dpi=200, bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------------------
# PR curves
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 6))
for model_name, (y_true, y_prob) in loaded.items():
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    ax.plot(recall, precision, label=f"{model_name} (AUC = {pr_auc:.3f})", linewidth=2)

baseline = float(np.mean(list(loaded.values())[0][0]))  # positive class prevalence
ax.axhline(baseline, linestyle="--", color="gray", label=f"Chance ({baseline:.3f})")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curves — Final CLAHE Test Set")
ax.legend(loc="upper right", fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(PR_FIG_PATH, dpi=200, bbox_inches="tight")
plt.close(fig)

print("\n" + "=" * 80)
print("FILES SAVED")
print("=" * 80)
print("ROC curves figure:", ROC_FIG_PATH)
print("PR curves figure:", PR_FIG_PATH)
print(f"\nModels plotted: {list(loaded.keys())}")
if skipped:
    print(f"Models skipped: {skipped}")
print("\nSTATUS: PASS")