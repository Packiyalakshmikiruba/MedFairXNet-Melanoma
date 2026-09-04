"""
MELONMA - CONSOLIDATED CONFUSION MATRIX ANALYSIS (Step 5)
============================================================
Reads the six *_final_clahe_test_evaluation.csv files (same files
final_model_comparison.py / create_master_comparison.py already use)
and builds:

    results/confusion_matrix_consolidated.csv
        One row per model: TN/FP/FN/TP + derived rates
        (TPR/Sensitivity, FPR, FNR, TNR/Specificity, FDR, NPV)

    results/confusion_matrix_grid.png
        A 2x3 grid of all six confusion matrices (raw counts),
        ready to drop into the paper as a single figure.

Run from the project's scripts/ folder:
    python generate_confusion_matrix_analysis.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"

OUTPUT_CSV = RESULTS_DIR / "confusion_matrix_consolidated.csv"
OUTPUT_FIG = RESULTS_DIR / "confusion_matrix_grid.png"

# Same file set / order as final_model_comparison.py, so this stays
# consistent with the rest of your results.
MODEL_FILES = {
    "ResNet101": "resnet101_final_clahe_test_evaluation.csv",
    "Swin-Tiny": "swin_final_clahe_test_evaluation.csv",
    "DenseNet121": "densenet121_final_clahe_test_evaluation.csv",
    "EfficientNetV2": "efficientnetv2_final_clahe_test_evaluation.csv",
    "ConvNeXtTiny": "convnext_final_clahe_test_evaluation.csv",
    "MedFairXNet": "medfairxnet_final_clahe_test_evaluation.csv",
}

print("=" * 80)
print("MELONMA - CONSOLIDATED CONFUSION MATRIX ANALYSIS")
print("=" * 80)

records = []
matrices = {}

for model_name, filename in MODEL_FILES.items():
    path = RESULTS_DIR / filename
    print(f"\n{model_name}: {path}")

    if not path.exists():
        print("  STATUS: FILE NOT FOUND -- skipping")
        continue

    df = pd.read_csv(path)
    row = df.iloc[0]

    required = ["true_negative", "false_positive", "false_negative", "true_positive"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"  STATUS: MISSING COLUMNS {missing} -- skipping")
        continue

    tn, fp, fn, tp = (int(row[c]) for c in required)
    total = tn + fp + fn + tp

    tpr = tp / (tp + fn) if (tp + fn) > 0 else np.nan   # sensitivity / recall
    tnr = tn / (tn + fp) if (tn + fp) > 0 else np.nan   # specificity
    fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
    fnr = fn / (fn + tp) if (fn + tp) > 0 else np.nan
    ppv = tp / (tp + fp) if (tp + fp) > 0 else np.nan   # precision
    npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan
    fdr = fp / (fp + tp) if (fp + tp) > 0 else np.nan

    print(f"  STATUS: LOADED  TN={tn} FP={fp} FN={fn} TP={tp} (n={total})")

    records.append({
        "model": model_name,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
        "total": total,
        "sensitivity_tpr": tpr,
        "specificity_tnr": tnr,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "precision_ppv": ppv,
        "negative_predictive_value": npv,
        "false_discovery_rate": fdr,
    })
    matrices[model_name] = np.array([[tn, fp], [fn, tp]])

if not records:
    raise RuntimeError("No model confusion matrices could be loaded.")

summary_df = pd.DataFrame(records)
summary_df.to_csv(OUTPUT_CSV, index=False)

print("\n" + "=" * 80)
print("CONSOLIDATED TABLE")
print("=" * 80)
print(
    summary_df[
        ["model", "true_negative", "false_positive", "false_negative", "true_positive",
         "sensitivity_tpr", "specificity_tnr"]
    ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
)

# ------------------------------------------------------------------
# Figure: 2x3 grid of confusion matrices
# ------------------------------------------------------------------
model_order = list(matrices.keys())
n_models = len(model_order)
n_cols = 3
n_rows = int(np.ceil(n_models / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows))
axes = np.array(axes).reshape(-1)

for ax, model_name in zip(axes, model_order):
    cm = matrices[model_name]
    im = ax.imshow(cm, cmap="Blues")

    ax.set_title(model_name, fontsize=13, fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Non-melanoma", "Melanoma"])
    ax.set_yticklabels(["Non-melanoma", "Melanoma"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    cm_max = cm.max()
    for i in range(2):
        for j in range(2):
            value = cm[i, j]
            color = "white" if value > cm_max / 2 else "black"
            ax.text(j, i, str(value), ha="center", va="center",
                     color=color, fontsize=14, fontweight="bold")

# Hide any unused subplot slots
for ax in axes[n_models:]:
    ax.axis("off")

fig.suptitle("Confusion Matrices — All Models (Final CLAHE Test Set)",
              fontsize=15, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(OUTPUT_FIG, dpi=200, bbox_inches="tight")
plt.close(fig)

print("\n" + "=" * 80)
print("FILES SAVED")
print("=" * 80)
print("Consolidated table:", OUTPUT_CSV)
print("Confusion matrix grid figure:", OUTPUT_FIG)
print("\nSTATUS: PASS")