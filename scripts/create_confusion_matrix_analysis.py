"""
MELONMA - Step 5: Confusion Matrix Consolidated Analysis
==========================================================
Loads each model's test evaluation CSV, computes the confusion matrix
(TP/FP/TN/FN + derived rates), writes one consolidated summary CSV,
and renders a 2x3 grid figure with all 6 models' confusion matrices.

Run from: C:\\Users\\HP\\Desktop\\Melonma
    python scripts\\create_confusion_matrix_analysis.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# CONFIG - same 6 models / same file pattern as create_master_comparison.py
# ----------------------------------------------------------------------
BASE_DIR = r"C:\Users\HP\Desktop\Melonma"
RESULTS_DIR = os.path.join(BASE_DIR, "results")

MODELS = {
    "EfficientNetV2": "efficientnetv2_final_clahe_test_evaluation.csv",
    "ConvNeXtTiny":   "convnext_final_clahe_test_evaluation.csv",
    "DenseNet121":    "densenet121_final_clahe_test_evaluation.csv",
    "ResNet101":      "resnet101_final_clahe_test_evaluation.csv",
    "Swin-Tiny":      "swin_final_clahe_test_evaluation.csv",
    "MedFairXNet":    "medfairxnet_final_clahe_test_evaluation.csv",
}

# These CSVs are already aggregated (one row per model) with the confusion
# matrix counts precomputed — no raw predictions to threshold.
CM_COLUMNS = ["true_negative", "false_positive", "false_negative", "true_positive"]


def get_confusion_counts(df, model_name):
    missing = [c for c in CM_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"[{model_name}] Missing confusion-matrix columns {missing}. "
            f"Available columns: {list(df.columns)}"
        )
    row = df.iloc[0]
    tn = int(row["true_negative"])
    fp = int(row["false_positive"])
    fn = int(row["false_negative"])
    tp = int(row["true_positive"])
    return tn, fp, fn, tp


def main():
    print("=" * 80)
    print("MELONMA - STEP 5: CONFUSION MATRIX CONSOLIDATED ANALYSIS")
    print("=" * 80)

    rows = []
    cm_store = {}

    for model_name, filename in MODELS.items():
        filepath = os.path.join(RESULTS_DIR, filename)
        print(f"Loading: {model_name}")
        print(f"File: {filepath}")
        if not os.path.exists(filepath):
            print(f"  !! FILE NOT FOUND - skipping {model_name}")
            continue

        df = pd.read_csv(filepath)
        try:
            tn, fp, fn, tp = get_confusion_counts(df, model_name)
        except ValueError as e:
            print(f"  !! {e}")
            continue

        total = tn + fp + fn + tp

        accuracy = (tp + tn) / total if total else np.nan
        sensitivity = tp / (tp + fn) if (tp + fn) else np.nan   # recall / TPR
        specificity = tn / (tn + fp) if (tn + fp) else np.nan   # TNR
        precision = tp / (tp + fp) if (tp + fp) else np.nan
        npv = tn / (tn + fn) if (tn + fn) else np.nan
        f1 = (2 * precision * sensitivity / (precision + sensitivity)
              if (precision and sensitivity and (precision + sensitivity)) else np.nan)

        cm_store[model_name] = np.array([[tn, fp], [fn, tp]])

        rows.append({
            "model": model_name,
            "TP": tp, "FP": fp, "TN": tn, "FN": fn,
            "total_samples": total,
            "accuracy": round(accuracy, 4),
            "sensitivity_recall": round(sensitivity, 4),
            "specificity": round(specificity, 4),
            "precision_ppv": round(precision, 4),
            "npv": round(npv, 4),
            "f1_score": round(f1, 4) if f1 == f1 else np.nan,
        })
        print("  Loaded and confusion matrix computed.")

    if not rows:
        print("No models processed - check file paths / column names.")
        sys.exit(1)

    summary_df = pd.DataFrame(rows).sort_values("f1_score", ascending=False).reset_index(drop=True)
    summary_df.insert(0, "rank_by_f1", summary_df.index + 1)

    print("=" * 80)
    print("CONFUSION MATRIX SUMMARY (all models)")
    print("=" * 80)
    print(summary_df.to_string(index=False))

    # ------------------------------------------------------------------
    # Save consolidated CSV
    # ------------------------------------------------------------------
    csv_out = os.path.join(RESULTS_DIR, "master_confusion_matrix_summary_final_clahe.csv")
    summary_df.to_csv(csv_out, index=False)

    # ------------------------------------------------------------------
    # 2x3 grid figure
    # ------------------------------------------------------------------
    model_order = list(cm_store.keys())
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, model_name in enumerate(model_order):
        ax = axes[idx]
        cm = cm_store[model_name]
        im = ax.imshow(cm, cmap="Blues")

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred: Benign", "Pred: Melanoma"])
        ax.set_yticklabels(["True: Benign", "True: Melanoma"])
        ax.set_title(model_name, fontsize=12, fontweight="bold")

        thresh = cm.max() / 2.0
        for i in range(2):
            for j in range(2):
                ax.text(j, i, format(cm[i, j], "d"),
                         ha="center", va="center", fontsize=13,
                         color="white" if cm[i, j] > thresh else "black")

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # hide unused subplots if fewer than 6 models loaded
    for idx in range(len(model_order), 6):
        axes[idx].axis("off")

    fig.suptitle("Confusion Matrices - All Models (Final CLAHE Test Set)",
                  fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    png_out = os.path.join(RESULTS_DIR, "master_confusion_matrix_grid.png")
    plt.savefig(png_out, dpi=300, bbox_inches="tight")
    plt.close()

    print("=" * 80)
    print("FILES SAVED")
    print("=" * 80)
    print("Confusion matrix summary CSV:")
    print(csv_out)
    print("Confusion matrix grid figure:")
    print(png_out)
    print("=" * 80)
    print("STATUS: PASS")
    print("=" * 80)


if __name__ == "__main__":
    main()