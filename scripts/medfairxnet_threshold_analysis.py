"""
MELONMA - Step 5: MedFairXNet Threshold Analysis
==================================================
Threshold-tuning done PROPERLY (no test-set leakage):
  1. Sweep thresholds 0.01-0.99 on the VALIDATION set predictions.
  2. Pick the optimal threshold via Youden's J statistic
     (maximises sensitivity + specificity - 1) and, separately, the
     F1-optimal threshold - both reported so you can choose per your
     paper's clinical framing.
  3. LOCK the chosen threshold.
  4. Apply it ONCE to the TEST set predictions and report metrics
     alongside the current default (0.50) for direct comparison.

Outputs:
  results\\medfairxnet_threshold_sweep_validation.csv   (full sweep table)
  results\\medfairxnet_threshold_comparison_test.csv    (0.50 vs tuned, on test)
  results\\medfairxnet_threshold_analysis_plots.png     (sweep curves + ROC)

Run from: C:\\Users\\HP\\Desktop\\Melonma
    python scripts\\medfairxnet_threshold_analysis.py

IMPORTANT: This script expects a VALIDATION predictions CSV (same format
as the test predictions CSV: true_label + raw_prediction columns). If your
validation predictions file has a different name, edit VAL_FILE below.
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
BASE_DIR = r"C:\Users\HP\Desktop\Melonma"
RESULTS_DIR = os.path.join(BASE_DIR, "results")

TEST_FILE = os.path.join(RESULTS_DIR, "medfairxnet_final_clahe_test_predictions.csv")

# Best-guess filename for the validation predictions file. If this doesn't
# exist, the script searches results\ for anything matching *val*predict*.csv
VAL_FILE = os.path.join(RESULTS_DIR, "medfairxnet_final_clahe_val_predictions.csv")

DEFAULT_THRESHOLD = 0.50

TRUE_LABEL_CANDIDATES = ["true_label", "y_true", "label", "target", "ground_truth"]
PRED_PROB_CANDIDATES = ["raw_prediction", "pred_prob", "y_pred_proba", "probability",
                         "y_prob", "prob_melanoma", "predicted_probability"]


def find_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    lower_map = {col.lower(): col for col in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def locate_val_file():
    if os.path.exists(VAL_FILE):
        return VAL_FILE
    print(f"!! Default validation file not found: {VAL_FILE}")
    print("   Searching results\\ for alternatives...")
    candidates = glob.glob(os.path.join(RESULTS_DIR, "*val*predict*.csv")) + \
                 glob.glob(os.path.join(RESULTS_DIR, "*validation*predict*.csv"))
    candidates = sorted(set(candidates))
    if not candidates:
        print("!! No validation predictions CSV found automatically.")
        print("   Edit VAL_FILE at the top of this script with the correct path and re-run.")
        sys.exit(1)
    print("   Found candidate(s):")
    for c in candidates:
        print(f"     - {c}")
    print(f"   Using: {candidates[0]}  (edit VAL_FILE if this is wrong)")
    return candidates[0]


def load_true_prob(filepath, label):
    df = pd.read_csv(filepath)
    true_col = find_column(df, TRUE_LABEL_CANDIDATES)
    prob_col = find_column(df, PRED_PROB_CANDIDATES)
    if true_col is None or prob_col is None:
        print(f"!! [{label}] Could not find required columns. "
              f"Available: {list(df.columns)}")
        print(f"   true_label candidates tried: {TRUE_LABEL_CANDIDATES}")
        print(f"   pred_prob candidates tried: {PRED_PROB_CANDIDATES}")
        sys.exit(1)
    y_true = df[true_col].astype(int).values
    y_prob = df[prob_col].astype(float).values
    print(f"[{label}] {filepath}")
    print(f"  rows={len(df)}  true_label='{true_col}'  pred_prob='{prob_col}'")
    return y_true, y_prob


def metrics_at_threshold(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    total = tp + tn + fp + fn

    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    accuracy = (tp + tn) / total if total else 0.0
    f1 = (2 * precision * sensitivity / (precision + sensitivity)
          if (precision + sensitivity) else 0.0)
    youden_j = sensitivity + specificity - 1
    balanced_acc = (sensitivity + specificity) / 2

    return {
        "threshold": round(threshold, 3),
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "accuracy": round(accuracy, 4),
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "precision": round(precision, 4),
        "f1_score": round(f1, 4),
        "youden_j": round(youden_j, 4),
        "balanced_accuracy": round(balanced_acc, 4),
    }


def main():
    print("=" * 80)
    print("MELONMA - STEP 5: MEDFAIRXNET THRESHOLD ANALYSIS")
    print("=" * 80)

    if not os.path.exists(TEST_FILE):
        print(f"!! TEST FILE NOT FOUND: {TEST_FILE}")
        sys.exit(1)

    val_file = locate_val_file()
    print("-" * 80)

    y_true_val, y_prob_val = load_true_prob(val_file, "VALIDATION")
    y_true_test, y_prob_test = load_true_prob(TEST_FILE, "TEST")
    print("-" * 80)

    # ------------------------------------------------------------------
    # Sweep thresholds on VALIDATION set only
    # ------------------------------------------------------------------
    thresholds = np.round(np.arange(0.01, 1.00, 0.01), 2)
    sweep_rows = [metrics_at_threshold(y_true_val, y_prob_val, t) for t in thresholds]
    sweep_df = pd.DataFrame(sweep_rows)

    sweep_out = os.path.join(RESULTS_DIR, "medfairxnet_threshold_sweep_validation.csv")
    sweep_df.to_csv(sweep_out, index=False)

    best_youden_row = sweep_df.loc[sweep_df["youden_j"].idxmax()]
    best_f1_row = sweep_df.loc[sweep_df["f1_score"].idxmax()]

    print("=" * 80)
    print("VALIDATION SET - THRESHOLD SWEEP RESULTS")
    print("=" * 80)
    print(f"Best threshold by Youden's J (sens+spec-1): {best_youden_row['threshold']:.2f} "
          f"(J={best_youden_row['youden_j']:.4f}, "
          f"sens={best_youden_row['sensitivity']:.4f}, spec={best_youden_row['specificity']:.4f})")
    print(f"Best threshold by F1-score:                 {best_f1_row['threshold']:.2f} "
          f"(F1={best_f1_row['f1_score']:.4f}, "
          f"sens={best_f1_row['sensitivity']:.4f}, spec={best_f1_row['specificity']:.4f})")

    # ------------------------------------------------------------------
    # LOCK the Youden-optimal threshold (standard for imbalanced medical
    # classification where both sensitivity and specificity matter) and
    # apply it ONCE to the test set. F1-optimal also reported for comparison.
    # ------------------------------------------------------------------
    locked_threshold = float(best_youden_row["threshold"])

    print("=" * 80)
    print(f"LOCKED THRESHOLD (Youden's J, selected on validation set only): {locked_threshold:.2f}")
    print("=" * 80)

    default_test = metrics_at_threshold(y_true_test, y_prob_test, DEFAULT_THRESHOLD)
    tuned_test = metrics_at_threshold(y_true_test, y_prob_test, locked_threshold)
    f1_tuned_test = metrics_at_threshold(y_true_test, y_prob_test, float(best_f1_row["threshold"]))

    comparison_df = pd.DataFrame([
        {"variant": f"Default (t={DEFAULT_THRESHOLD:.2f})", **default_test},
        {"variant": f"Youden-tuned (t={locked_threshold:.2f}, locked on val)", **tuned_test},
        {"variant": f"F1-tuned (t={best_f1_row['threshold']:.2f}, locked on val)", **f1_tuned_test},
    ])

    print("=" * 80)
    print("TEST SET - THRESHOLD COMPARISON (evaluated ONCE, threshold locked from validation)")
    print("=" * 80)
    print(comparison_df.to_string(index=False))

    comparison_out = os.path.join(RESULTS_DIR, "medfairxnet_threshold_comparison_test.csv")
    comparison_df.to_csv(comparison_out, index=False)

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel 1: sensitivity/specificity/F1/Youden vs threshold (validation)
    ax = axes[0]
    ax.plot(sweep_df["threshold"], sweep_df["sensitivity"], label="Sensitivity", color="#1D9E75")
    ax.plot(sweep_df["threshold"], sweep_df["specificity"], label="Specificity", color="#378ADD")
    ax.plot(sweep_df["threshold"], sweep_df["f1_score"], label="F1-score", color="#D85A30")
    ax.plot(sweep_df["threshold"], sweep_df["youden_j"], label="Youden's J", color="#7F77DD",
            linestyle="--")
    ax.axvline(locked_threshold, color="#0F6E56", linestyle=":", linewidth=1.5,
               label=f"Locked t={locked_threshold:.2f}")
    ax.axvline(DEFAULT_THRESHOLD, color="gray", linestyle=":", linewidth=1,
               label=f"Default t={DEFAULT_THRESHOLD:.2f}")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Metric value")
    ax.set_title("Validation set: metrics vs threshold")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1)

    # Panel 2: ROC curve (test set) with default & locked threshold points marked
    ax = axes[1]
    roc_thresholds = np.round(np.arange(0.0, 1.01, 0.01), 2)
    tprs, fprs = [], []
    for t in roc_thresholds:
        m = metrics_at_threshold(y_true_test, y_prob_test, t)
        tprs.append(m["sensitivity"])
        fprs.append(1 - m["specificity"])
    ax.plot(fprs, tprs, color="#7F77DD", label="ROC curve (test)")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1, label="Chance")
    ax.scatter([1 - default_test["specificity"]], [default_test["sensitivity"]],
               color="gray", zorder=5, label=f"Default t={DEFAULT_THRESHOLD:.2f}")
    ax.scatter([1 - tuned_test["specificity"]], [tuned_test["sensitivity"]],
               color="#0F6E56", zorder=5, label=f"Locked t={locked_threshold:.2f}")
    ax.set_xlabel("1 - Specificity (FPR)")
    ax.set_ylabel("Sensitivity (TPR)")
    ax.set_title("Test set: ROC curve with operating points")
    ax.legend(fontsize=8, loc="lower right")

    fig.suptitle("MedFairXNet - Threshold Analysis", fontsize=15, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    plot_out = os.path.join(RESULTS_DIR, "medfairxnet_threshold_analysis_plots.png")
    plt.savefig(plot_out, dpi=300, bbox_inches="tight")
    plt.close()

    print("=" * 80)
    print("FILES SAVED")
    print("=" * 80)
    print("Validation sweep CSV:  ", sweep_out)
    print("Test comparison CSV:   ", comparison_out)
    print("Plots:                 ", plot_out)
    print("=" * 80)
    print("STATUS: PASS")
    print("=" * 80)


if __name__ == "__main__":
    main()