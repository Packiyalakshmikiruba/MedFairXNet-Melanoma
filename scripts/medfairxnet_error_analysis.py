"""
MELONMA - Step 4: MedFairXNet Error Analysis
==============================================
Loads the per-sample predictions CSV for MedFairXNet, classifies every
sample as TP / TN / FP / FN, and breaks down prediction confidence,
predictive entropy, and uncertainty by correctness and by class
(melanoma vs non-melanoma).

Outputs:
  results\\medfairxnet_error_analysis_per_sample.csv   (every sample + category)
  results\\medfairxnet_error_analysis_summary.csv       (aggregated stats per category)
  results\\medfairxnet_error_analysis_plots.png         (4-panel diagnostic figure)

Run from: C:\\Users\\HP\\Desktop\\Melonma
    python scripts\\medfairxnet_error_analysis.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
BASE_DIR = r"C:\Users\HP\Desktop\Melonma"
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PRED_FILE = os.path.join(RESULTS_DIR, "medfairxnet_final_clahe_test_predictions.csv")

THRESHOLD = 0.50  # current locked threshold (Step 5 will revisit this)

# Candidate column names - script auto-detects whichever exist in your CSV
ID_CANDIDATES = ["image_id", "filename", "image", "file", "id", "image_name"]
TRUE_LABEL_CANDIDATES = ["y_true", "true_label", "label", "target", "ground_truth"]
PRED_PROB_CANDIDATES = ["y_pred_proba", "pred_prob", "probability", "y_prob",
                         "prob_melanoma", "predicted_probability", "melanoma_prob"]
PRED_LABEL_CANDIDATES = ["y_pred", "pred_label", "prediction", "predicted_label"]
ENTROPY_CANDIDATES = ["entropy", "predictive_entropy", "mean_entropy", "pred_entropy"]
UNCERTAINTY_CANDIDATES = ["uncertainty", "mc_dropout_std", "prob_std", "epistemic_uncertainty", "std"]


def find_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    lower_map = {col.lower(): col for col in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def main():
    print("=" * 80)
    print("MELONMA - STEP 4: MEDFAIRXNET ERROR ANALYSIS")
    print("=" * 80)

    if not os.path.exists(PRED_FILE):
        print(f"!! FILE NOT FOUND: {PRED_FILE}")
        print("   Check the filename/path and update PRED_FILE at the top of this script.")
        sys.exit(1)

    df = pd.read_csv(PRED_FILE)
    print(f"Loaded: {PRED_FILE}")
    print(f"Rows: {len(df)}")
    print(f"Columns found: {list(df.columns)}")
    print("-" * 80)

    # ------------------------------------------------------------------
    # Detect columns
    # ------------------------------------------------------------------
    id_col = find_column(df, ID_CANDIDATES)
    true_col = find_column(df, TRUE_LABEL_CANDIDATES)
    prob_col = find_column(df, PRED_PROB_CANDIDATES)
    pred_col = find_column(df, PRED_LABEL_CANDIDATES)
    entropy_col = find_column(df, ENTROPY_CANDIDATES)
    uncertainty_col = find_column(df, UNCERTAINTY_CANDIDATES)

    if true_col is None:
        print(f"!! Could not find a true-label column. Available columns: {list(df.columns)}")
        print("   Add the correct name to TRUE_LABEL_CANDIDATES at the top of this script.")
        sys.exit(1)

    if pred_col is None and prob_col is None:
        print(f"!! Could not find a predicted-label or predicted-probability column. "
              f"Available columns: {list(df.columns)}")
        print("   Add the correct name to PRED_PROB_CANDIDATES / PRED_LABEL_CANDIDATES.")
        sys.exit(1)

    print(f"Detected -> true_label: {true_col} | pred_prob: {prob_col} | "
          f"pred_label: {pred_col} | entropy: {entropy_col} | uncertainty: {uncertainty_col}")
    print("-" * 80)

    out = pd.DataFrame()
    out["sample_id"] = df[id_col] if id_col else np.arange(len(df))
    out["y_true"] = df[true_col].astype(int)

    if prob_col is not None:
        out["pred_prob"] = df[prob_col].astype(float)
        if pred_col is not None:
            out["y_pred"] = df[pred_col].astype(int)
        else:
            out["y_pred"] = (out["pred_prob"] >= THRESHOLD).astype(int)
        # confidence = distance from decision boundary, mapped to [0.5, 1]
        out["confidence"] = np.where(out["y_pred"] == 1, out["pred_prob"], 1 - out["pred_prob"])
    else:
        out["y_pred"] = df[pred_col].astype(int)
        out["pred_prob"] = np.nan
        out["confidence"] = np.nan

    if entropy_col is not None:
        out["entropy"] = df[entropy_col].astype(float)
    if uncertainty_col is not None:
        out["uncertainty"] = df[uncertainty_col].astype(float)

    # ------------------------------------------------------------------
    # Category: TP / TN / FP / FN
    # ------------------------------------------------------------------
    def categorize(row):
        if row["y_true"] == 1 and row["y_pred"] == 1:
            return "TP"
        if row["y_true"] == 0 and row["y_pred"] == 0:
            return "TN"
        if row["y_true"] == 0 and row["y_pred"] == 1:
            return "FP"
        return "FN"

    out["category"] = out.apply(categorize, axis=1)
    out["correct"] = out["category"].isin(["TP", "TN"])
    out["true_class"] = out["y_true"].map({1: "melanoma", 0: "non_melanoma"})

    # ------------------------------------------------------------------
    # Save per-sample CSV
    # ------------------------------------------------------------------
    per_sample_out = os.path.join(RESULTS_DIR, "medfairxnet_error_analysis_per_sample.csv")
    out.to_csv(per_sample_out, index=False)

    # ------------------------------------------------------------------
    # Summary by category (TP/TN/FP/FN)
    # ------------------------------------------------------------------
    agg_cols = {"sample_id": "count"}
    if "pred_prob" in out and out["pred_prob"].notna().any():
        agg_cols["pred_prob"] = "mean"
    if "confidence" in out and out["confidence"].notna().any():
        agg_cols["confidence"] = "mean"
    if "entropy" in out.columns:
        agg_cols["entropy"] = "mean"
    if "uncertainty" in out.columns:
        agg_cols["uncertainty"] = "mean"

    summary_by_cat = out.groupby("category").agg(agg_cols).rename(
        columns={"sample_id": "n_samples"}
    ).reset_index()

    print("=" * 80)
    print("SUMMARY BY CATEGORY (TP/TN/FP/FN)")
    print("=" * 80)
    print(summary_by_cat.to_string(index=False))

    # ------------------------------------------------------------------
    # Summary by correctness x class (melanoma vs non-melanoma errors)
    # ------------------------------------------------------------------
    print("=" * 80)
    print("MELANOMA vs NON-MELANOMA ERROR BREAKDOWN")
    print("=" * 80)
    class_summary = out.groupby(["true_class", "correct"]).agg(agg_cols).rename(
        columns={"sample_id": "n_samples"}
    ).reset_index()
    print(class_summary.to_string(index=False))

    # ------------------------------------------------------------------
    # High-confidence errors & low-confidence correct predictions
    # ------------------------------------------------------------------
    dangerous_errors = pd.DataFrame()
    safe_correct = pd.DataFrame()
    if out["confidence"].notna().any():
        errors = out[~out["correct"]].sort_values("confidence", ascending=False)
        dangerous_errors = errors.head(10)
        correct = out[out["correct"]].sort_values("confidence", ascending=True)
        safe_correct = correct.head(10)

        print("=" * 80)
        print("TOP 10 HIGH-CONFIDENCE ERRORS (model was confidently wrong)")
        print("=" * 80)
        cols_show = [c for c in ["sample_id", "true_class", "category", "pred_prob",
                                  "confidence", "entropy", "uncertainty"] if c in out.columns]
        print(dangerous_errors[cols_show].to_string(index=False))

        print("=" * 80)
        print("TOP 10 LOW-CONFIDENCE CORRECT PREDICTIONS (model was unsure but right)")
        print("=" * 80)
        print(safe_correct[cols_show].to_string(index=False))

    # ------------------------------------------------------------------
    # Save summary CSV (both breakdowns stacked)
    # ------------------------------------------------------------------
    summary_out = os.path.join(RESULTS_DIR, "medfairxnet_error_analysis_summary.csv")
    with open(summary_out, "w", newline="") as f:
        f.write("# Summary by TP/TN/FP/FN category\n")
        summary_by_cat.to_csv(f, index=False)
        f.write("\n# Summary by melanoma vs non-melanoma x correctness\n")
        class_summary.to_csv(f, index=False)
        if not dangerous_errors.empty:
            f.write("\n# Top 10 high-confidence errors\n")
            dangerous_errors[cols_show].to_csv(f, index=False)
            f.write("\n# Top 10 low-confidence correct predictions\n")
            safe_correct[cols_show].to_csv(f, index=False)

    # ------------------------------------------------------------------
    # Plots (4-panel diagnostic figure)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # Panel 1: confidence distribution by correctness
    ax = axes[0, 0]
    if out["confidence"].notna().any():
        ax.hist(out.loc[out["correct"], "confidence"], bins=20, alpha=0.6,
                label="Correct", color="#1D9E75")
        ax.hist(out.loc[~out["correct"], "confidence"], bins=20, alpha=0.6,
                label="Incorrect", color="#D85A30")
        ax.set_xlabel("Prediction confidence")
        ax.set_ylabel("Count")
        ax.set_title("Confidence distribution: correct vs incorrect")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No confidence column found", ha="center", va="center")

    # Panel 2: entropy distribution by correctness
    ax = axes[0, 1]
    if "entropy" in out.columns:
        ax.hist(out.loc[out["correct"], "entropy"], bins=20, alpha=0.6,
                label="Correct", color="#1D9E75")
        ax.hist(out.loc[~out["correct"], "entropy"], bins=20, alpha=0.6,
                label="Incorrect", color="#D85A30")
        ax.set_xlabel("Predictive entropy")
        ax.set_ylabel("Count")
        ax.set_title("Entropy distribution: correct vs incorrect")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No entropy column found", ha="center", va="center")

    # Panel 3: uncertainty vs prediction probability scatter
    ax = axes[1, 0]
    if "uncertainty" in out.columns and out["pred_prob"].notna().any():
        colors = out["correct"].map({True: "#1D9E75", False: "#D85A30"})
        ax.scatter(out["pred_prob"], out["uncertainty"], c=colors, alpha=0.5, s=15)
        ax.set_xlabel("Predicted probability (melanoma)")
        ax.set_ylabel("Uncertainty")
        ax.set_title("Uncertainty vs predicted probability")
    elif "entropy" in out.columns and out["pred_prob"].notna().any():
        colors = out["correct"].map({True: "#1D9E75", False: "#D85A30"})
        ax.scatter(out["pred_prob"], out["entropy"], c=colors, alpha=0.5, s=15)
        ax.set_xlabel("Predicted probability (melanoma)")
        ax.set_ylabel("Entropy")
        ax.set_title("Entropy vs predicted probability")
    else:
        ax.text(0.5, 0.5, "No uncertainty/entropy column found", ha="center", va="center")

    # Panel 4: category counts bar chart
    ax = axes[1, 1]
    cat_order = ["TP", "TN", "FP", "FN"]
    counts = [len(out[out["category"] == c]) for c in cat_order]
    bar_colors = ["#1D9E75", "#0F6E56", "#D85A30", "#993C1D"]
    ax.bar(cat_order, counts, color=bar_colors)
    ax.set_ylabel("Count")
    ax.set_title("Prediction category counts")
    for i, v in enumerate(counts):
        ax.text(i, v + max(counts) * 0.01, str(v), ha="center")

    fig.suptitle("MedFairXNet - Error Analysis (Final CLAHE Test Set)", fontsize=15, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    plot_out = os.path.join(RESULTS_DIR, "medfairxnet_error_analysis_plots.png")
    plt.savefig(plot_out, dpi=300, bbox_inches="tight")
    plt.close()

    print("=" * 80)
    print("FILES SAVED")
    print("=" * 80)
    print("Per-sample CSV:  ", per_sample_out)
    print("Summary CSV:     ", summary_out)
    print("Diagnostic plots:", plot_out)
    print("=" * 80)
    print("STATUS: PASS")
    print("=" * 80)


if __name__ == "__main__":
    main()