"""
MELONMA - MEDFAIRXNET UNCERTAINTY ANALYSIS (Step 8)
============================================================
Uses the MC-Dropout columns already saved in
medfairxnet_final_clahe_test_predictions.csv (mc_dropout_std,
predictive_entropy) -- no re-inference needed.

Produces:
    results/medfairxnet_uncertainty_summary.csv
        Summary stats: mean/median entropy & std for correct vs
        incorrect predictions, and for melanoma vs non-melanoma.

    results/medfairxnet_uncertainty_analysis.png
        A 2x2 figure:
          (1) Entropy distribution: correct vs incorrect
          (2) MC-Dropout std distribution: correct vs incorrect
          (3) Reliability-style plot: predicted probability bins vs
              actual accuracy in each bin (calibration)
          (4) Entropy vs prediction confidence scatter (colored by
              correct/incorrect)

Run from the project's scripts/ folder:
    python generate_medfairxnet_uncertainty_analysis.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"

PREDICTIONS_PATH = RESULTS_DIR / "medfairxnet_final_clahe_test_predictions.csv"
SUMMARY_CSV_PATH = RESULTS_DIR / "medfairxnet_uncertainty_summary.csv"
FIGURE_PATH = RESULTS_DIR / "medfairxnet_uncertainty_analysis.png"

print("=" * 80)
print("MELONMA - MEDFAIRXNET UNCERTAINTY ANALYSIS")
print("=" * 80)

if not PREDICTIONS_PATH.exists():
    raise FileNotFoundError(f"Predictions file not found:\n{PREDICTIONS_PATH}")

df = pd.read_csv(PREDICTIONS_PATH)

required = ["true_label", "raw_prediction", "predicted_label",
            "mc_dropout_std", "predictive_entropy"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(
        f"Missing expected columns {missing} in {PREDICTIONS_PATH}. "
        "This script expects the output of evaluate_medfairxnet_final_clahe.py."
    )

df["correct"] = df["true_label"] == df["predicted_label"]

print("\nTotal predictions:", len(df))
print("Correct:", int(df["correct"].sum()), " Incorrect:", int((~df["correct"]).sum()))


# ============================================================
# SUMMARY TABLE
# ============================================================

def summarize(group_df, label):
    return {
        "group": label,
        "n": len(group_df),
        "mean_entropy": group_df["predictive_entropy"].mean(),
        "median_entropy": group_df["predictive_entropy"].median(),
        "mean_mc_std": group_df["mc_dropout_std"].mean(),
        "median_mc_std": group_df["mc_dropout_std"].median(),
    }

rows = [
    summarize(df[df["correct"]], "Correct predictions"),
    summarize(df[~df["correct"]], "Incorrect predictions"),
    summarize(df[df["true_label"] == 1], "True melanoma cases"),
    summarize(df[df["true_label"] == 0], "True non-melanoma cases"),
]

summary_df = pd.DataFrame(rows)
summary_df.to_csv(SUMMARY_CSV_PATH, index=False)

print("\n" + "=" * 80)
print("UNCERTAINTY SUMMARY")
print("=" * 80)
print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

print(
    "\nInterpretation: if mean_entropy and mean_mc_std are noticeably "
    "higher for 'Incorrect predictions' than 'Correct predictions', the "
    "model's uncertainty estimate is meaningful -- it tends to be less "
    "confident exactly when it is wrong, which is the desired calibration "
    "behaviour for a clinical-support tool."
)


# ============================================================
# FIGURE: 2x2 uncertainty analysis panel
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(13, 11))

# (1) Entropy distribution: correct vs incorrect
ax = axes[0, 0]
ax.hist(df.loc[df["correct"], "predictive_entropy"], bins=30, alpha=0.6,
        label="Correct", color="#2ca02c", density=True)
ax.hist(df.loc[~df["correct"], "predictive_entropy"], bins=30, alpha=0.6,
        label="Incorrect", color="#d62728", density=True)
ax.set_xlabel("Predictive entropy")
ax.set_ylabel("Density")
ax.set_title("Entropy: Correct vs Incorrect Predictions")
ax.legend()
ax.grid(alpha=0.3)

# (2) MC-Dropout std distribution: correct vs incorrect
ax = axes[0, 1]
ax.hist(df.loc[df["correct"], "mc_dropout_std"], bins=30, alpha=0.6,
        label="Correct", color="#2ca02c", density=True)
ax.hist(df.loc[~df["correct"], "mc_dropout_std"], bins=30, alpha=0.6,
        label="Incorrect", color="#d62728", density=True)
ax.set_xlabel("MC-Dropout std (epistemic uncertainty)")
ax.set_ylabel("Density")
ax.set_title("MC-Dropout Std: Correct vs Incorrect Predictions")
ax.legend()
ax.grid(alpha=0.3)

# (3) Reliability / calibration plot
ax = axes[1, 0]
n_bins = 10
bin_edges = np.linspace(0, 1, n_bins + 1)
df["prob_bin"] = pd.cut(df["raw_prediction"], bins=bin_edges, include_lowest=True)

bin_stats = df.groupby("prob_bin", observed=True).agg(
    mean_predicted=("raw_prediction", "mean"),
    mean_actual=("true_label", "mean"),
    count=("true_label", "size"),
).reset_index()
bin_stats = bin_stats[bin_stats["count"] > 0]

ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
ax.plot(bin_stats["mean_predicted"], bin_stats["mean_actual"],
        marker="o", color="#1f77b4", label="MedFairXNet")
ax.set_xlabel("Mean predicted probability (bin)")
ax.set_ylabel("Observed melanoma fraction (bin)")
ax.set_title("Reliability / Calibration Plot")
ax.legend()
ax.grid(alpha=0.3)

# (4) Entropy vs confidence scatter
ax = axes[1, 1]
confidence = np.abs(df["raw_prediction"] - 0.5) * 2  # 0 = uncertain, 1 = confident
colors = np.where(df["correct"], "#2ca02c", "#d62728")
ax.scatter(confidence, df["predictive_entropy"], c=colors, alpha=0.4, s=15)
ax.set_xlabel("Prediction confidence (|p - 0.5| x 2)")
ax.set_ylabel("Predictive entropy")
ax.set_title("Confidence vs Entropy (green=correct, red=incorrect)")
ax.grid(alpha=0.3)

fig.suptitle("MedFairXNet — MC-Dropout Uncertainty Analysis (Final CLAHE Test Set)",
              fontsize=15, fontweight="bold", y=1.0)
fig.tight_layout()
fig.savefig(FIGURE_PATH, dpi=200, bbox_inches="tight")
plt.close(fig)

print("\n" + "=" * 80)
print("FILES SAVED")
print("=" * 80)
print("Summary table:", SUMMARY_CSV_PATH)
print("Uncertainty analysis figure:", FIGURE_PATH)
print("\nSTATUS: PASS")