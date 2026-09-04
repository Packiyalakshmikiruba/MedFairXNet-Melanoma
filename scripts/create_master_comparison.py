"""
MELONMA - MASTER MODEL COMPARISON
============================================================
Combines final CLAHE test results from:

1. EfficientNetV2
2. ConvNeXtTiny
3. DenseNet121
4. ResNet101
5. Swin-Tiny
6. MedFairXNet

Outputs:
- Master comparison CSV
- Metric ranking
- ROC-AUC comparison plot
- Accuracy/F1 comparison plot
- Sensitivity/Specificity comparison plot

IMPORTANT:
All values are read from existing evaluation CSV files.
No results are invented or modified.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(r"C:\Users\HP\Desktop\Melonma")
RESULTS_DIR = PROJECT_ROOT / "results"


MODEL_FILES = {
    "EfficientNetV2":
        RESULTS_DIR / "efficientnetv2_final_clahe_test_evaluation.csv",

    "ConvNeXtTiny":
        RESULTS_DIR / "convnext_final_clahe_test_evaluation.csv",

    "DenseNet121":
        RESULTS_DIR / "densenet121_final_clahe_test_evaluation.csv",

    "ResNet101":
        RESULTS_DIR / "resnet101_final_clahe_test_evaluation.csv",

    "Swin-Tiny":
        RESULTS_DIR / "swin_final_clahe_test_evaluation.csv",

    "MedFairXNet":
        RESULTS_DIR / "medfairxnet_final_clahe_test_evaluation.csv",
}


# ============================================================
# OUTPUT FILES
# ============================================================

MASTER_CSV = RESULTS_DIR / "master_model_comparison_final_clahe.csv"

ROC_AUC_PLOT = RESULTS_DIR / "master_comparison_roc_auc.png"
ACC_F1_PLOT = RESULTS_DIR / "master_comparison_accuracy_f1.png"
SENS_SPEC_PLOT = RESULTS_DIR / "master_comparison_sensitivity_specificity.png"


# ============================================================
# LOAD RESULTS
# ============================================================

records = []

print("=" * 80)
print("MELONMA - MASTER MODEL COMPARISON")
print("=" * 80)

for model_name, file_path in MODEL_FILES.items():

    print(f"\nLoading: {model_name}")
    print(f"File: {file_path}")

    if not file_path.exists():
        raise FileNotFoundError(
            f"Missing evaluation file:\n{file_path}"
        )

    df = pd.read_csv(file_path)

    if df.empty:
        raise ValueError(
            f"Evaluation CSV is empty:\n{file_path}"
        )

    row = df.iloc[0].to_dict()

    row["model"] = model_name

    records.append(row)

    print("Loaded successfully.")


# ============================================================
# CREATE MASTER DATAFRAME
# ============================================================

comparison = pd.DataFrame(records)


# ============================================================
# STANDARDIZE COLUMN NAMES
# ============================================================

required_metrics = [
    "accuracy",
    "roc_auc",
    "pr_auc",
    "precision",
    "sensitivity",
    "specificity",
    "f1_score",
]


for metric in required_metrics:

    if metric not in comparison.columns:
        comparison[metric] = float("nan")


# ============================================================
# SELECT IMPORTANT COLUMNS
# ============================================================

columns = [
    "model",
    "accuracy",
    "roc_auc",
    "pr_auc",
    "precision",
    "sensitivity",
    "specificity",
    "f1_score",
    "true_negative",
    "false_positive",
    "false_negative",
    "true_positive",
]


optional_columns = [
    "mean_entropy_correct",
    "mean_entropy_incorrect",
    "mc_dropout_samples",
    "evaluation_time_seconds",
]


for col in optional_columns:
    if col in comparison.columns:
        columns.append(col)


comparison = comparison[
    [c for c in columns if c in comparison.columns]
]


# ============================================================
# NUMERIC CONVERSION
# ============================================================

for col in comparison.columns:

    if col != "model":
        comparison[col] = pd.to_numeric(
            comparison[col],
            errors="coerce"
        )


# ============================================================
# RANKINGS
# ============================================================

comparison["rank_accuracy"] = (
    comparison["accuracy"]
    .rank(method="min", ascending=False)
    .astype(int)
)

comparison["rank_roc_auc"] = (
    comparison["roc_auc"]
    .rank(method="min", ascending=False)
    .astype(int)
)

comparison["rank_pr_auc"] = (
    comparison["pr_auc"]
    .rank(method="min", ascending=False)
    .astype(int)
)

comparison["rank_f1"] = (
    comparison["f1_score"]
    .rank(method="min", ascending=False)
    .astype(int)
)

comparison["rank_sensitivity"] = (
    comparison["sensitivity"]
    .rank(method="min", ascending=False)
    .astype(int)
)

comparison["rank_specificity"] = (
    comparison["specificity"]
    .rank(method="min", ascending=False)
    .astype(int)
)


# ============================================================
# OVERALL RANK
# ============================================================

rank_columns = [
    "rank_accuracy",
    "rank_roc_auc",
    "rank_pr_auc",
    "rank_f1",
    "rank_sensitivity",
    "rank_specificity",
]

comparison["mean_rank"] = comparison[rank_columns].mean(axis=1)

comparison["overall_rank"] = (
    comparison["mean_rank"]
    .rank(method="min", ascending=True)
    .astype(int)
)


# ============================================================
# SORT BY OVERALL RANK
# ============================================================

comparison = comparison.sort_values(
    "overall_rank"
).reset_index(drop=True)


# ============================================================
# SAVE MASTER CSV
# ============================================================

comparison.to_csv(
    MASTER_CSV,
    index=False
)


# ============================================================
# PRINT MASTER TABLE
# ============================================================

print("\n")
print("=" * 80)
print("FINAL MASTER COMPARISON")
print("=" * 80)

display_columns = [
    "overall_rank",
    "model",
    "accuracy",
    "roc_auc",
    "pr_auc",
    "precision",
    "sensitivity",
    "specificity",
    "f1_score",
]

print(
    comparison[display_columns].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# BEST MODEL PER METRIC
# ============================================================

print("\n")
print("=" * 80)
print("BEST MODEL PER METRIC")
print("=" * 80)

for metric in required_metrics:

    best_idx = comparison[metric].idxmax()

    best_model = comparison.loc[best_idx, "model"]
    best_value = comparison.loc[best_idx, metric]

    print(
        f"{metric:15s} : "
        f"{best_model:18s} = {best_value:.4f}"
    )


# ============================================================
# PLOT 1 - ROC-AUC
# ============================================================

plot_df = comparison.sort_values(
    "roc_auc",
    ascending=True
)

plt.figure(figsize=(10, 6))

plt.barh(
    plot_df["model"],
    plot_df["roc_auc"]
)

plt.xlabel("ROC-AUC")
plt.ylabel("Model")
plt.title(
    "Final CLAHE Test: ROC-AUC Comparison"
)

plt.xlim(0, 1)

plt.tight_layout()

plt.savefig(
    ROC_AUC_PLOT,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# PLOT 2 - ACCURACY AND F1
# ============================================================

plot_df = comparison.sort_values(
    "accuracy",
    ascending=True
)

x = range(len(plot_df))

width = 0.35

plt.figure(figsize=(11, 6))

plt.bar(
    [i - width / 2 for i in x],
    plot_df["accuracy"],
    width=width,
    label="Accuracy"
)

plt.bar(
    [i + width / 2 for i in x],
    plot_df["f1_score"],
    width=width,
    label="F1-score"
)

plt.xticks(
    list(x),
    plot_df["model"],
    rotation=30,
    ha="right"
)

plt.ylabel("Score")
plt.title(
    "Final CLAHE Test: Accuracy and F1-score Comparison"
)

plt.ylim(0, 1)

plt.legend()

plt.tight_layout()

plt.savefig(
    ACC_F1_PLOT,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# PLOT 3 - SENSITIVITY AND SPECIFICITY
# ============================================================

plot_df = comparison.sort_values(
    "sensitivity",
    ascending=True
)

x = range(len(plot_df))

width = 0.35

plt.figure(figsize=(11, 6))

plt.bar(
    [i - width / 2 for i in x],
    plot_df["sensitivity"],
    width=width,
    label="Sensitivity"
)

plt.bar(
    [i + width / 2 for i in x],
    plot_df["specificity"],
    width=width,
    label="Specificity"
)

plt.xticks(
    list(x),
    plot_df["model"],
    rotation=30,
    ha="right"
)

plt.ylabel("Score")
plt.title(
    "Final CLAHE Test: Sensitivity and Specificity Comparison"
)

plt.ylim(0, 1)

plt.legend()

plt.tight_layout()

plt.savefig(
    SENS_SPEC_PLOT,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FINAL STATUS
# ============================================================

print("\n")
print("=" * 80)
print("FILES SAVED")
print("=" * 80)

print(f"Master comparison:")
print(MASTER_CSV)

print(f"\nROC-AUC plot:")
print(ROC_AUC_PLOT)

print(f"\nAccuracy/F1 plot:")
print(ACC_F1_PLOT)

print(f"\nSensitivity/Specificity plot:")
print(SENS_SPEC_PLOT)

print("\n")
print("=" * 80)
print("STATUS: PASS")
print("=" * 80)