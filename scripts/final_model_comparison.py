from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================
# MELONMA - FINAL MODEL COMPARISON
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"

OUTPUT_DIR = RESULTS_DIR / "final_model_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def find_csv(relative_path):
    """
    Return CSV path if it exists.
    """
    path = RESULTS_DIR / relative_path

    if path.exists():
        return path

    return None


def read_metrics(path):
    """
    Read a metrics CSV and return the first row.
    """
    if path is None:
        return None

    try:
        df = pd.read_csv(path)

        if df.empty:
            return None

        return df.iloc[0].to_dict()

    except Exception as error:
        print(f"WARNING: Could not read {path}")
        print(f"Reason: {error}")
        return None


def get_value(data, possible_names, default=np.nan):
    """
    Find a metric using multiple possible column names.
    """
    if data is None:
        return default

    for name in possible_names:

        if name in data:
            try:
                return float(data[name])
            except:
                return data[name]

    return default


# ============================================================
# HEADER
# ============================================================

print("=" * 100)
print("MELONMA - FINAL MODEL COMPARISON")
print("=" * 100)

print("\nResults directory:")
print(RESULTS_DIR)


# ============================================================
# MODEL RESULT FILES
# ============================================================

model_files = {

    "ResNet101_Final_CLAHE":
        "resnet101_final_clahe_test_evaluation.csv",

    "ConvNeXt_Final_CLAHE":
        "convnext_final_clahe_test_evaluation.csv",

    "Swin_Final_CLAHE":
        "swin_final_clahe_test_evaluation.csv",

    "MedFairXNet_Final_CLAHE":
        "medfairxnet_final_clahe_test_evaluation.csv",

    "DenseNet121_Final_CLAHE":
        "densenet121_final_clahe_test_evaluation.csv",

    "EfficientNetV2_Final_CLAHE":
        "efficientnetv2_final_clahe_test_evaluation.csv",

    "DenseNet121_ROI":
        "roi_densenet121/evaluation_metrics.csv",

    "EfficientNetB0_ROI":
        "roi_efficientnetb0/evaluation_metrics.csv",

    "EfficientNetV2_PH2_CLAHE":
        "efficientnetv2_ph2_clahe_test_evaluation.csv",

    "DenseNet121_Threshold_027":
        "densenet121_thresholded_test/thresholded_test_metrics.csv",
}


# ============================================================
# LOAD RESULTS
# ============================================================

records = []

print("\n" + "=" * 100)
print("LOADING MODEL RESULTS")
print("=" * 100)

for model_name, relative_path in model_files.items():

    path = find_csv(relative_path)

    print(f"\n{model_name}")

    print("File:")
    print(RESULTS_DIR / relative_path)

    if path is None:

        print("STATUS: FILE NOT FOUND")
        continue

    data = read_metrics(path)

    if data is None:

        print("STATUS: COULD NOT READ")
        continue

    print("STATUS: LOADED")

    record = {
        "model": model_name,

        "accuracy": get_value(
            data,
            ["accuracy", "Accuracy"]
        ),

        "roc_auc": get_value(
            data,
            ["roc_auc", "ROC-AUC", "roc_auc_score"]
        ),

        "pr_auc": get_value(
            data,
            ["pr_auc", "PR-AUC", "average_precision"]
        ),

        "precision": get_value(
            data,
            ["precision", "Precision"]
        ),

        "sensitivity": get_value(
            data,
            ["sensitivity", "Sensitivity", "recall", "Recall"]
        ),

        "specificity": get_value(
            data,
            ["specificity", "Specificity"]
        ),

        "f1_score": get_value(
            data,
            ["f1_score", "F1-score", "f1", "F1 Score"]
        ),

        "threshold": get_value(
            data,
            ["threshold"],
            default=0.50
        ),

        "source_file": str(path),

    }

    records.append(record)


# ============================================================
# CHECK RESULTS
# ============================================================

if not records:

    raise RuntimeError(
        "\nNo model result CSV files could be loaded."
    )


comparison = pd.DataFrame(records)


# ============================================================
# NORMALIZE METRICS
# ============================================================

metric_columns = [
    "accuracy",
    "roc_auc",
    "pr_auc",
    "precision",
    "sensitivity",
    "specificity",
    "f1_score",
]

for column in metric_columns:

    comparison[column] = pd.to_numeric(
        comparison[column],
        errors="coerce"
    )


# ============================================================
# CONVERT TO PERCENTAGES
# ============================================================

for column in metric_columns:

    comparison[
        column + "_percent"
    ] = comparison[column] * 100


# ============================================================
# CREATE OVERALL SCORE
# ============================================================

"""
The overall score gives balanced importance to:

ROC-AUC
PR-AUC
Sensitivity
Specificity
F1-score

Accuracy is intentionally given less importance because
the dataset is imbalanced (many more non-melanoma cases).
"""

comparison["balanced_score"] = (

    0.25 * comparison["roc_auc"].fillna(0)

    + 0.20 * comparison["pr_auc"].fillna(0)

    + 0.20 * comparison["sensitivity"].fillna(0)

    + 0.15 * comparison["specificity"].fillna(0)

    + 0.20 * comparison["f1_score"].fillna(0)
)


# ============================================================
# RANKING
# ============================================================

comparison = comparison.sort_values(
    by="balanced_score",
    ascending=False
).reset_index(drop=True)

comparison["rank"] = (
    np.arange(len(comparison)) + 1
)


# ============================================================
# DISPLAY MAIN TABLE
# ============================================================

print("\n" + "=" * 100)
print("FINAL MODEL PERFORMANCE COMPARISON")
print("=" * 100)

display_columns = [
    "rank",
    "model",
    "accuracy",
    "roc_auc",
    "pr_auc",
    "precision",
    "sensitivity",
    "specificity",
    "f1_score",
    "threshold",
    "balanced_score",
]

print(
    comparison[
        display_columns
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# BEST MODEL BY INDIVIDUAL METRICS
# ============================================================

print("\n" + "=" * 100)
print("BEST MODEL BY METRIC")
print("=" * 100)

for metric in metric_columns:

    valid = comparison.dropna(
        subset=[metric]
    )

    if valid.empty:
        continue

    best_row = valid.loc[
        valid[metric].idxmax()
    ]

    print(
        f"\n{metric.upper():<15}: "
        f"{best_row['model']} "
        f"({best_row[metric]:.4f})"
    )


# ============================================================
# TOP MODEL
# ============================================================

best_model = comparison.iloc[0]

print("\n" + "=" * 100)
print("OVERALL BEST MODEL")
print("=" * 100)

print(
    f"\nModel          : {best_model['model']}"
)

print(
    f"Balanced Score : {best_model['balanced_score']:.4f}"
)

print(
    f"Accuracy       : {best_model['accuracy']:.4f}"
)

print(
    f"ROC-AUC        : {best_model['roc_auc']:.4f}"
)

print(
    f"PR-AUC         : {best_model['pr_auc']:.4f}"
)

print(
    f"Precision      : {best_model['precision']:.4f}"
)

print(
    f"Sensitivity    : {best_model['sensitivity']:.4f}"
)

print(
    f"Specificity    : {best_model['specificity']:.4f}"
)

print(
    f"F1-score       : {best_model['f1_score']:.4f}"
)


# ============================================================
# IMPORTANT MODEL NOTES
# ============================================================

print("\n" + "=" * 100)
print("MODEL INTERPRETATION")
print("=" * 100)

for _, row in comparison.iterrows():

    model = row["model"]

    print(f"\n{model}:")

    auc = row["roc_auc"]
    sensitivity = row["sensitivity"]
    specificity = row["specificity"]
    f1 = row["f1_score"]

    if not pd.isna(auc):

        if auc >= 0.70:
            print("  - Good ranking/discrimination capability.")

        elif auc >= 0.60:
            print("  - Moderate discrimination capability.")

        elif auc >= 0.50:
            print("  - Weak discrimination capability.")

        else:
            print("  - ROC-AUC below 0.50; prediction behaviour requires investigation.")

    if not pd.isna(sensitivity):

        print(
            f"  - Melanoma sensitivity: "
            f"{sensitivity * 100:.2f}%"
        )

    if not pd.isna(specificity):

        print(
            f"  - Non-melanoma specificity: "
            f"{specificity * 100:.2f}%"
        )

    if not pd.isna(f1):

        print(
            f"  - Melanoma F1-score: "
            f"{f1 * 100:.2f}%"
        )


# ============================================================
# SAVE FULL COMPARISON
# ============================================================

comparison_path = (
    OUTPUT_DIR
    / "final_model_comparison.csv"
)

comparison.to_csv(
    comparison_path,
    index=False
)


# ============================================================
# SAVE SHORT SUMMARY
# ============================================================

summary_columns = [
    "rank",
    "model",
    "accuracy",
    "roc_auc",
    "pr_auc",
    "precision",
    "sensitivity",
    "specificity",
    "f1_score",
    "threshold",
    "balanced_score",
]

summary = comparison[
    summary_columns
].copy()

summary_path = (
    OUTPUT_DIR
    / "model_ranking_summary.csv"
)

summary.to_csv(
    summary_path,
    index=False
)


# ============================================================
# SAVE BEST MODEL
# ============================================================

best_model_df = pd.DataFrame([
    {
        "selected_model":
            best_model["model"],

        "selection_basis":
            "Balanced score using ROC-AUC, PR-AUC, sensitivity, specificity and F1-score",

        "accuracy":
            best_model["accuracy"],

        "roc_auc":
            best_model["roc_auc"],

        "pr_auc":
            best_model["pr_auc"],

        "precision":
            best_model["precision"],

        "sensitivity":
            best_model["sensitivity"],

        "specificity":
            best_model["specificity"],

        "f1_score":
            best_model["f1_score"],

        "threshold":
            best_model["threshold"],

        "balanced_score":
            best_model["balanced_score"],
    }
])

best_model_path = (
    OUTPUT_DIR
    / "selected_best_model.csv"
)

best_model_df.to_csv(
    best_model_path,
    index=False
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 100)
print("FILES SAVED")
print("=" * 100)

print("\nFull comparison:")
print(comparison_path)

print("\nModel ranking:")
print(summary_path)

print("\nSelected model:")
print(best_model_path)

print("\n" + "=" * 100)
print("STATUS: PASS")
print("Final model comparison completed successfully.")
print("=" * 100)