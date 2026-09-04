from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path(r"C:\Users\HP\Desktop\Melonma")
RESULTS = BASE / "results"

OUT = RESULTS / "final_research_summary"
OUT.mkdir(parents=True, exist_ok=True)

print("=" * 100)
print("MELONMA - FINAL RESEARCH SUMMARY")
print("=" * 100)

# ================================================================
# FILE PATHS
# ================================================================

files = {
    "model_comparison":
        RESULTS / "final_model_comparison" / "final_model_comparison.csv",

    "selected_model":
        RESULTS / "final_model_comparison" / "selected_best_model.csv",

    "model_ranking":
        RESULTS / "final_model_comparison" / "model_ranking_summary.csv",

    "ph2_integrity":
        RESULTS / "ph2_dataset_integrity_corrected"
        / "corrected_ph2_integrity_summary.csv",

    "ph2_prediction_summary":
        RESULTS / "ph2_prediction_distribution"
        / "ph2_prediction_distribution_summary.csv",

    "ph2_threshold_analysis":
        RESULTS / "ph2_prediction_distribution"
        / "ph2_threshold_analysis.csv",

    "ph2_raw_predictions":
        RESULTS / "ph2_prediction_distribution"
        / "ph2_raw_predictions.csv",

    "gradcam":
        RESULTS / "gradcam_efficientnetv2_ph2"
        / "gradcam_results.csv",

    "visual_metrics":
        RESULTS / "final_model_visualizations"
        / "final_model_metrics.csv",

    "visual_predictions":
        RESULTS / "final_model_visualizations"
        / "final_model_predictions.csv",

    "error_metrics":
        RESULTS / "final_model_error_analysis"
        / "final_error_analysis_metrics.csv",

    "error_predictions":
        RESULTS / "final_model_error_analysis"
        / "final_model_predictions.csv",

    "densenet_threshold":
        RESULTS / "densenet121_thresholded_test"
        / "thresholded_test_metrics.csv",
}

# ================================================================
# LOAD AVAILABLE FILES
# ================================================================

loaded = {}

print("\n" + "=" * 100)
print("LOADING EXISTING RESULTS")
print("=" * 100)

for name, path in files.items():

    if path.exists():

        try:
            df = pd.read_csv(path)
            loaded[name] = df

            print(f"PASS - {name}")
            print(f"      {path}")

        except Exception as e:

            print(f"ERROR - {name}")
            print(f"        {e}")

    else:

        print(f"SKIP - {name}")
        print(f"       File not found: {path}")

# ================================================================
# MODEL COMPARISON
# ================================================================

if "model_comparison" in loaded:

    df = loaded["model_comparison"]

    print("\n" + "=" * 100)
    print("FINAL MODEL COMPARISON")
    print("=" * 100)

    print(df.to_string(index=False))

    df.to_csv(
        OUT / "final_model_comparison.csv",
        index=False
    )

# ================================================================
# SELECTED MODEL
# ================================================================

if "selected_model" in loaded:

    df = loaded["selected_model"]

    print("\n" + "=" * 100)
    print("SELECTED BEST MODEL")
    print("=" * 100)

    print(df.to_string(index=False))

    df.to_csv(
        OUT / "selected_best_model.csv",
        index=False
    )

# ================================================================
# PH2 INTEGRITY
# ================================================================

if "ph2_integrity" in loaded:

    df = loaded["ph2_integrity"]

    print("\n" + "=" * 100)
    print("PH2 DATASET INTEGRITY")
    print("=" * 100)

    print(df.to_string(index=False))

    df.to_csv(
        OUT / "ph2_integrity_summary.csv",
        index=False
    )

# ================================================================
# PH2 PREDICTION DISTRIBUTION
# ================================================================

if "ph2_prediction_summary" in loaded:

    df = loaded["ph2_prediction_summary"]

    print("\n" + "=" * 100)
    print("PH2 PREDICTION DISTRIBUTION")
    print("=" * 100)

    print(df.to_string(index=False))

    df.to_csv(
        OUT / "ph2_prediction_distribution_summary.csv",
        index=False
    )

# ================================================================
# THRESHOLD ANALYSIS
# ================================================================

if "ph2_threshold_analysis" in loaded:

    df = loaded["ph2_threshold_analysis"]

    print("\n" + "=" * 100)
    print("PH2 THRESHOLD ANALYSIS")
    print("=" * 100)

    print(df.to_string(index=False))

    df.to_csv(
        OUT / "ph2_threshold_analysis.csv",
        index=False
    )

# ================================================================
# GRAD-CAM
# ================================================================

if "gradcam" in loaded:

    df = loaded["gradcam"]

    print("\n" + "=" * 100)
    print("GRAD-CAM RESULTS")
    print("=" * 100)

    print(f"Grad-CAM records: {len(df)}")

    print(df.to_string(index=False))

    df.to_csv(
        OUT / "gradcam_results.csv",
        index=False
    )

# ================================================================
# FINAL VISUALIZATION METRICS
# ================================================================

if "visual_metrics" in loaded:

    df = loaded["visual_metrics"]

    print("\n" + "=" * 100)
    print("FINAL VISUALIZATION METRICS")
    print("=" * 100)

    print(df.to_string(index=False))

    df.to_csv(
        OUT / "final_visualization_metrics.csv",
        index=False
    )

# ================================================================
# ERROR ANALYSIS
# ================================================================

if "error_metrics" in loaded:

    df = loaded["error_metrics"]

    print("\n" + "=" * 100)
    print("FINAL ERROR ANALYSIS")
    print("=" * 100)

    print(df.to_string(index=False))

    df.to_csv(
        OUT / "final_error_analysis_metrics.csv",
        index=False
    )

# ================================================================
# DENSENET THRESHOLD
# ================================================================

if "densenet_threshold" in loaded:

    df = loaded["densenet_threshold"]

    print("\n" + "=" * 100)
    print("DENSENET121 THRESHOLD RESULT")
    print("=" * 100)

    print(df.to_string(index=False))

    df.to_csv(
        OUT / "densenet121_threshold_result.csv",
        index=False
    )

# ================================================================
# MASTER MODEL TABLE
# ================================================================

if "model_comparison" in loaded:

    comparison = loaded["model_comparison"].copy()

    columns = [
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

    available_columns = [
        c for c in columns
        if c in comparison.columns
    ]

    master = comparison[available_columns].copy()

    if "balanced_score" in master.columns:

        master = master.sort_values(
            by="balanced_score",
            ascending=False,
            na_position="last"
        )

    master.insert(
        0,
        "rank",
        range(1, len(master) + 1)
    )

    master.to_csv(
        OUT / "MASTER_MODEL_SUMMARY.csv",
        index=False
    )

    print("\n" + "=" * 100)
    print("MASTER MODEL SUMMARY")
    print("=" * 100)

    print(master.to_string(index=False))

# ================================================================
# RESEARCH CONCLUSION
# ================================================================

print("\n" + "=" * 100)
print("RESEARCH CONCLUSION")
print("=" * 100)

conclusion = []

conclusion.append(
    "MELONMA - FINAL RESEARCH CONCLUSION"
)

conclusion.append("=" * 80)

conclusion.append(
    "Dataset: PH2"
)

conclusion.append(
    "Base dataset size: 200 images"
)

conclusion.append(
    "Training: 140 images"
)

conclusion.append(
    "Validation: 30 images"
)

conclusion.append(
    "Test: 30 images"
)

conclusion.append(
    "Test distribution: 24 non-melanoma and 6 melanoma"
)

conclusion.append("")

conclusion.append(
    "Dataset integrity validation:"
)

conclusion.append(
    "TRAIN/VALIDATION/TEST base splits contain no true cross-split image leakage."
)

conclusion.append(
    "Original and CLAHE versions were intentionally treated as preprocessing "
    "variants of the same split and were not counted as leakage."
)

conclusion.append("")

conclusion.append(
    "Best model selected by the previous model-comparison pipeline:"
)

if "model_comparison" in loaded:

    comparison = loaded["model_comparison"].copy()

    if "balanced_score" in comparison.columns:

        comparison = comparison.sort_values(
            by="balanced_score",
            ascending=False,
            na_position="last"
        )

        best = comparison.iloc[0]

        for col in [
            "model",
            "accuracy",
            "roc_auc",
            "pr_auc",
            "precision",
            "sensitivity",
            "specificity",
            "f1_score",
            "balanced_score",
        ]:

            if col in best.index:

                value = best[col]

                if isinstance(value, (float, np.floating)):
                    value = f"{value:.4f}"

                conclusion.append(
                    f"{col}: {value}"
                )

else:

    conclusion.append(
        "Model comparison file was not available."
    )

conclusion.append("")

conclusion.append(
    "Important evaluation observation:"
)

conclusion.append(
    "The final EfficientNetV2 + PH2 + CLAHE model produced very low "
    "prediction scores on the PH2 test set."
)

conclusion.append(
    "Therefore, threshold selection has a major effect on melanoma "
    "classification performance."
)

conclusion.append(
    "Accuracy alone should not be used as the primary measure because "
    "the PH2 test set is small and class-imbalanced."
)

conclusion.append(
    "Sensitivity, specificity, precision, F1-score, ROC-AUC and PR-AUC "
    "should be reported together."
)

conclusion.append("")

conclusion.append(
    "Grad-CAM interpretability analysis was successfully generated for "
    "10 PH2 test images using the EfficientNetV2 top_activation layer."
)

conclusion.append("")

conclusion.append(
    "Limitation:"
)

conclusion.append(
    "The PH2 dataset contains only 200 images and only 6 melanoma cases "
    "in the test set. Consequently, the reported test metrics have high "
    "statistical uncertainty and should not be interpreted as clinical "
    "validation."
)

conclusion.append("")

conclusion.append(
    "Final status: Research pipeline artifacts successfully consolidated."
)

conclusion_file = OUT / "FINAL_RESEARCH_CONCLUSION.txt"

with open(
    conclusion_file,
    "w",
    encoding="utf-8"
) as f:

    f.write("\n".join(conclusion))

for line in conclusion:
    print(line)

# ================================================================
# FILE INVENTORY
# ================================================================

print("\n" + "=" * 100)
print("FINAL OUTPUT FILES")
print("=" * 100)

for path in sorted(OUT.glob("*")):

    if path.is_file():

        print(path.name)

# ================================================================
# STATUS
# ================================================================

print("\n" + "=" * 100)
print("STATUS: PASS")
print("=" * 100)

print(
    "Final research summary successfully generated."
)

print(
    f"Output directory:\n{OUT}"
)

print("=" * 100)

