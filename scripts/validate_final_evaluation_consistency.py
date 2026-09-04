from pathlib import Path
import sys
import pandas as pd
import numpy as np


# ============================================================
# MELONMA - FINAL EVALUATION CONSISTENCY VALIDATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = PROJECT_ROOT / "results"

MODEL_COMPARISON = (
    RESULTS_DIR
    / "final_model_comparison"
    / "final_model_comparison.csv"
)

SELECTED_MODEL = (
    RESULTS_DIR
    / "final_model_comparison"
    / "selected_best_model.csv"
)

PH2_RAW = (
    RESULTS_DIR
    / "ph2_prediction_distribution"
    / "ph2_raw_predictions.csv"
)

PH2_THRESHOLD = (
    RESULTS_DIR
    / "ph2_prediction_distribution"
    / "ph2_threshold_analysis.csv"
)

FINAL_PREDICTIONS = (
    RESULTS_DIR
    / "final_model_visualizations"
    / "final_model_predictions.csv"
)

FINAL_METRICS = (
    RESULTS_DIR
    / "final_model_visualizations"
    / "final_model_metrics.csv"
)

OUTPUT_DIR = (
    RESULTS_DIR
    / "final_evaluation_consistency"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


SEP = "=" * 100


def load_csv(path, name):
    print(f"\n{name}")
    print("-" * 100)
    print(f"File: {path}")

    if not path.exists():
        print("STATUS: MISSING")
        return None

    try:
        df = pd.read_csv(path)
        print(f"Records: {len(df)}")
        print(f"Columns: {list(df.columns)}")
        print("STATUS: PASS")
        return df
    except Exception as exc:
        print(f"STATUS: ERROR")
        print(f"Error: {exc}")
        return None


def find_column(df, candidates):
    if df is None:
        return None

    lower_map = {
        str(col).strip().lower(): col
        for col in df.columns
    }

    for candidate in candidates:
        key = candidate.strip().lower()
        if key in lower_map:
            return lower_map[key]

    for col in df.columns:
        normalized = str(col).strip().lower()

        for candidate in candidates:
            candidate_normalized = candidate.strip().lower()

            if candidate_normalized in normalized:
                return col

    return None


def safe_float(value):
    try:
        return float(value)
    except Exception:
        return np.nan


def metric_row(df):
    if df is None or df.empty:
        return None

    row = df.iloc[0]

    result = {}

    for metric in [
        "accuracy",
        "roc_auc",
        "pr_auc",
        "precision",
        "sensitivity",
        "specificity",
        "f1_score",
        "balanced_score",
        "threshold",
    ]:
        col = find_column(df, [metric])

        if col is not None:
            result[metric] = safe_float(row[col])
        else:
            result[metric] = np.nan

    return result


def compare_metric(name, a, b, tolerance=1e-6):

    if pd.isna(a) or pd.isna(b):
        status = "NOT_COMPARABLE"
    elif abs(a - b) <= tolerance:
        status = "MATCH"
    else:
        status = "MISMATCH"

    print(
        f"{name:<20} "
        f"A={a!s:<12} "
        f"B={b!s:<12} "
        f"{status}"
    )

    return status


print(SEP)
print("MELONMA - FINAL EVALUATION CONSISTENCY VALIDATION")
print(SEP)

print(f"Project root : {PROJECT_ROOT}")
print(f"Output       : {OUTPUT_DIR}")

# ============================================================
# LOAD RESULTS
# ============================================================

model_comparison = load_csv(
    MODEL_COMPARISON,
    "MODEL COMPARISON"
)

selected_model = load_csv(
    SELECTED_MODEL,
    "SELECTED BEST MODEL"
)

ph2_raw = load_csv(
    PH2_RAW,
    "PH2 RAW PREDICTIONS"
)

ph2_threshold = load_csv(
    PH2_THRESHOLD,
    "PH2 THRESHOLD ANALYSIS"
)

final_predictions = load_csv(
    FINAL_PREDICTIONS,
    "FINAL MODEL PREDICTIONS"
)

final_metrics = load_csv(
    FINAL_METRICS,
    "FINAL MODEL METRICS"
)


# ============================================================
# VALIDATION 1 - SELECTED MODEL
# ============================================================

print("\n" + SEP)
print("VALIDATION 1 - SELECTED MODEL")
print(SEP)

selected_name = None

if selected_model is not None and not selected_model.empty:

    selected_col = find_column(
        selected_model,
        ["selected_model", "model"]
    )

    if selected_col:
        selected_name = str(
            selected_model.iloc[0][selected_col]
        )

        print(f"Selected model: {selected_name}")

    else:
        print("WARNING: Selected model column not found.")

else:
    print("WARNING: Selected model file unavailable.")


# ============================================================
# VALIDATION 2 - MODEL COMPARISON ENTRY
# ============================================================

print("\n" + SEP)
print("VALIDATION 2 - MODEL COMPARISON ENTRY")
print(SEP)

comparison_metrics = None

if model_comparison is not None and selected_name:

    model_col = find_column(
        model_comparison,
        ["model"]
    )

    if model_col:

        matches = model_comparison[
            model_comparison[model_col].astype(str)
            == selected_name
        ]

        if len(matches) > 0:

            comparison_metrics = metric_row(matches)

            print("Selected model found in comparison table.")
            print(comparison_metrics)

        else:
            print(
                "WARNING: Selected model not found "
                "in comparison table."
            )

    else:
        print("WARNING: Model column not found.")

# ============================================================
# VALIDATION 3 - FINAL METRICS
# ============================================================

print("\n" + SEP)
print("VALIDATION 3 - FINAL VISUALIZATION METRICS")
print(SEP)

final_metric_values = metric_row(final_metrics)

if final_metric_values:
    for key, value in final_metric_values.items():
        print(f"{key:<20}: {value}")

# ============================================================
# VALIDATION 4 - METRIC COMPARISON
# ============================================================

print("\n" + SEP)
print("VALIDATION 4 - MODEL COMPARISON vs FINAL EVALUATION")
print(SEP)

comparison_results = []

if comparison_metrics and final_metric_values:

    for metric in [
        "accuracy",
        "roc_auc",
        "pr_auc",
        "precision",
        "sensitivity",
        "specificity",
        "f1_score",
        "balanced_score",
        "threshold",
    ]:

        a = comparison_metrics.get(metric, np.nan)
        b = final_metric_values.get(metric, np.nan)

        status = compare_metric(
            metric,
            a,
            b
        )

        comparison_results.append({
            "metric": metric,
            "model_comparison_value": a,
            "final_visualization_value": b,
            "difference": (
                b - a
                if not pd.isna(a) and not pd.isna(b)
                else np.nan
            ),
            "status": status
        })

else:
    print("Metric comparison could not be performed.")


# ============================================================
# VALIDATION 5 - RAW PREDICTIONS
# ============================================================

print("\n" + SEP)
print("VALIDATION 5 - RAW PREDICTION SCORE INSPECTION")
print(SEP)

raw_scores = None
raw_labels = None

if ph2_raw is not None and not ph2_raw.empty:

    score_col = find_column(
        ph2_raw,
        [
            "prediction_probability",
            "probability",
            "prediction_score",
            "score",
            "y_prob",
            "prob"
        ]
    )

    label_col = find_column(
        ph2_raw,
        [
            "true_label",
            "binary_label",
            "label",
            "y_true"
        ]
    )

    print(f"Score column: {score_col}")
    print(f"Label column: {label_col}")

    if score_col:

        raw_scores = pd.to_numeric(
            ph2_raw[score_col],
            errors="coerce"
        ).dropna()

        print(f"Score count : {len(raw_scores)}")
        print(f"Min score   : {raw_scores.min():.6f}")
        print(f"Max score   : {raw_scores.max():.6f}")
        print(f"Mean score  : {raw_scores.mean():.6f}")

        print(
            f"Scores >= 0.50: "
            f"{(raw_scores >= 0.50).sum()}"
        )

        print(
            f"Scores >= 0.29: "
            f"{(raw_scores >= 0.29).sum()}"
        )

    if label_col:
        raw_labels = pd.to_numeric(
            ph2_raw[label_col],
            errors="coerce"
        )


# ============================================================
# VALIDATION 6 - FINAL PREDICTION INSPECTION
# ============================================================

print("\n" + SEP)
print("VALIDATION 6 - FINAL MODEL PREDICTIONS")
print(SEP)

final_scores = None
final_labels = None

if final_predictions is not None and not final_predictions.empty:

    score_col = find_column(
        final_predictions,
        [
            "prediction_probability",
            "probability",
            "prediction_score",
            "score",
            "y_prob",
            "prob"
        ]
    )

    label_col = find_column(
        final_predictions,
        [
            "true_label",
            "binary_label",
            "label",
            "y_true"
        ]
    )

    pred_col = find_column(
        final_predictions,
        [
            "predicted_label",
            "prediction",
            "predicted"
        ]
    )

    print(f"Score column     : {score_col}")
    print(f"True label column: {label_col}")
    print(f"Prediction column: {pred_col}")

    if score_col:

        final_scores = pd.to_numeric(
            final_predictions[score_col],
            errors="coerce"
        )

        print(
            f"Prediction count: "
            f"{final_scores.notna().sum()}"
        )

        print(
            f"Min probability : "
            f"{final_scores.min():.6f}"
        )

        print(
            f"Max probability : "
            f"{final_scores.max():.6f}"
        )

        print(
            f"Mean probability: "
            f"{final_scores.mean():.6f}"
        )

        print(
            f"Predicted positive at 0.50: "
            f"{(final_scores >= 0.50).sum()}"
        )

        print(
            f"Predicted positive at 0.29: "
            f"{(final_scores >= 0.29).sum()}"
        )

    if label_col:

        final_labels = pd.to_numeric(
            final_predictions[label_col],
            errors="coerce"
        )


# ============================================================
# VALIDATION 7 - SCORE CONSISTENCY
# ============================================================

print("\n" + SEP)
print("VALIDATION 7 - RAW vs FINAL SCORE CONSISTENCY")
print(SEP)

score_consistency = "NOT_CHECKED"

if (
    raw_scores is not None
    and final_scores is not None
    and len(raw_scores) == len(final_scores)
):

    raw_array = raw_scores.to_numpy()
    final_array = final_scores.to_numpy()

    difference = np.abs(
        raw_array - final_array
    )

    max_difference = difference.max()

    print(
        f"Maximum absolute score difference: "
        f"{max_difference:.10f}"
    )

    if max_difference <= 1e-6:
        score_consistency = "MATCH"
        print("PASS - prediction scores match.")

    else:
        score_consistency = "MISMATCH"
        print(
            "WARNING - prediction scores differ."
        )

else:
    print(
        "Could not compare prediction arrays."
    )


# ============================================================
# VALIDATION 8 - THRESHOLD ANALYSIS
# ============================================================

print("\n" + SEP)
print("VALIDATION 8 - THRESHOLD ANALYSIS")
print(SEP)

threshold_summary = []

if ph2_threshold is not None and not ph2_threshold.empty:

    threshold_col = find_column(
        ph2_threshold,
        ["threshold"]
    )

    accuracy_col = find_column(
        ph2_threshold,
        ["accuracy"]
    )

    sensitivity_col = find_column(
        ph2_threshold,
        ["sensitivity", "recall"]
    )

    specificity_col = find_column(
        ph2_threshold,
        ["specificity"]
    )

    f1_col = find_column(
        ph2_threshold,
        ["f1_score", "f1"]
    )

    balanced_col = find_column(
        ph2_threshold,
        ["balanced_score"]
    )

    if threshold_col:

        thresholds = pd.to_numeric(
            ph2_threshold[threshold_col],
            errors="coerce"
        )

        requested_thresholds = [0.29, 0.30, 0.50]

        for threshold in requested_thresholds:

            idx = np.where(
                np.isclose(
                    thresholds.to_numpy(),
                    threshold,
                    atol=1e-9
                )
            )[0]

            if len(idx) > 0:

                row = ph2_threshold.iloc[idx[0]]

                record = {
                    "threshold": threshold,
                    "accuracy": (
                        row[accuracy_col]
                        if accuracy_col else np.nan
                    ),
                    "sensitivity": (
                        row[sensitivity_col]
                        if sensitivity_col else np.nan
                    ),
                    "specificity": (
                        row[specificity_col]
                        if specificity_col else np.nan
                    ),
                    "f1_score": (
                        row[f1_col]
                        if f1_col else np.nan
                    ),
                    "balanced_score": (
                        row[balanced_col]
                        if balanced_col else np.nan
                    )
                }

                threshold_summary.append(record)

                print(
                    f"Threshold {threshold:.2f}: "
                    f"accuracy={record['accuracy']}, "
                    f"sensitivity={record['sensitivity']}, "
                    f"specificity={record['specificity']}, "
                    f"F1={record['f1_score']}, "
                    f"balanced={record['balanced_score']}"
                )


# ============================================================
# VALIDATION 9 - DATASET SIZE
# ============================================================

print("\n" + SEP)
print("VALIDATION 9 - TEST DATASET SIZE")
print(SEP)

dataset_result = {
    "raw_prediction_records": (
        len(ph2_raw)
        if ph2_raw is not None
        else np.nan
    ),
    "final_prediction_records": (
        len(final_predictions)
        if final_predictions is not None
        else np.nan
    )
}

print(
    f"Raw prediction records  : "
    f"{dataset_result['raw_prediction_records']}"
)

print(
    f"Final prediction records: "
    f"{dataset_result['final_prediction_records']}"
)


# ============================================================
# SAVE COMPARISON
# ============================================================

if comparison_results:

    comparison_df = pd.DataFrame(
        comparison_results
    )

    comparison_path = (
        OUTPUT_DIR
        / "model_comparison_vs_final_evaluation.csv"
    )

    comparison_df.to_csv(
        comparison_path,
        index=False
    )

    print("\nSaved:")
    print(comparison_path)


if threshold_summary:

    threshold_df = pd.DataFrame(
        threshold_summary
    )

    threshold_path = (
        OUTPUT_DIR
        / "threshold_consistency_summary.csv"
    )

    threshold_df.to_csv(
        threshold_path,
        index=False
    )

    print("Saved:")
    print(threshold_path)


# ============================================================
# FINAL STATUS
# ============================================================

print("\n" + SEP)
print("FINAL CONSISTENCY VALIDATION SUMMARY")
print(SEP)

mismatch_count = sum(
    1
    for item in comparison_results
    if item["status"] == "MISMATCH"
)

match_count = sum(
    1
    for item in comparison_results
    if item["status"] == "MATCH"
)

print(f"Metric matches    : {match_count}")
print(f"Metric mismatches : {mismatch_count}")
print(f"Score consistency : {score_consistency}")

if mismatch_count == 0 and score_consistency in [
    "MATCH",
    "NOT_CHECKED"
]:

    final_status = "PASS"

else:

    final_status = "REVIEW_REQUIRED"


summary = pd.DataFrame([
    {
        "metric_matches": match_count,
        "metric_mismatches": mismatch_count,
        "prediction_score_consistency": score_consistency,
        "final_status": final_status
    }
])

summary_path = (
    OUTPUT_DIR
    / "final_evaluation_consistency_summary.csv"
)

summary.to_csv(
    summary_path,
    index=False
)

print(f"Final status: {final_status}")

print("\nSummary saved:")
print(summary_path)

print("\n" + SEP)

if final_status == "PASS":
    print("STATUS: PASS")
    print(
        "Final evaluation sources are internally consistent."
    )
else:
    print("STATUS: REVIEW REQUIRED")
    print(
        "Model-comparison and final-evaluation results "
        "are not fully consistent."
    )

print(SEP)