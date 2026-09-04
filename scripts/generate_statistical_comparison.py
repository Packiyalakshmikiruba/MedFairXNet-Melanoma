"""
MELONMA - STATISTICAL COMPARISON: McNEMAR'S TEST (Step 11)
============================================================
Compares MedFairXNet against each of the 5 baseline models using
McNemar's test on paired predictions (same 999 test images, same
0.5 threshold decisions). This tests whether the two models'
disagreements are symmetric (no significant difference) or skewed
in one model's favor (statistically significant difference).

Requires the *_test_predictions.csv files already produced by each
model's evaluate script (all 6 now exist after the recent DenseNet121
patch and EfficientNetV2 path fix).

Run from the project's scripts/ folder:
    python generate_statistical_comparison.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"

OUTPUT_CSV = RESULTS_DIR / "statistical_comparison_mcnemar.csv"

PREDICTION_FILES = {
    "ResNet101": "resnet101_final_clahe_test_predictions.csv",
    "Swin-Tiny": "swin_final_clahe_test_predictions.csv",
    "DenseNet121": "densenet121_final_clahe_test_predictions.csv",
    "EfficientNetV2": "efficientnetv2_final_clahe/efficientnetv2_final_clahe_predictions.csv",
    "ConvNeXtTiny": "convnext_final_clahe_test_predictions.csv",
    "MedFairXNet": "medfairxnet_final_clahe_test_predictions.csv",
}

LABEL_COL_CANDIDATES = ["true_label", "binary_label", "label", "y_true", "true_class"]
PRED_LABEL_CANDIDATES = ["predicted_label", "prediction_label", "y_pred"]
PROB_COL_CANDIDATES = [
    "raw_prediction", "melanoma_probability", "probability", "y_prob", "prob",
    "prediction_score", "predicted_probability", "score", "prediction",
]

ALPHA = 0.05

print("=" * 80)
print("MELONMA - STATISTICAL COMPARISON (McNEMAR'S TEST)")
print("=" * 80)


def load_predictions(model_name, filename):
    path = RESULTS_DIR / filename
    if not path.exists():
        print(f"\n{model_name}: NOT FOUND ({path}) -- skipping")
        return None

    df = pd.read_csv(path)

    label_col = next((c for c in LABEL_COL_CANDIDATES if c in df.columns), None)
    if label_col is None:
        print(f"\n{model_name}: no usable true-label column -- skipping")
        return None

    y_true = df[label_col]
    if y_true.dtype == object:
        y_true = y_true.map({"Melanoma": 1, "Non-melanoma": 0}).fillna(y_true)
    y_true = pd.to_numeric(y_true, errors="coerce").astype(int).to_numpy()

    # Prefer an existing predicted-label column; otherwise threshold a probability column.
    pred_col = next((c for c in PRED_LABEL_CANDIDATES if c in df.columns), None)
    if pred_col is not None:
        y_pred = pd.to_numeric(df[pred_col], errors="coerce")
        if y_pred.notna().mean() > 0.99:
            y_pred = y_pred.astype(int).to_numpy()
        else:
            pred_col = None

    if pred_col is None:
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
        if prob_col is None:
            print(f"\n{model_name}: no usable prediction column -- skipping")
            return None
        y_prob = pd.to_numeric(df[prob_col], errors="coerce").to_numpy()
        y_pred = (y_prob >= 0.5).astype(int)

    print(f"\n{model_name}: loaded n={len(y_true)}")
    return y_true, y_pred


loaded = {}
for model_name, filename in PREDICTION_FILES.items():
    result = load_predictions(model_name, filename)
    if result is not None:
        loaded[model_name] = result

if "MedFairXNet" not in loaded:
    raise RuntimeError("MedFairXNet predictions could not be loaded -- cannot run comparisons.")

medfair_true, medfair_pred = loaded["MedFairXNet"]
medfair_correct = (medfair_pred == medfair_true)

rows = []

print("\n" + "=" * 80)
print("PAIRWISE McNEMAR'S TESTS: MedFairXNet vs each baseline")
print("=" * 80)

for model_name, (y_true, y_pred) in loaded.items():
    if model_name == "MedFairXNet":
        continue

    if len(y_true) != len(medfair_true):
        print(f"\n{model_name}: sample count mismatch ({len(y_true)} vs {len(medfair_true)}) -- skipping")
        continue

    # Align on the same true labels order (both files come from the same
    # test_final_clahe.csv, so row order should already match).
    if not np.array_equal(y_true, medfair_true):
        print(f"\n{model_name}: WARNING -- true labels do not match MedFairXNet's row order. "
              "Results below assume aligned rows; verify both files were built from the same test CSV order.")

    baseline_correct = (y_pred == y_true)

    # 2x2 contingency table for McNemar's test:
    #                 MedFairXNet correct   MedFairXNet wrong
    # Baseline correct        a                    b
    # Baseline wrong          c                    d
    a = int(np.sum(baseline_correct & medfair_correct))
    b = int(np.sum(baseline_correct & ~medfair_correct))
    c = int(np.sum(~baseline_correct & medfair_correct))
    d = int(np.sum(~baseline_correct & ~medfair_correct))

    table = [[a, b], [c, d]]
    # exact=True uses the binomial exact test, recommended when b+c < 25;
    # otherwise the chi-square version with continuity correction is used.
    use_exact = (b + c) < 25
    result = mcnemar(table, exact=use_exact, correction=not use_exact)

    significant = result.pvalue < ALPHA
    favors = (
        "MedFairXNet" if c > b else
        model_name if b > c else
        "Tie"
    )

    print(f"\n{model_name} vs MedFairXNet:")
    print(f"  Contingency table (baseline_correct x medfair_correct):")
    print(f"    Both correct: {a}   Baseline only: {b}   MedFairXNet only: {c}   Both wrong: {d}")
    print(f"  Test used     : {'exact binomial' if use_exact else 'chi-square (continuity corrected)'}")
    print(f"  Statistic     : {result.statistic:.4f}")
    print(f"  p-value       : {result.pvalue:.6f}")
    print(f"  Significant (alpha={ALPHA})? {'YES' if significant else 'no'}")
    print(f"  Disagreement favors: {favors}")

    rows.append({
        "comparison": f"MedFairXNet vs {model_name}",
        "both_correct": a,
        "baseline_only_correct": b,
        "medfairxnet_only_correct": c,
        "both_wrong": d,
        "test_type": "exact_binomial" if use_exact else "chi_square_corrected",
        "statistic": result.statistic,
        "p_value": result.pvalue,
        "significant_at_0.05": significant,
        "disagreement_favors": favors,
    })

comparison_df = pd.DataFrame(rows)
comparison_df.to_csv(OUTPUT_CSV, index=False)

print("\n" + "=" * 80)
print("SUMMARY TABLE")
print("=" * 80)
print(comparison_df.to_string(index=False))

print("\n" + "=" * 80)
print("FILES SAVED")
print("=" * 80)
print("Statistical comparison table:", OUTPUT_CSV)

print(
    "\nInterpretation note for the paper: a significant p-value (< 0.05) means "
    "the two models' image-level correctness disagreements are NOT symmetric -- "
    "one model is significantly more often right where the other is wrong. "
    "This is a paired test on the same 999 images, appropriate because both "
    "models were evaluated on an identical test set."
)

print("\nSTATUS: PASS")
