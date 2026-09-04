"""
MELONMA - Statistical Significance Testing (McNemar's Test)
==============================================================
Tests whether the difference between ResNet101 (best conventional
baseline) and MedFairXNet (proposed model) on the SAME 999-image
paired test set is statistically significant - not just a difference
that could arise from random chance.

McNemar's test is the correct choice here because both models were
evaluated on the exact same test samples (paired predictions), which
violates the independence assumption of a plain chi-square or
two-proportion z-test.

Also runs pairwise McNemar's tests across ALL 6 models so you have a
complete significance matrix for the paper's results section.

Outputs (results/):
    mcnemar_resnet101_vs_medfairxnet.csv   <- the headline comparison
    mcnemar_pairwise_all_models.csv        <- full 6x6 comparison matrix

Run from the project's scripts/ folder:
    python scripts\\mcnemar_significance_test.py

Requires per-sample predictions CSVs for each model (same true_label
ordering / same test set). Uses the predicted_label column at each
model's default threshold (0.50) unless PRED_FILES below is edited.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"

# Per-sample prediction files for each model on the SAME 999-image test set.
# Only ResNet101 and MedFairXNet have confirmed per-sample predictions CSVs
# so far (from the confusion-matrix work). Add the other 4 here once you
# generate their per-sample predictions CSVs the same way.
PRED_FILES = {
    "ResNet101": RESULTS_DIR / "resnet101_final_clahe_test_predictions.csv",
    "MedFairXNet": RESULTS_DIR / "medfairxnet_final_clahe_test_predictions.csv",
    "Swin-Tiny": RESULTS_DIR / "swin_final_clahe_test_predictions.csv",
    "ConvNeXtTiny": RESULTS_DIR / "convnext_final_clahe_test_predictions.csv",
    # EfficientNetV2 / DenseNet121 per-sample predictions CSVs were not
    # found in your results\ folder yet - add their paths here once
    # generated, following the same pattern as generate_resnet101_test_predictions.py.
}

TRUE_LABEL_CANDIDATES = ["true_label", "y_true", "label", "target"]
PRED_LABEL_CANDIDATES = ["predicted_label", "y_pred", "pred_label", "prediction"]
ID_CANDIDATES = ["image_id", "filename", "image", "id"]


def find_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    lower_map = {col.lower(): col for col in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def load_predictions(name, filepath):
    if not filepath.exists():
        print(f"  !! {name}: file not found - {filepath}")
        return None
    df = pd.read_csv(filepath)
    id_col = find_column(df, ID_CANDIDATES)
    true_col = find_column(df, TRUE_LABEL_CANDIDATES)
    pred_col = find_column(df, PRED_LABEL_CANDIDATES)
    if true_col is None or pred_col is None:
        print(f"  !! {name}: missing required columns. Available: {list(df.columns)}")
        return None
    out = pd.DataFrame({
        "sample_id": df[id_col] if id_col else np.arange(len(df)),
        "y_true": df[true_col].astype(int),
        "y_pred": df[pred_col].astype(int),
    })
    print(f"  {name}: loaded {len(out)} rows from {filepath.name}")
    return out


def mcnemar_test(y_true, pred_a, pred_b, model_a_name, model_b_name):
    """
    Standard McNemar's test (with continuity correction) comparing two
    classifiers' correct/incorrect calls on the same paired samples.

    b = model_a correct, model_b wrong
    c = model_a wrong, model_b correct
    """
    correct_a = (pred_a == y_true)
    correct_b = (pred_b == y_true)

    b = int(np.sum(correct_a & ~correct_b))   # A right, B wrong
    c = int(np.sum(~correct_a & correct_b))   # A wrong, B right
    both_right = int(np.sum(correct_a & correct_b))
    both_wrong = int(np.sum(~correct_a & ~correct_b))

    if (b + c) == 0:
        stat, p_value = 0.0, 1.0
    else:
        # continuity-corrected McNemar statistic
        stat = (abs(b - c) - 1) ** 2 / (b + c)
        p_value = 1 - chi2.cdf(stat, df=1)

    return {
        "model_a": model_a_name,
        "model_b": model_b_name,
        f"{model_a_name}_only_correct": b,
        f"{model_b_name}_only_correct": c,
        "both_correct": both_right,
        "both_wrong": both_wrong,
        "chi2_statistic": round(stat, 4),
        "p_value": round(p_value, 6),
        "significant_at_0.05": p_value < 0.05,
    }


def main():
    print("=" * 90)
    print("MELONMA - MCNEMAR'S TEST FOR STATISTICAL SIGNIFICANCE")
    print("=" * 90)
    print("\nLoading per-sample predictions...")

    loaded = {}
    for name, filepath in PRED_FILES.items():
        result = load_predictions(name, filepath)
        if result is not None:
            loaded[name] = result

    if len(loaded) < 2:
        print("\n!! Need at least 2 models' predictions loaded to run any comparison.")
        print("   Edit PRED_FILES at the top of this script with the correct paths.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Sanity check: all models must share the same y_true ordering/values
    # ------------------------------------------------------------------
    names = list(loaded.keys())
    reference_true = loaded[names[0]]["y_true"].to_numpy()
    for name in names[1:]:
        this_true = loaded[name]["y_true"].to_numpy()
        if len(this_true) != len(reference_true):
            print(f"\n!! {name} has {len(this_true)} samples but {names[0]} has "
                  f"{len(reference_true)}. These must be the SAME test set to pair correctly.")
            sys.exit(1)
        if not np.array_equal(this_true, reference_true):
            print(f"\n!! WARNING: {name}'s true_label values don't match {names[0]}'s "
                  f"exactly. If the row order differs between files, pairing will be wrong. "
                  f"Verify both CSVs are sorted the same way (e.g. by image_id).")

    y_true = reference_true

    # ------------------------------------------------------------------
    # Headline comparison: ResNet101 vs MedFairXNet (if both available)
    # ------------------------------------------------------------------
    if "ResNet101" in loaded and "MedFairXNet" in loaded:
        print("\n" + "=" * 90)
        print("HEADLINE COMPARISON: ResNet101 vs MedFairXNet")
        print("=" * 90)
        result = mcnemar_test(
            y_true,
            loaded["ResNet101"]["y_pred"].to_numpy(),
            loaded["MedFairXNet"]["y_pred"].to_numpy(),
            "ResNet101", "MedFairXNet",
        )
        for k, v in result.items():
            print(f"  {k}: {v}")
        pd.DataFrame([result]).to_csv(
            RESULTS_DIR / "mcnemar_resnet101_vs_medfairxnet.csv", index=False
        )
        verdict = "SIGNIFICANT" if result["significant_at_0.05"] else "NOT significant"
        print(f"\n  Verdict: difference is {verdict} at alpha=0.05 "
              f"(p={result['p_value']:.4f})")
    else:
        missing = [n for n in ["ResNet101", "MedFairXNet"] if n not in loaded]
        print(f"\n!! Cannot run headline ResNet101 vs MedFairXNet comparison - "
              f"missing predictions for: {missing}")
        print("   Add the correct file path(s) to PRED_FILES and re-run.")

    # ------------------------------------------------------------------
    # Full pairwise matrix across every model that loaded successfully
    # ------------------------------------------------------------------
    print("\n" + "=" * 90)
    print(f"PAIRWISE MCNEMAR'S TESTS ACROSS ALL {len(names)} LOADED MODELS")
    print("=" * 90)

    pairwise_rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            result = mcnemar_test(
                y_true,
                loaded[a]["y_pred"].to_numpy(),
                loaded[b]["y_pred"].to_numpy(),
                a, b,
            )
            pairwise_rows.append({
                "model_a": a, "model_b": b,
                "chi2_statistic": result["chi2_statistic"],
                "p_value": result["p_value"],
                "significant_at_0.05": result["significant_at_0.05"],
            })

    pairwise_df = pd.DataFrame(pairwise_rows)
    print(pairwise_df.to_string(index=False))

    pairwise_out = RESULTS_DIR / "mcnemar_pairwise_all_models.csv"
    pairwise_df.to_csv(pairwise_out, index=False)

    print("\n" + "=" * 90)
    print("FILES SAVED")
    print("=" * 90)
    if "ResNet101" in loaded and "MedFairXNet" in loaded:
        print("Headline comparison:", RESULTS_DIR / "mcnemar_resnet101_vs_medfairxnet.csv")
    print("Pairwise matrix:     ", pairwise_out)
    print("\nSTATUS: PASS")


if __name__ == "__main__":
    main()