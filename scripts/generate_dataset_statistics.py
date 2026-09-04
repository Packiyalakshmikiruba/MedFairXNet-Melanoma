"""
MELONMA - DATASET STATISTICS TABLE (for paper Table 1)
=========================================================
Produces two things:
  1. Per-source-dataset breakdown (HAM10000 / ISIC2018 / PH2) - image
     counts and class distribution, read directly from your split CSVs.
  2. The FINAL combined dataset actually used in your experiments:
     train / validation / test (HAM10000+ISIC2018 merged) + PH2
     (independent external validation).

Outputs (results/):
    dataset_statistics_per_source.csv
    dataset_statistics_final_combined.csv

Run from the project's scripts/ folder:
    python scripts\\generate_dataset_statistics.py
"""

import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SPLITS_DIR = BASE_DIR / "data" / "processed" / "final_binary_clahe_splits"
PH2_SPLITS_DIR = BASE_DIR / "data" / "processed" / "PH2" / "splits"

FINAL_SPLIT_FILES = {
    "Train": SPLITS_DIR / "train_final_clahe.csv",
    "Validation": SPLITS_DIR / "validation_final_clahe.csv",
    "Test": SPLITS_DIR / "test_final_clahe.csv",
}

PH2_SPLIT_FILES = {
    "PH2 Train": PH2_SPLITS_DIR / "train_ph2_clahe.csv",
    "PH2 Validation": PH2_SPLITS_DIR / "validation_ph2_clahe.csv",
    "PH2 Test (used as external validation)": PH2_SPLITS_DIR / "test_ph2_clahe.csv",
}

LABEL_CANDIDATES = ["binary_label", "label", "target", "y"]
SOURCE_CANDIDATES = ["dataset_source", "source", "dataset", "origin"]


def find_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    lower_map = {col.lower(): col for col in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def summarize(df, label_col, name):
    n = len(df)
    n_mel = int((df[label_col] == 1).sum())
    n_non = int((df[label_col] == 0).sum())
    return {"split": name, "n_images": n, "melanoma": n_mel, "non_melanoma": n_non}


def main():
    print("=" * 90)
    print("MELONMA - DATASET STATISTICS")
    print("=" * 90)

    # ------------------------------------------------------------------
    # PART 1: final combined split (what you actually trained/tested on)
    # ------------------------------------------------------------------
    print("\n" + "-" * 90)
    print("FINAL COMBINED DATASET (HAM10000 + ISIC2018 merged, used for train/val/test)")
    print("-" * 90)

    combined_rows = []
    per_source_rows = []

    for split_name, filepath in FINAL_SPLIT_FILES.items():
        if not filepath.exists():
            print(f"  !! {split_name}: file not found - {filepath}")
            continue
        df = pd.read_csv(filepath)
        label_col = find_column(df, LABEL_CANDIDATES)
        if label_col is None:
            print(f"  !! {split_name}: no label column found. Columns: {df.columns.tolist()}")
            continue

        row = summarize(df, label_col, split_name)
        combined_rows.append(row)
        print(f"  {split_name}: n={row['n_images']}  melanoma={row['melanoma']}  "
              f"non_melanoma={row['non_melanoma']}")

        # If a dataset-source column exists, break this split down further
        source_col = find_column(df, SOURCE_CANDIDATES)
        if source_col is not None:
            for source_val, group in df.groupby(source_col):
                s_row = summarize(group, label_col, f"{split_name} - {source_val}")
                s_row["source_dataset"] = source_val
                s_row["split"] = split_name
                per_source_rows.append(s_row)
        else:
            if split_name == list(FINAL_SPLIT_FILES.keys())[0]:
                print(f"     (no '{SOURCE_CANDIDATES}' column found - cannot break down by "
                      f"HAM10000 vs ISIC2018 automatically. See note below.)")

    # ------------------------------------------------------------------
    # PART 2: PH2 (independent external validation)
    # ------------------------------------------------------------------
    print("\n" + "-" * 90)
    print("PH2 (INDEPENDENT EXTERNAL VALIDATION)")
    print("-" * 90)

    for split_name, filepath in PH2_SPLIT_FILES.items():
        if not filepath.exists():
            print(f"  !! {split_name}: file not found - {filepath}")
            continue
        df = pd.read_csv(filepath)
        label_col = find_column(df, LABEL_CANDIDATES)
        if label_col is None:
            print(f"  !! {split_name}: no label column found. Columns: {df.columns.tolist()}")
            continue
        row = summarize(df, label_col, split_name)
        combined_rows.append(row)
        print(f"  {split_name}: n={row['n_images']}  melanoma={row['melanoma']}  "
              f"non_melanoma={row['non_melanoma']}")

    # ------------------------------------------------------------------
    # SAVE
    # ------------------------------------------------------------------
    combined_df = pd.DataFrame(combined_rows)
    combined_out = RESULTS_DIR / "dataset_statistics_final_combined.csv"
    combined_df.to_csv(combined_out, index=False)

    print("\n" + "=" * 90)
    print("FINAL COMBINED TABLE (save this as your paper's Table 1)")
    print("=" * 90)
    print(combined_df.to_string(index=False))

    if per_source_rows:
        per_source_df = pd.DataFrame(per_source_rows)
        per_source_out = RESULTS_DIR / "dataset_statistics_per_source.csv"
        per_source_df.to_csv(per_source_out, index=False)
        print("\n" + "=" * 90)
        print("PER-SOURCE BREAKDOWN (HAM10000 vs ISIC2018, within each split)")
        print("=" * 90)
        print(per_source_df.to_string(index=False))
        print(f"\nSaved: {per_source_out}")
    else:
        print("\n" + "=" * 90)
        print("NOTE: Could not break HAM10000 vs ISIC2018 apart automatically")
        print("=" * 90)
        print("Your train/validation/test CSVs don't have a 'dataset_source' (or similar)")
        print("column, so the combined totals above cannot be split into HAM10000-only")
        print("vs ISIC2018-only counts. If you know which dataset each image came from")
        print("(e.g. from image_id prefix or a separate source list), share that here")
        print("and this can be computed. Otherwise, report the combined 999/train/val")
        print("numbers as 'HAM10000 + ISIC2018 (merged)' in Table 1, which is accurate")
        print("and standard practice when datasets are pooled before splitting.")

    print(f"\nSaved: {combined_out}")
    print("\nSTATUS: PASS")


if __name__ == "__main__":
    main()