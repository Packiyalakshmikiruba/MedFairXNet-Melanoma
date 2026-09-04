import pandas as pd
from pathlib import Path


# ============================================================
# PH2 DATASET - CORRECT LABEL EXTRACTION
# ============================================================

EXCEL_PATH = Path(
    "data/raw/PH2/PH2Dataset/PH2_dataset.xlsx"
)

OUTPUT_PATH = Path(
    "data/processed/ph2_labels.csv"
)


def clean(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def main():

    print("=" * 80)
    print("PH2 DATASET - CORRECT LABEL EXTRACTION")
    print("=" * 80)

    # --------------------------------------------------------
    # Read PH2 Excel
    # Row 13 in Excel = header -> pandas header=12
    # --------------------------------------------------------

    df = pd.read_excel(
        EXCEL_PATH,
        sheet_name=0,
        header=12
    )

    # Clean column names
    df.columns = [
        str(c).replace("\n", " ").strip()
        for c in df.columns
    ]

    print("\nColumns detected:")
    for c in df.columns:
        print(f"  - {c}")

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    required_columns = [
        "Image Name",
        "Histological Diagnosis",
        "Common Nevus",
        "Atypical Nevus",
        "Melanoma"
    ]

    for column in required_columns:
        if column not in df.columns:
            raise RuntimeError(
                f"Required column not found: {column}"
            )

    records = []

    # --------------------------------------------------------
    # Extract labels
    # --------------------------------------------------------

    for _, row in df.iterrows():

        image_id = clean(row["Image Name"])

        # Ignore non-image rows
        if not image_id.startswith("IMD"):
            continue

        histological = clean(
            row["Histological Diagnosis"]
        )

        common_nevus = clean(
            row["Common Nevus"]
        )

        atypical_nevus = clean(
            row["Atypical Nevus"]
        )

        melanoma = clean(
            row["Melanoma"]
        )

        # ----------------------------------------------------
        # Determine PH2 class
        # Priority:
        # Melanoma -> Atypical Nevus -> Common Nevus
        # ----------------------------------------------------

        if melanoma.upper() == "X":
            clinical_class = "melanoma"

        elif atypical_nevus.upper() == "X":
            clinical_class = "atypical_nevus"

        elif common_nevus.upper() == "X":
            clinical_class = "common_nevus"

        else:
            clinical_class = "unknown"

        records.append({
            "image_id": image_id,
            "histological_diagnosis": histological,
            "common_nevus": common_nevus,
            "atypical_nevus": atypical_nevus,
            "melanoma": melanoma,
            "clinical_class": clinical_class
        })

    # --------------------------------------------------------
    # Create DataFrame
    # --------------------------------------------------------

    result = pd.DataFrame(records)

    # Remove duplicate image IDs
    result = result.drop_duplicates(
        subset=["image_id"],
        keep="first"
    )

    result = result.sort_values(
        "image_id"
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    result.to_csv(
        OUTPUT_PATH,
        index=False
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("PH2 LABEL SUMMARY")
    print("=" * 80)

    counts = (
        result["clinical_class"]
        .value_counts()
    )

    print(counts.to_string())

    print("\nTotal images:", len(result))

    print("\nClass percentages:")

    percentages = (
        result["clinical_class"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

    for cls, percentage in percentages.items():
        print(
            f"  {cls:20s}: {percentage:6.2f}%"
        )

    # --------------------------------------------------------
    # Check for unknown labels
    # --------------------------------------------------------

    unknown = result[
        result["clinical_class"] == "unknown"
    ]

    print("\nUnknown labels:", len(unknown))

    if len(unknown) > 0:

        print("\nUnknown image IDs:")

        print(
            unknown["image_id"]
            .to_list()
        )

    # --------------------------------------------------------
    # Show sample
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("SAMPLE LABELS")
    print("=" * 80)

    print(
        result[
            [
                "image_id",
                "histological_diagnosis",
                "clinical_class"
            ]
        ]
        .head(20)
        .to_string(index=False)
    )

    print("\n" + "=" * 80)
    print("OUTPUT")
    print("=" * 80)

    print(OUTPUT_PATH)

    print("\n" + "=" * 80)
    print("PH2 LABEL EXTRACTION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()