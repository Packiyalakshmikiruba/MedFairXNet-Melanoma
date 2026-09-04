from pathlib import Path
import pandas as pd


# ============================================================
# PH2 IMAGE + LABEL MAPPING AUDIT
# ============================================================

PH2_IMAGE_ROOT = Path(
    "data/raw/PH2/PH2Dataset/PH2 Dataset images"
)

LABEL_FILE = Path(
    "data/processed/ph2_labels.csv"
)


def main():

    print("=" * 80)
    print("PH2 IMAGE + LABEL MAPPING AUDIT")
    print("=" * 80)

    # --------------------------------------------------------
    # Load labels
    # --------------------------------------------------------

    labels = pd.read_csv(LABEL_FILE)

    print("\nLabels loaded:", len(labels))

    # --------------------------------------------------------
    # Find PH2 case folders
    # --------------------------------------------------------

    case_dirs = sorted(
        [
            p for p in PH2_IMAGE_ROOT.iterdir()
            if p.is_dir()
        ]
    )

    print("PH2 case folders:", len(case_dirs))

    # --------------------------------------------------------
    # Build mapping
    # --------------------------------------------------------

    records = []

    for case_dir in case_dirs:

        image_id = case_dir.name

        # Actual PH2 folder structure
        original_image = (
            case_dir
            / f"{image_id}_Dermoscopic_Image"
            / f"{image_id}.bmp"
        )

        lesion_mask = (
            case_dir
            / f"{image_id}_lesion"
            / f"{image_id}_lesion.bmp"
        )

        label4 = (
            case_dir
            / f"{image_id}_roi"
            / f"{image_id}_R1_Label4.bmp"
        )

        label3 = (
            case_dir
            / f"{image_id}_roi"
            / f"{image_id}_R2_Label3.bmp"
        )

        # ----------------------------------------------------
        # Find label
        # ----------------------------------------------------

        label_rows = labels[
            labels["image_id"] == image_id
        ]

        if len(label_rows) == 0:

            clinical_class = "MISSING_LABEL"

        else:

            clinical_class = (
                label_rows.iloc[0]["clinical_class"]
            )

        # ----------------------------------------------------
        # Store record
        # ----------------------------------------------------

        records.append({

            "image_id": image_id,

            "original_image_exists":
                original_image.exists(),

            "original_image_path":
                str(original_image),

            "lesion_mask_exists":
                lesion_mask.exists(),

            "lesion_mask_path":
                str(lesion_mask),

            "label4_exists":
                label4.exists(),

            "label4_path":
                str(label4),

            "label3_exists":
                label3.exists(),

            "label3_path":
                str(label3),

            "clinical_class":
                clinical_class
        })

    result = pd.DataFrame(records)

    # --------------------------------------------------------
    # File availability
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("FILE AVAILABILITY")
    print("=" * 80)

    print(
        "Original images :",
        result["original_image_exists"].sum(),
        "/",
        len(result)
    )

    print(
        "Lesion masks    :",
        result["lesion_mask_exists"].sum(),
        "/",
        len(result)
    )

    print(
        "Label4 masks    :",
        result["label4_exists"].sum(),
        "/",
        len(result)
    )

    print(
        "Label3 masks    :",
        result["label3_exists"].sum(),
        "/",
        len(result)
    )

    # --------------------------------------------------------
    # Missing files
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("MISSING FILES")
    print("=" * 80)

    for column in [
        "original_image_exists",
        "lesion_mask_exists",
        "label4_exists",
        "label3_exists"
    ]:

        missing = result[
            ~result[column]
        ]

        print(
            f"{column}: {len(missing)} missing"
        )

        if len(missing) > 0:

            print(
                missing["image_id"].tolist()
            )

    # --------------------------------------------------------
    # Label distribution
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("LABEL DISTRIBUTION")
    print("=" * 80)

    print(
        result["clinical_class"]
        .value_counts()
        .to_string()
    )

    # --------------------------------------------------------
    # Save audit
    # --------------------------------------------------------

    output = Path(
        "data/processed/ph2_mapping_audit.csv"
    )

    result.to_csv(
        output,
        index=False
    )

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    required_ok = (
        result["original_image_exists"].all()
        and result["lesion_mask_exists"].all()
        and result["clinical_class"].ne(
            "MISSING_LABEL"
        ).all()
    )

    print("\n" + "=" * 80)

    if required_ok:

        print("STATUS: PASS")

        print(
            "All PH2 images have valid labels "
            "and lesion masks."
        )

    else:

        print("STATUS: CHECK REQUIRED")

    print("=" * 80)

    print("\nAudit saved to:")
    print(output)


if __name__ == "__main__":
    main()