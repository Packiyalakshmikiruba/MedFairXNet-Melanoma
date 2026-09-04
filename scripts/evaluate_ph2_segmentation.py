from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.spatial.distance import directed_hausdorff
from tqdm import tqdm


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parents[1]

PH2_DIR = (
    BASE_DIR
    / "data"
    / "raw"
    / "PH2"
    / "PH2Dataset"
    / "PH2 Dataset images"
)

RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# SEGMENTATION
# =========================================================

def create_lesion_mask(image):

    lab = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB
    )

    l_channel = lab[:, :, 0]

    blurred = cv2.GaussianBlur(
        l_channel,
        (5, 5),
        0
    )

    _, mask = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (7, 7)
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    image_area = image.shape[0] * image.shape[1]

    valid_contours = [
        c
        for c in contours
        if cv2.contourArea(c) > image_area * 0.01
    ]

    if not valid_contours:
        return None

    largest = max(
        valid_contours,
        key=cv2.contourArea
    )

    predicted = np.zeros(
        mask.shape,
        dtype=np.uint8
    )

    cv2.drawContours(
        predicted,
        [largest],
        -1,
        255,
        thickness=cv2.FILLED
    )

    return predicted


# =========================================================
# DICE
# =========================================================

def dice_score(gt, pred):

    gt = gt > 0
    pred = pred > 0

    intersection = np.logical_and(
        gt,
        pred
    ).sum()

    denominator = gt.sum() + pred.sum()

    if denominator == 0:
        return 1.0

    return (
        2.0 * intersection / denominator
    )


# =========================================================
# IOU
# =========================================================

def iou_score(gt, pred):

    gt = gt > 0
    pred = pred > 0

    intersection = np.logical_and(
        gt,
        pred
    ).sum()

    union = np.logical_or(
        gt,
        pred
    ).sum()

    if union == 0:
        return 1.0

    return intersection / union


# =========================================================
# BOUNDARY ACCURACY
# =========================================================

def boundary_accuracy(gt, pred):

    kernel = np.ones(
        (3, 3),
        dtype=np.uint8
    )

    gt_boundary = cv2.morphologyEx(
        gt,
        cv2.MORPH_GRADIENT,
        kernel
    ) > 0

    pred_boundary = cv2.morphologyEx(
        pred,
        cv2.MORPH_GRADIENT,
        kernel
    ) > 0

    gt_count = gt_boundary.sum()

    if gt_count == 0:
        return np.nan

    overlap = np.logical_and(
        gt_boundary,
        pred_boundary
    ).sum()

    return overlap / gt_count


# =========================================================
# HAUSDORFF DISTANCE
# =========================================================

def hausdorff_distance(gt, pred):

    gt_points = np.column_stack(
        np.where(gt > 0)
    )

    pred_points = np.column_stack(
        np.where(pred > 0)
    )

    if len(gt_points) == 0 or len(pred_points) == 0:
        return np.nan

    forward = directed_hausdorff(
        gt_points,
        pred_points
    )[0]

    backward = directed_hausdorff(
        pred_points,
        gt_points
    )[0]

    return max(
        forward,
        backward
    )


# =========================================================
# MAIN
# =========================================================

print("=" * 70)
print("PH2 LESION SEGMENTATION EVALUATION")
print("=" * 70)

patient_dirs = sorted(
    PH2_DIR.glob("IMD*")
)

print(
    "\nPH2 patient folders:",
    len(patient_dirs)
)

results = []
failed = 0


for patient_dir in tqdm(
    patient_dirs,
    desc="Evaluating PH2"
):

    patient_id = patient_dir.name

    # -----------------------------------------------------
    # ORIGINAL IMAGE
    # -----------------------------------------------------

    image_dir = (
        patient_dir
        / f"{patient_id}_Dermoscopic_Image"
    )

    image_files = sorted(
        image_dir.glob("*.bmp")
    )

    if not image_files:
        print(
            f"\nMissing image: {patient_id}"
        )
        failed += 1
        continue

    image_file = image_files[0]

    # -----------------------------------------------------
    # GROUND-TRUTH LESION MASK
    # -----------------------------------------------------

    lesion_dir = (
        patient_dir
        / f"{patient_id}_lesion"
    )

    gt_files = sorted(
        lesion_dir.glob("*.bmp")
    )

    if not gt_files:
        print(
            f"\nMissing GT mask: {patient_id}"
        )
        failed += 1
        continue

    gt_file = gt_files[0]

    # -----------------------------------------------------
    # READ
    # -----------------------------------------------------

    image = cv2.imread(
        str(image_file),
        cv2.IMREAD_COLOR
    )

    gt = cv2.imread(
        str(gt_file),
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        print(
            f"\nCannot read image: {patient_id}"
        )
        failed += 1
        continue

    if gt is None:
        print(
            f"\nCannot read GT mask: {patient_id}"
        )
        failed += 1
        continue

    # -----------------------------------------------------
    # PREDICT MASK
    # -----------------------------------------------------

    pred = create_lesion_mask(
        image
    )

    if pred is None:
        print(
            f"\nSegmentation failed: {patient_id}"
        )
        failed += 1
        continue

    # -----------------------------------------------------
    # SIZE CHECK
    # -----------------------------------------------------

    if pred.shape != gt.shape:

        pred = cv2.resize(
            pred,
            (
                gt.shape[1],
                gt.shape[0]
            ),
            interpolation=cv2.INTER_NEAREST
        )

    # -----------------------------------------------------
    # BINARY MASK
    # -----------------------------------------------------

    gt = np.where(
        gt > 0,
        255,
        0
    ).astype(np.uint8)

    pred = np.where(
        pred > 0,
        255,
        0
    ).astype(np.uint8)

    # -----------------------------------------------------
    # METRICS
    # -----------------------------------------------------

    dice = dice_score(
        gt,
        pred
    )

    iou = iou_score(
        gt,
        pred
    )

    boundary = boundary_accuracy(
        gt,
        pred
    )

    hd = hausdorff_distance(
        gt,
        pred
    )

    results.append({
        "patient_id": patient_id,
        "dice": dice,
        "iou": iou,
        "boundary_accuracy": boundary,
        "hausdorff_distance": hd
    })


# =========================================================
# DATAFRAME
# =========================================================

df = pd.DataFrame(
    results
)

result_file = (
    RESULTS_DIR
    / "ph2_segmentation_evaluation.csv"
)

df.to_csv(
    result_file,
    index=False
)


# =========================================================
# SUMMARY
# =========================================================

if len(df) > 0:

    summary = {
        "patients_total":
            len(patient_dirs),

        "patients_evaluated":
            len(df),

        "failed":
            failed,

        "mean_dice":
            df["dice"].mean(),

        "std_dice":
            df["dice"].std(),

        "mean_iou":
            df["iou"].mean(),

        "std_iou":
            df["iou"].std(),

        "mean_boundary_accuracy":
            df["boundary_accuracy"].mean(),

        "std_boundary_accuracy":
            df["boundary_accuracy"].std(),

        "mean_hausdorff":
            df["hausdorff_distance"].mean(),

        "std_hausdorff":
            df["hausdorff_distance"].std(),
    }

else:

    summary = {
        "patients_total":
            len(patient_dirs),

        "patients_evaluated":
            0,

        "failed":
            failed,

        "mean_dice":
            np.nan,

        "std_dice":
            np.nan,

        "mean_iou":
            np.nan,

        "std_iou":
            np.nan,

        "mean_boundary_accuracy":
            np.nan,

        "std_boundary_accuracy":
            np.nan,

        "mean_hausdorff":
            np.nan,

        "std_hausdorff":
            np.nan,
    }


summary_file = (
    RESULTS_DIR
    / "ph2_segmentation_summary.csv"
)

pd.DataFrame(
    [summary]
).to_csv(
    summary_file,
    index=False
)


# =========================================================
# FINAL OUTPUT
# =========================================================

print("\n" + "=" * 70)
print("PH2 SEGMENTATION EVALUATION COMPLETE")
print("=" * 70)

print(
    "\nPatients total      :",
    summary["patients_total"]
)

print(
    "Patients evaluated  :",
    summary["patients_evaluated"]
)

print(
    "Failed              :",
    summary["failed"]
)

print(
    "\nMean Dice           :",
    summary["mean_dice"]
)

print(
    "Mean IoU            :",
    summary["mean_iou"]
)

print(
    "Boundary Accuracy   :",
    summary["mean_boundary_accuracy"]
)

print(
    "Mean Hausdorff      :",
    summary["mean_hausdorff"]
)

print(
    "\nDetailed results:",
    result_file
)

print(
    "Summary:",
    summary_file
)