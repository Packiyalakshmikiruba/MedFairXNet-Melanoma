from pathlib import Path
import time

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms
import timm

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


# ============================================================
# MELONMA - SWIN TRANSFORMER FINAL CLAHE TEST EVALUATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "final_binary_clahe_splits"
)

TEST_CSV = DATA_DIR / "test_final_clahe.csv"

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "melanoma_swin_final_clahe.pth"
)

RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

METRICS_PATH = (
    RESULTS_DIR
    / "swin_final_clahe_test_evaluation.csv"
)

PREDICTIONS_PATH = (
    RESULTS_DIR
    / "swin_final_clahe_test_predictions.csv"
)

CONFUSION_PATH = (
    RESULTS_DIR
    / "swin_final_clahe_confusion_matrix.csv"
)

REPORT_PATH = (
    RESULTS_DIR
    / "swin_final_clahe_classification_report.txt"
)


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = 224
BATCH_SIZE = 8

MODEL_NAME = "swin_tiny_patch4_window7_224"

THRESHOLD = 0.5

NUM_WORKERS = 0
PIN_MEMORY = False

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 90)
print("MELONMA - SWIN TRANSFORMER FINAL CLAHE TEST EVALUATION")
print("=" * 90)

print("\nPyTorch version:", torch.__version__)
print("TorchVision version:", __import__("torchvision").__version__)
print("timm version:", timm.__version__)

print("\nTest CSV:")
print(TEST_CSV)

print("\nModel:")
print(MODEL_PATH)

print("\nSwin model:")
print(MODEL_NAME)

print("\nDevice:")
print(DEVICE)

print("\nImage size:")
print(f"{IMAGE_SIZE} x {IMAGE_SIZE}")

print("\nBatch size:")
print(BATCH_SIZE)

print("\nThreshold:")
print(THRESHOLD)


# ============================================================
# FILE CHECKS
# ============================================================

if not TEST_CSV.exists():

    raise FileNotFoundError(
        f"Test CSV not found:\n{TEST_CSV}"
    )


if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"Swin model not found:\n{MODEL_PATH}"
    )


# ============================================================
# LOAD TEST CSV
# ============================================================

test_df = pd.read_csv(TEST_CSV)

print("\n" + "=" * 90)
print("TEST DATA INFORMATION")
print("=" * 90)

print(
    "\nTest images:",
    len(test_df)
)

print(
    "\nColumns:"
)

print(
    test_df.columns.tolist()
)


# ============================================================
# FIND IMAGE PATH COLUMN
# ============================================================

possible_path_columns = [
    "image_path",
    "clahe_image_path",
    "filepath",
    "file_path",
    "path",
]

path_column = None

for column in possible_path_columns:

    if column in test_df.columns:

        path_column = column
        break


if path_column is None:

    raise ValueError(
        "Could not find image path column.\n"
        f"Available columns: {test_df.columns.tolist()}"
    )


# ============================================================
# FIND LABEL COLUMN
# ============================================================

possible_label_columns = [
    "binary_label",
    "label",
    "target",
    "y",
]

label_column = None

for column in possible_label_columns:

    if column in test_df.columns:

        label_column = column
        break


if label_column is None:

    raise ValueError(
        "Could not find label column.\n"
        f"Available columns: {test_df.columns.tolist()}"
    )


print(
    "\nImage path column:",
    path_column
)

print(
    "Label column:",
    label_column
)


# ============================================================
# ABSOLUTE PATH CONVERSION
# ============================================================

def make_absolute_path(path):

    path = Path(str(path))

    if path.is_absolute():

        return str(path)

    return str(
        (BASE_DIR / path).resolve()
    )


test_df[path_column] = (
    test_df[path_column]
    .apply(make_absolute_path)
)


# ============================================================
# IMAGE VALIDATION
# ============================================================

missing_images = test_df[
    ~test_df[path_column].apply(
        lambda x: Path(x).exists()
    )
]


print(
    "\nMissing test images:",
    len(missing_images)
)


if len(missing_images) > 0:

    print(
        missing_images[
            [path_column]
        ]
        .head(10)
        .to_string(index=False)
    )

    raise FileNotFoundError(
        "Some test images are missing."
    )


print(
    "All test images are available."
)


# ============================================================
# LABEL PREPARATION
# ============================================================

y_true = (
    test_df[label_column]
    .astype(int)
    .to_numpy()
)


print("\n" + "=" * 90)
print("TEST CLASS DISTRIBUTION")
print("=" * 90)

unique, counts = np.unique(
    y_true,
    return_counts=True
)

for label, count in zip(
    unique,
    counts
):

    if label == 1:

        class_name = "Melanoma"

    else:

        class_name = "Non-melanoma"

    print(
        f"{class_name}: {count}"
    )


# ============================================================
# TRANSFORM
# ============================================================

test_transform = transforms.Compose(
    [

        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406
            ],

            std=[
                0.229,
                0.224,
                0.225
            ]
        ),

    ]
)


# ============================================================
# DATASET
# ============================================================

class MelanomaTestDataset(
    Dataset
):

    def __init__(
        self,
        dataframe,
        transform=None
    ):

        self.df = dataframe.reset_index(
            drop=True
        )

        self.transform = transform


    def __len__(self):

        return len(self.df)


    def __getitem__(
        self,
        index
    ):

        row = self.df.iloc[index]

        image_path = row[path_column]

        label = int(
            row[label_column]
        )

        image = Image.open(
            image_path
        ).convert("RGB")


        if self.transform is not None:

            image = self.transform(
                image
            )


        return image, label


# ============================================================
# CREATE TEST DATASET
# ============================================================

test_dataset = MelanomaTestDataset(
    test_df,
    transform=test_transform
)


# ============================================================
# CREATE TEST DATALOADER
# ============================================================

test_loader = DataLoader(

    test_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=NUM_WORKERS,

    pin_memory=PIN_MEMORY,

)


print("\n" + "=" * 90)
print("TEST DATALOADER")
print("=" * 90)

print(
    "\nTest samples:",
    len(test_dataset)
)

print(
    "Test batches:",
    len(test_loader)
)


# ============================================================
# SWIN BACKBONE
# ============================================================

print("\n" + "=" * 90)
print("LOADING SWIN-TINY BACKBONE")
print("=" * 90)

swin_backbone = timm.create_model(

    MODEL_NAME,

    pretrained=False,

    num_classes=0,

    global_pool="avg",

)


# ============================================================
# CLASSIFIER
# ============================================================

class SwinMelanomaClassifier(
    nn.Module
):

    def __init__(
        self,
        backbone,
        feature_dim
    ):

        super().__init__()

        self.backbone = backbone

        self.classifier = nn.Sequential(

            nn.LayerNorm(
                feature_dim
            ),

            nn.Dropout(
                p=0.35
            ),

            nn.Linear(
                feature_dim,
                1
            ),

        )


    def forward(self, x):

        with torch.no_grad():

            features = self.backbone(
                x
            )

        logits = self.classifier(
            features
        )

        return logits.squeeze(1)


# ============================================================
# CREATE MODEL
# ============================================================

feature_dim = (
    swin_backbone.num_features
)


model = SwinMelanomaClassifier(
    swin_backbone,
    feature_dim
)


# ============================================================
# LOAD CHECKPOINT
# ============================================================

print("\nLoading checkpoint...")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)


print(
    "\nCheckpoint type:",
    type(checkpoint)
)


# The training script saves a dictionary checkpoint.
if isinstance(
    checkpoint,
    dict
):

    if "model_state_dict" in checkpoint:

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        print(
            "Loaded model_state_dict."
        )

    else:

        raise KeyError(
            "Checkpoint does not contain "
            "'model_state_dict'."
        )

else:

    model.load_state_dict(
        checkpoint
    )

    print(
        "Loaded direct state_dict."
    )


model = model.to(
    DEVICE
)


model.eval()


print(
    "\nModel loaded successfully."
)

print(
    "Feature dimension:",
    feature_dim
)


# ============================================================
# PARAMETER INFORMATION
# ============================================================

total_parameters = sum(
    p.numel()
    for p in model.parameters()
)

trainable_parameters = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)


print(
    "\nTotal parameters:",
    f"{total_parameters:,}"
)

print(
    "Trainable parameters:",
    f"{trainable_parameters:,}"
)


# ============================================================
# PREDICTION
# ============================================================

print("\n" + "=" * 90)
print("RUNNING FINAL SWIN TEST PREDICTION")
print("=" * 90)

evaluation_start = time.time()

all_labels = []
all_probabilities = []


with torch.no_grad():

    for batch_index, (
        images,
        labels
    ) in enumerate(test_loader):

        images = images.to(
            DEVICE
        )

        logits = model(
            images
        )

        probabilities = torch.sigmoid(
            logits
        )


        all_labels.extend(
            labels.numpy().tolist()
        )

        all_probabilities.extend(
            probabilities.cpu()
            .numpy()
            .tolist()
        )


        if (
            (batch_index + 1) % 25 == 0
            or
            (batch_index + 1) == len(test_loader)
        ):

            print(
                f"Batch "
                f"{batch_index + 1}/"
                f"{len(test_loader)}"
            )


evaluation_time = (
    time.time()
    - evaluation_start
)


y_true = np.asarray(
    all_labels,
    dtype=int
)

y_prob = np.asarray(
    all_probabilities,
    dtype=float
)


print(
    "\nPrediction count:",
    len(y_prob)
)

print(
    "Test image count:",
    len(test_df)
)


if len(y_prob) != len(test_df):

    raise RuntimeError(
        "Prediction count does not match "
        "test image count."
    )


# ============================================================
# PREDICTION SANITY CHECK
# ============================================================

print("\n" + "=" * 90)
print("PREDICTION SCORE SANITY CHECK")
print("=" * 90)

print(
    f"\nMinimum prediction: "
    f"{y_prob.min():.6f}"
)

print(
    f"Maximum prediction: "
    f"{y_prob.max():.6f}"
)

print(
    f"Mean prediction: "
    f"{y_prob.mean():.6f}"
)

print(
    f"Median prediction: "
    f"{np.median(y_prob):.6f}"
)


# ============================================================
# BINARY PREDICTIONS
# ============================================================

y_pred = (
    y_prob >= THRESHOLD
).astype(int)


# ============================================================
# CONFUSION MATRIX
# ============================================================

tn, fp, fn, tp = confusion_matrix(

    y_true,

    y_pred,

    labels=[
        0,
        1
    ]

).ravel()


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred,
    zero_division=0
)

sensitivity = recall_score(
    y_true,
    y_pred,
    zero_division=0
)

specificity = (

    tn / (tn + fp)

    if (tn + fp) > 0

    else 0.0

)

f1 = f1_score(
    y_true,
    y_pred,
    zero_division=0
)


try:

    roc_auc = roc_auc_score(
        y_true,
        y_prob
    )

except ValueError:

    roc_auc = np.nan


try:

    pr_auc = average_precision_score(
        y_true,
        y_prob
    )

except ValueError:

    pr_auc = np.nan


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 90)
print("FINAL SWIN TRANSFORMER TEST RESULTS")
print("=" * 90)

print(
    f"\nTest images   : {len(y_true)}"
)

print(
    f"Threshold     : {THRESHOLD}"
)

print(
    f"Accuracy      : {accuracy:.6f}"
)

print(
    f"ROC-AUC       : {roc_auc:.6f}"
)

print(
    f"PR-AUC        : {pr_auc:.6f}"
)

print(
    f"Precision      : {precision:.6f}"
)

print(
    f"Sensitivity    : {sensitivity:.6f}"
)

print(
    f"Specificity    : {specificity:.6f}"
)

print(
    f"F1-score       : {f1:.6f}"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 90)
print("CONFUSION MATRIX")
print("=" * 90)

print(
    "\n              Predicted"
)

print(
    "              Non-Mel    Melanoma"
)

print(
    f"Actual Non-Mel   "
    f"{tn:4d}       {fp:4d}"
)

print(
    f"Actual Melanoma  "
    f"{fn:4d}       {tp:4d}"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(

    y_true,

    y_pred,

    target_names=[
        "Non-melanoma",
        "Melanoma"
    ],

    zero_division=0

)


print("\n" + "=" * 90)
print("CLASSIFICATION REPORT")
print("=" * 90)

print(
    report
)


# ============================================================
# SAVE METRICS
# ============================================================

metrics_df = pd.DataFrame(
    [
        {

            "model":
                "Swin-Tiny",

            "dataset":
                "Final CLAHE Test",

            "test_images":
                len(y_true),

            "threshold":
                THRESHOLD,

            "accuracy":
                accuracy,

            "roc_auc":
                roc_auc,

            "pr_auc":
                pr_auc,

            "precision":
                precision,

            "sensitivity":
                sensitivity,

            "specificity":
                specificity,

            "f1_score":
                f1,

            "true_negative":
                tn,

            "false_positive":
                fp,

            "false_negative":
                fn,

            "true_positive":
                tp,

            "evaluation_time_seconds":
                evaluation_time,

        }
    ]
)


metrics_df.to_csv(
    METRICS_PATH,
    index=False
)


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

cm_df = pd.DataFrame(

    [
        [tn, fp],
        [fn, tp]
    ],

    columns=[
        "Predicted_Non_Melanoma",
        "Predicted_Melanoma"
    ],

    index=[
        "Actual_Non_Melanoma",
        "Actual_Melanoma"
    ]

)


cm_df.to_csv(
    CONFUSION_PATH
)


# ============================================================
# SAVE CLASSIFICATION REPORT
# ============================================================

with open(
    REPORT_PATH,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        report
    )


# ============================================================
# SAVE PREDICTIONS
# ============================================================

if "image_id" in test_df.columns:

    image_ids = (
        test_df["image_id"]
        .astype(str)
        .to_numpy()
    )

else:

    image_ids = np.arange(
        len(test_df)
    )


predictions_df = pd.DataFrame(

    {

        "image_id":
            image_ids,

        "true_label":
            y_true,

        "raw_prediction":
            y_prob,

        "predicted_label":
            y_pred,

        "true_class": [

            "Melanoma"
            if x == 1
            else "Non-melanoma"

            for x in y_true

        ],

        "predicted_class": [

            "Melanoma"
            if x == 1
            else "Non-melanoma"

            for x in y_pred

        ],

    }

)


predictions_df.to_csv(
    PREDICTIONS_PATH,
    index=False
)


# ============================================================
# FINAL STATUS
# ============================================================

print("\n" + "=" * 90)
print("FILES SAVED")
print("=" * 90)

print(
    "\nMetrics:"
)

print(
    METRICS_PATH
)

print(
    "\nConfusion matrix:"
)

print(
    CONFUSION_PATH
)

print(
    "\nClassification report:"
)

print(
    REPORT_PATH
)

print(
    "\nPredictions:"
)

print(
    PREDICTIONS_PATH
)

print(
    "\nEvaluation time:"
)

print(
    f"{evaluation_time:.2f} seconds"
)


print("\n" + "=" * 90)
print("STATUS: PASS")
print(
    "Swin Transformer final CLAHE "
    "test evaluation completed successfully."
)
print("=" * 90)