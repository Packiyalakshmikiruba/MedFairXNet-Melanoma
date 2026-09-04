from pathlib import Path
import time

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import timm

from PIL import Image

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
)

import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent


# ============================================================
# PH2 TEST CSV
# ============================================================

TEST_CSV = (
    BASE_DIR
    / "data"
    / "processed"
    / "PH2"
    / "splits"
    / "test_ph2_clahe.csv"
)


# ============================================================
# TRAINED SWIN MODEL
# ============================================================

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "melanoma_swin_final_clahe.pth"
)


# ============================================================
# RESULTS
# ============================================================

RESULTS_DIR = (
    BASE_DIR
    / "results"
    / "swin_ph2"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


PREDICTIONS_PATH = (
    RESULTS_DIR
    / "swin_ph2_predictions.csv"
)

METRICS_PATH = (
    RESULTS_DIR
    / "swin_ph2_metrics.csv"
)

REPORT_PATH = (
    RESULTS_DIR
    / "swin_ph2_classification_report.txt"
)

ROC_CURVE_PATH = (
    RESULTS_DIR
    / "swin_ph2_roc_curve.png"
)

PR_CURVE_PATH = (
    RESULTS_DIR
    / "swin_ph2_pr_curve.png"
)

CONFUSION_MATRIX_PATH = (
    RESULTS_DIR
    / "swin_ph2_confusion_matrix.png"
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_NAME = (
    "swin_tiny_patch4_window7_224"
)

IMAGE_SIZE = 224

BATCH_SIZE = 8

THRESHOLD = 0.5

RANDOM_SEED = 42


# ============================================================
# IMAGENET NORMALIZATION
# ============================================================

MEAN = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32
)

STD = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

torch.manual_seed(
    RANDOM_SEED
)

if torch.cuda.is_available():

    torch.cuda.manual_seed_all(
        RANDOM_SEED
    )


# ============================================================
# HEADER
# ============================================================

print("=" * 90)

print(
    "MELONMA - SWIN TRANSFORMER "
    "PH2 INDEPENDENT TEST EVALUATION"
)

print("=" * 90)

print(
    "\nPyTorch version:"
)

print(
    torch.__version__
)

print(
    "\nTimm version:"
)

print(
    timm.__version__
)

print(
    "\nTest CSV:"
)

print(
    TEST_CSV
)

print(
    "\nModel:"
)

print(
    MODEL_PATH
)

print(
    "\nSwin model:"
)

print(
    MODEL_NAME
)

print(
    "\nImage size:"
)

print(
    f"{IMAGE_SIZE} x {IMAGE_SIZE}"
)

print(
    "\nBatch size:"
)

print(
    BATCH_SIZE
)

print(
    "\nThreshold:"
)

print(
    THRESHOLD
)

print(
    "\nEvaluation preprocessing:"
)

print(
    "CLAHE images"
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    "\nDevice:"
)

print(
    DEVICE
)

if DEVICE.type == "cuda":

    print(
        "CUDA available: YES"
    )

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

else:

    print(
        "CUDA available: NO"
    )

    print(
        "Evaluation will run on CPU."
    )


# ============================================================
# FILE CHECK
# ============================================================

if not TEST_CSV.exists():

    raise FileNotFoundError(
        f"\nPH2 test CSV not found:\n"
        f"{TEST_CSV}"
    )


if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"\nSwin model not found:\n"
        f"{MODEL_PATH}"
    )


# ============================================================
# LOAD CSV
# ============================================================

test_df = pd.read_csv(
    TEST_CSV
)


print(
    "\n" + "=" * 90
)

print(
    "PH2 TEST DATASET INFORMATION"
)

print(
    "=" * 90
)

print(
    "\nNumber of test images:"
)

print(
    len(test_df)
)

print(
    "\nColumns:"
)

print(
    list(test_df.columns)
)


# ============================================================
# FIND LABEL COLUMN
# ============================================================

required_label_candidates = [
    "binary_label",
    "label",
    "target"
]


def find_column(
    dataframe,
    candidates,
    description
):

    for column in candidates:

        if column in dataframe.columns:

            return column

    raise ValueError(
        f"\nCould not find {description} column.\n"
        f"Expected one of: {candidates}\n"
        f"Available columns: {list(dataframe.columns)}"
    )


LABEL_COLUMN = find_column(
    test_df,
    required_label_candidates,
    "binary label"
)


print(
    "\nLabel column:"
)

print(
    LABEL_COLUMN
)


# ============================================================
# USE CLAHE IMAGE COLUMN
# ============================================================

# The trained model is:
#
# melanoma_swin_final_clahe.pth
#
# Therefore PH2 evaluation must use the CLAHE images
# when clahe_image_path is available.

if "clahe_image_path" in test_df.columns:

    IMAGE_COLUMN = "clahe_image_path"

    print(
        "\nImage column:"
    )

    print(
        IMAGE_COLUMN
    )

    print(
        "Using CLAHE images for evaluation."
    )

elif "image_path" in test_df.columns:

    IMAGE_COLUMN = "image_path"

    print(
        "\nWARNING:"
    )

    print(
        "clahe_image_path was not found."
    )

    print(
        "Falling back to image_path."
    )

else:

    raise ValueError(
        "\nNo valid image column found."
    )


# ============================================================
# IMAGE PATH CONVERSION
# ============================================================

def make_absolute_path(path):

    path = Path(
        str(path)
    )

    if path.is_absolute():

        return str(path)

    return str(
        BASE_DIR / path
    )


test_df[
    "resolved_image_path"
] = (
    test_df[
        IMAGE_COLUMN
    ]
    .apply(
        make_absolute_path
    )
)


# ============================================================
# LABEL VALIDATION
# ============================================================

test_df[
    LABEL_COLUMN
] = (
    pd.to_numeric(
        test_df[
            LABEL_COLUMN
        ],
        errors="raise"
    )
    .astype(int)
)


valid_labels = {
    0,
    1
}


actual_labels = set(
    test_df[
        LABEL_COLUMN
    ].unique()
)


if not actual_labels.issubset(
    valid_labels
):

    raise ValueError(
        f"Invalid PH2 labels detected: "
        f"{actual_labels}"
    )


# ============================================================
# CHECK MISSING IMAGES
# ============================================================

missing_images = test_df[
    ~test_df[
        "resolved_image_path"
    ]
    .apply(
        lambda x: Path(x).exists()
    )
]


print(
    "\nMissing PH2 images:"
)

print(
    len(missing_images)
)


if len(missing_images) > 0:

    print(
        "\nFirst missing images:"
    )

    print(
        missing_images[
            [
                IMAGE_COLUMN,
                "resolved_image_path"
            ]
        ]
        .head(10)
        .to_string(
            index=False
        )
    )

    raise FileNotFoundError(
        "\nPH2 test images are missing."
    )


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print(
    "\n" + "=" * 90
)

print(
    "PH2 TEST CLASS DISTRIBUTION"
)

print(
    "=" * 90
)


print(
    test_df[
        LABEL_COLUMN
    ]
    .value_counts()
    .sort_index()
)


label_counts = (
    test_df[
        LABEL_COLUMN
    ]
    .value_counts()
    .sort_index()
)


print(
    "\nNon-melanoma (0):",
    int(
        label_counts.get(
            0,
            0
        )
    )
)


print(
    "Melanoma (1):",
    int(
        label_counts.get(
            1,
            0
        )
    )
)


# ============================================================
# DATASET
# ============================================================

class PH2Dataset(
    Dataset
):

    def __init__(
        self,
        dataframe,
        image_column,
        label_column
    ):

        self.dataframe = (
            dataframe
            .reset_index(
                drop=True
            )
        )

        self.image_column = (
            image_column
        )

        self.label_column = (
            label_column
        )

    def __len__(self):

        return len(
            self.dataframe
        )

    def __getitem__(
        self,
        index
    ):

        row = (
            self.dataframe
            .iloc[index]
        )

        image_path = (
            row[
                "resolved_image_path"
            ]
        )

        label = int(
            row[
                self.label_column
            ]
        )


        # ----------------------------------------------------
        # Load image
        # ----------------------------------------------------

        image = Image.open(
            image_path
        ).convert(
            "RGB"
        )


        # ----------------------------------------------------
        # Resize
        # ----------------------------------------------------

        image = image.resize(
            (
                IMAGE_SIZE,
                IMAGE_SIZE
            ),
            Image.Resampling.BILINEAR
        )


        # ----------------------------------------------------
        # Convert to numpy
        # ----------------------------------------------------

        image = np.asarray(
            image,
            dtype=np.float32
        )


        # ----------------------------------------------------
        # Scale 0-255 -> 0-1
        # ----------------------------------------------------

        image = (
            image / 255.0
        )


        # ----------------------------------------------------
        # ImageNet normalization
        # ----------------------------------------------------

        image = (
            image - MEAN
        ) / STD


        # ----------------------------------------------------
        # HWC -> CHW
        # ----------------------------------------------------

        image = np.transpose(
            image,
            (
                2,
                0,
                1
            )
        )


        image = torch.tensor(
            image,
            dtype=torch.float32
        )


        label = torch.tensor(
            label,
            dtype=torch.float32
        )


        return (
            image,
            label,
            image_path
        )


# ============================================================
# CREATE DATASET
# ============================================================

test_dataset = PH2Dataset(
    test_df,
    IMAGE_COLUMN,
    LABEL_COLUMN
)


# ============================================================
# CREATE DATALOADER
# ============================================================

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=False
)


print(
    "\n" + "=" * 90
)

print(
    "DATALOADER"
)

print(
    "=" * 90
)

print(
    "\nTest images:"
)

print(
    len(test_dataset)
)

print(
    "\nTest batches:"
)

print(
    len(test_loader)
)


# ============================================================
# SWIN MODEL CLASS
# ============================================================

class SwinMelanomaClassifier(
    nn.Module
):

    def __init__(self):

        super().__init__()


        # ----------------------------------------------------
        # EXACT TRAINING BACKBONE
        # ----------------------------------------------------

        self.backbone = timm.create_model(
            MODEL_NAME,
            pretrained=False,
            num_classes=0,
            global_pool="avg"
        )


        # ----------------------------------------------------
        # EXACT TRAINING CLASSIFIER
        #
        # LayerNorm
        # Dropout
        # Linear
        # ----------------------------------------------------

        self.classifier = nn.Sequential(

            nn.LayerNorm(768),

            nn.Dropout(
                0.35
            ),

            nn.Linear(
                768,
                1
            )
        )


    def forward(
        self,
        x
    ):

        features = self.backbone(
            x
        )

        output = self.classifier(
            features
        )

        return output


# ============================================================
# CREATE MODEL
# ============================================================

print(
    "\n" + "=" * 90
)

print(
    "CREATING SWIN MODEL"
)

print(
    "=" * 90
)


model = SwinMelanomaClassifier()


# ============================================================
# LOAD CHECKPOINT
# ============================================================

print(
    "\nLoading checkpoint..."
)


checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)


# ============================================================
# DETECT CHECKPOINT FORMAT
# ============================================================

if isinstance(
    checkpoint,
    dict
):

    if (
        "model_state_dict"
        in checkpoint
    ):

        state_dict = (
            checkpoint[
                "model_state_dict"
            ]
        )

        print(
            "Checkpoint format: "
            "model_state_dict"
        )

    elif (
        "state_dict"
        in checkpoint
    ):

        state_dict = (
            checkpoint[
                "state_dict"
            ]
        )

        print(
            "Checkpoint format: "
            "state_dict"
        )

    else:

        state_dict = checkpoint

        print(
            "Checkpoint format: "
            "direct state_dict"
        )

else:

    state_dict = checkpoint

    print(
        "Checkpoint format: "
        "direct state_dict"
    )


# ============================================================
# CLEAN CHECKPOINT KEYS
# ============================================================

clean_state_dict = {}


for key, value in state_dict.items():

    # --------------------------------------------------------
    # Remove DataParallel prefix
    # --------------------------------------------------------

    if key.startswith(
        "module."
    ):

        key = key[
            len("module.") :
        ]


    # --------------------------------------------------------
    # IMPORTANT:
    #
    # DO NOT remove "backbone."
    #
    # The model architecture itself contains:
    #
    # self.backbone
    #
    # Therefore checkpoint keys:
    #
    # backbone.patch_embed...
    #
    # must remain unchanged.
    # --------------------------------------------------------

    clean_state_dict[
        key
    ] = value


print(
    "\n" + "=" * 90
)

print(
    "CHECKPOINT KEY INFORMATION"
)

print(
    "=" * 90
)

print(
    "\nOriginal keys:",
    len(state_dict)
)

print(
    "Cleaned keys:",
    len(clean_state_dict)
)


# ============================================================
# MODEL KEY INFORMATION
# ============================================================

model_state = (
    model.state_dict()
)


model_keys = set(
    model_state.keys()
)


checkpoint_keys = set(
    clean_state_dict.keys()
)


matched_keys = (
    model_keys
    &
    checkpoint_keys
)


missing_keys_before = (
    model_keys
    -
    checkpoint_keys
)


unexpected_keys_before = (
    checkpoint_keys
    -
    model_keys
)


print(
    "\n" + "=" * 90
)

print(
    "CHECKPOINT COMPATIBILITY"
)

print(
    "=" * 90
)


print(
    "\nModel keys:",
    len(model_keys)
)

print(
    "Checkpoint keys:",
    len(checkpoint_keys)
)

print(
    "Matched keys:",
    len(matched_keys)
)

print(
    "Missing keys:",
    len(missing_keys_before)
)

print(
    "Unexpected keys:",
    len(unexpected_keys_before)
)


# ============================================================
# SHOW KEY MISMATCHES BEFORE LOADING
# ============================================================

if len(missing_keys_before) > 0:

    print(
        "\nMissing keys before loading:"
    )

    for key in sorted(
        missing_keys_before
    )[:30]:

        print(
            "  ",
            key
        )


if len(unexpected_keys_before) > 0:

    print(
        "\nUnexpected keys before loading:"
    )

    for key in sorted(
        unexpected_keys_before
    )[:30]:

        print(
            "  ",
            key
        )


# ============================================================
# LOAD WEIGHTS
# ============================================================

missing_keys, unexpected_keys = (
    model.load_state_dict(
        clean_state_dict,
        strict=False
    )
)


# ============================================================
# WEIGHT LOADING REPORT
# ============================================================

print(
    "\n" + "=" * 90
)

print(
    "WEIGHT LOADING REPORT"
)

print(
    "=" * 90
)


if len(
    missing_keys
) > 0:

    print(
        "\nMissing keys:"
    )

    for key in missing_keys:

        print(
            "  ",
            key
        )

else:

    print(
        "\nMissing keys: 0"
    )


if len(
    unexpected_keys
) > 0:

    print(
        "\nUnexpected keys:"
    )

    for key in unexpected_keys:

        print(
            "  ",
            key
        )

else:

    print(
        "Unexpected keys: 0"
    )


# ============================================================
# SAFETY CHECK
# ============================================================

if (
    len(missing_keys) > 0
    or
    len(unexpected_keys) > 0
):

    raise RuntimeError(
        "\nERROR: Checkpoint was not loaded "
        "completely.\n"
        "Evaluation stopped to prevent "
        "invalid PH2 results."
    )


print(
    "\nAll checkpoint weights loaded successfully."
)


# ============================================================
# VERIFY SWIN BACKBONE
# ============================================================

backbone_model_keys = []


for key in model_state.keys():

    if key.startswith(
        "backbone."
    ):

        backbone_model_keys.append(
            key
        )


loaded_backbone_keys = [
    key
    for key in backbone_model_keys
    if key not in missing_keys
]


print(
    "\n" + "=" * 90
)

print(
    "SWIN BACKBONE WEIGHT VERIFICATION"
)

print(
    "=" * 90
)


print(
    "\nSwin backbone keys:",
    len(
        backbone_model_keys
    )
)


print(
    "Loaded backbone keys:",
    len(
        loaded_backbone_keys
    )
)


missing_backbone_count = (
    len(
        backbone_model_keys
    )
    -
    len(
        loaded_backbone_keys
    )
)


print(
    "Missing backbone keys:",
    missing_backbone_count
)


if len(
    backbone_model_keys
) > 0:

    backbone_loading_ratio = (
        len(
            loaded_backbone_keys
        )
        /
        len(
            backbone_model_keys
        )
    )

else:

    backbone_loading_ratio = 0.0


print(
    "\nBackbone loading ratio:",
    f"{backbone_loading_ratio:.4f}"
)


if (
    backbone_loading_ratio
    < 1.0
):

    raise RuntimeError(
        "\nERROR: Swin backbone was not "
        "loaded completely."
    )


print(
    "\nSwin backbone weights "
    "loaded successfully."
)


# ============================================================
# VERIFY CLASSIFIER HEAD
# ============================================================

classifier_model_keys = [
    key
    for key in model_state.keys()
    if key.startswith(
        "classifier."
    )
]


classifier_loaded_keys = [
    key
    for key in classifier_model_keys
    if key not in missing_keys
]


print(
    "\n" + "=" * 90
)

print(
    "CLASSIFIER HEAD VERIFICATION"
)

print(
    "=" * 90
)


print(
    "\nClassifier keys:",
    len(
        classifier_model_keys
    )
)


print(
    "Loaded classifier keys:",
    len(
        classifier_loaded_keys
    )
)


print(
    "Missing classifier keys:",
    len(
        classifier_model_keys
    )
    -
    len(
        classifier_loaded_keys
    )
)


if (
    len(classifier_model_keys)
    !=
    len(classifier_loaded_keys)
):

    raise RuntimeError(
        "\nERROR: Trained classifier head "
        "was not loaded completely."
    )


print(
    "\nTrained classifier head "
    "loaded successfully."
)


# ============================================================
# MOVE MODEL TO DEVICE
# ============================================================

model = model.to(
    DEVICE
)


model.eval()


# ============================================================
# MODEL INFORMATION
# ============================================================

print(
    "\n" + "=" * 90
)

print(
    "FINAL MODEL INFORMATION"
)

print(
    "=" * 90
)


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
    total_parameters
)


print(
    "Trainable parameters:",
    trainable_parameters
)


print(
    "Model device:",
    DEVICE
)


# ============================================================
# LOSS
# ============================================================

criterion = (
    nn.BCEWithLogitsLoss()
)


# ============================================================
# EVALUATION
# ============================================================

print(
    "\n" + "=" * 90
)

print(
    "STARTING PH2 INDEPENDENT TEST EVALUATION"
)

print(
    "=" * 90
)


all_labels = []

all_probabilities = []

all_paths = []


start_time = time.time()


with torch.no_grad():

    for batch_index, (
        images,
        labels,
        paths
    ) in enumerate(
        test_loader
    ):

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )


        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        logits = model(
            images
        ).squeeze(1)


        # ----------------------------------------------------
        # Convert logits to probability
        # ----------------------------------------------------

        probabilities = torch.sigmoid(
            logits
        )


        # ----------------------------------------------------
        # Store results
        # ----------------------------------------------------

        all_labels.extend(
            labels.cpu()
            .numpy()
            .tolist()
        )


        all_probabilities.extend(
            probabilities.cpu()
            .numpy()
            .tolist()
        )


        all_paths.extend(
            list(paths)
        )


        if (
            (batch_index + 1) % 5 == 0
            or
            (batch_index + 1)
            == len(test_loader)
        ):

            print(
                f"  Batch "
                f"{batch_index + 1}/"
                f"{len(test_loader)}"
            )


evaluation_time = (
    time.time()
    -
    start_time
)


# ============================================================
# NUMPY ARRAYS
# ============================================================

y_true = np.asarray(
    all_labels,
    dtype=int
)


y_probability = np.asarray(
    all_probabilities,
    dtype=float
)


y_pred = (
    y_probability
    >= THRESHOLD
).astype(int)


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


recall = recall_score(
    y_true,
    y_pred,
    zero_division=0
)


f1 = f1_score(
    y_true,
    y_pred,
    zero_division=0
)


# ============================================================
# ROC-AUC
# ============================================================

try:

    roc_auc = roc_auc_score(
        y_true,
        y_probability
    )

except ValueError:

    roc_auc = float(
        "nan"
    )


# ============================================================
# PR-AUC
# ============================================================

try:

    pr_auc = (
        average_precision_score(
            y_true,
            y_probability
        )
    )

except ValueError:

    pr_auc = float(
        "nan"
    )


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=[
        0,
        1
    ]
)


tn, fp, fn, tp = (
    cm.ravel()
)


# ============================================================
# SPECIFICITY
# ============================================================

if (
    tn + fp
) > 0:

    specificity = (
        tn
        /
        (
            tn + fp
        )
    )

else:

    specificity = float(
        "nan"
    )


# ============================================================
# RESULTS
# ============================================================

print(
    "\n" + "=" * 90
)

print(
    "PH2 INDEPENDENT TEST RESULTS"
)

print(
    "=" * 90
)


print(
    "\nTest samples:",
    len(y_true)
)


print(
    "\nAccuracy:",
    f"{accuracy:.4f}"
)


print(
    "ROC-AUC:",
    f"{roc_auc:.4f}"
)


print(
    "PR-AUC:",
    f"{pr_auc:.4f}"
)


print(
    "Precision:",
    f"{precision:.4f}"
)


print(
    "Recall / Sensitivity:",
    f"{recall:.4f}"
)


print(
    "Specificity:",
    f"{specificity:.4f}"
)


print(
    "F1-score:",
    f"{f1:.4f}"
)


print(
    "\nTrue Negative:",
    tn
)


print(
    "False Positive:",
    fp
)


print(
    "False Negative:",
    fn
)


print(
    "True Positive:",
    tp
)


print(
    "\nEvaluation time:",
    f"{evaluation_time:.2f} seconds"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

classification_report_text = (
    classification_report(
        y_true,
        y_pred,
        labels=[
            0,
            1
        ],
        target_names=[
            "Non-melanoma",
            "Melanoma"
        ],
        digits=4,
        zero_division=0
    )
)


print(
    "\n" + "=" * 90
)

print(
    "CLASSIFICATION REPORT"
)

print(
    "=" * 90
)


print(
    classification_report_text
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
        "SWIN TRANSFORMER - PH2 "
        "INDEPENDENT TEST "
        "CLASSIFICATION REPORT\n"
    )

    file.write(
        "=" * 80
        +
        "\n\n"
    )

    file.write(
        classification_report_text
    )

    file.write(
        "\n\nAdditional Metrics\n"
    )

    file.write(
        "=" * 80
        +
        "\n"
    )

    file.write(
        f"Model: "
        f"{MODEL_NAME}\n"
    )

    file.write(
        f"Preprocessing: "
        f"CLAHE\n"
    )

    file.write(
        f"Image Size: "
        f"{IMAGE_SIZE}x{IMAGE_SIZE}\n"
    )

    file.write(
        f"Accuracy: "
        f"{accuracy:.6f}\n"
    )

    file.write(
        f"ROC-AUC: "
        f"{roc_auc:.6f}\n"
    )

    file.write(
        f"PR-AUC: "
        f"{pr_auc:.6f}\n"
    )

    file.write(
        f"Precision: "
        f"{precision:.6f}\n"
    )

    file.write(
        f"Recall: "
        f"{recall:.6f}\n"
    )

    file.write(
        f"Specificity: "
        f"{specificity:.6f}\n"
    )

    file.write(
        f"F1-score: "
        f"{f1:.6f}\n"
    )

    file.write(
        f"True Negative: "
        f"{tn}\n"
    )

    file.write(
        f"False Positive: "
        f"{fp}\n"
    )

    file.write(
        f"False Negative: "
        f"{fn}\n"
    )

    file.write(
        f"True Positive: "
        f"{tp}\n"
    )

    file.write(
        f"Threshold: "
        f"{THRESHOLD:.4f}\n"
    )


# ============================================================
# SAVE METRICS CSV
# ============================================================

metrics_df = pd.DataFrame(
    {
        "model": [
            "Swin-Tiny"
        ],

        "dataset": [
            "PH2 Independent Test"
        ],

        "preprocessing": [
            "CLAHE"
        ],

        "test_samples": [
            len(y_true)
        ],

        "threshold": [
            THRESHOLD
        ],

        "accuracy": [
            accuracy
        ],

        "roc_auc": [
            roc_auc
        ],

        "pr_auc": [
            pr_auc
        ],

        "precision": [
            precision
        ],

        "recall": [
            recall
        ],

        "specificity": [
            specificity
        ],

        "f1": [
            f1
        ],

        "true_negative": [
            tn
        ],

        "false_positive": [
            fp
        ],

        "false_negative": [
            fn
        ],

        "true_positive": [
            tp
        ],

        "evaluation_time_seconds": [
            evaluation_time
        ]
    }
)


metrics_df.to_csv(
    METRICS_PATH,
    index=False
)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

prediction_df = (
    test_df.copy()
)


prediction_df[
    "true_label"
] = y_true


prediction_df[
    "melanoma_probability"
] = y_probability


prediction_df[
    "predicted_label"
] = y_pred


prediction_df[
    "prediction"
] = np.where(
    y_pred == 1,
    "Melanoma",
    "Non-melanoma"
)


prediction_df[
    "correct"
] = (
    y_true == y_pred
)


prediction_df.to_csv(
    PREDICTIONS_PATH,
    index=False
)


# ============================================================
# ROC CURVE
# ============================================================

if (
    len(
        np.unique(y_true)
    )
    == 2
):

    fpr, tpr, _ = (
        roc_curve(
            y_true,
            y_probability
        )
    )


    plt.figure(
        figsize=(7, 6)
    )


    plt.plot(
        fpr,
        tpr,
        label=(
            f"Swin-Tiny "
            f"(AUC = {roc_auc:.4f})"
        )
    )


    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )


    plt.xlabel(
        "False Positive Rate"
    )


    plt.ylabel(
        "True Positive Rate"
    )


    plt.title(
        "Swin-Tiny ROC Curve - "
        "PH2 Independent Test"
    )


    plt.legend(
        loc="lower right"
    )


    plt.grid(
        alpha=0.3
    )


    plt.tight_layout()


    plt.savefig(
        ROC_CURVE_PATH,
        dpi=300
    )


    plt.close()


# ============================================================
# PRECISION-RECALL CURVE
# ============================================================

if (
    len(
        np.unique(y_true)
    )
    == 2
):

    precision_curve, recall_curve, _ = (
        precision_recall_curve(
            y_true,
            y_probability
        )
    )


    plt.figure(
        figsize=(7, 6)
    )


    plt.plot(
        recall_curve,
        precision_curve,
        label=(
            f"PR-AUC = "
            f"{pr_auc:.4f}"
        )
    )


    plt.xlabel(
        "Recall"
    )


    plt.ylabel(
        "Precision"
    )


    plt.title(
        "Swin-Tiny Precision-Recall "
        "Curve - PH2"
    )


    plt.legend()


    plt.grid(
        alpha=0.3
    )


    plt.tight_layout()


    plt.savefig(
        PR_CURVE_PATH,
        dpi=300
    )


    plt.close()


# ============================================================
# CONFUSION MATRIX
# ============================================================

plt.figure(
    figsize=(7, 6)
)


plt.imshow(
    cm,
    interpolation="nearest"
)


plt.title(
    "Swin-Tiny Confusion Matrix - PH2"
)


plt.colorbar()


tick_marks = np.arange(
    2
)


plt.xticks(
    tick_marks,
    [
        "Non-melanoma",
        "Melanoma"
    ],
    rotation=20
)


plt.yticks(
    tick_marks,
    [
        "Non-melanoma",
        "Melanoma"
    ]
)


for i in range(2):

    for j in range(2):

        plt.text(
            j,
            i,
            str(
                cm[i, j]
            ),
            ha="center",
            va="center",
            fontsize=14
        )


plt.ylabel(
    "True Label"
)


plt.xlabel(
    "Predicted Label"
)


plt.tight_layout()


plt.savefig(
    CONFUSION_MATRIX_PATH,
    dpi=300
)


plt.close()


# ============================================================
# FINAL OUTPUT
# ============================================================

print(
    "\n" + "=" * 90
)

print(
    "PH2 EVALUATION COMPLETED"
)

print(
    "=" * 90
)


print(
    "\nMetrics saved:"
)

print(
    METRICS_PATH
)


print(
    "\nPredictions saved:"
)

print(
    PREDICTIONS_PATH
)


print(
    "\nClassification report saved:"
)

print(
    REPORT_PATH
)


print(
    "\nROC curve saved:"
)

print(
    ROC_CURVE_PATH
)


print(
    "\nPR curve saved:"
)

print(
    PR_CURVE_PATH
)


print(
    "\nConfusion matrix saved:"
)

print(
    CONFUSION_MATRIX_PATH
)


print(
    "\n" + "=" * 90
)

print(
    "STATUS: PASS"
)

print(
    "Swin-Tiny PH2 independent "
    "evaluation completed successfully."
)

print(
    "=" * 90
)
