from pathlib import Path
import random
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
)


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "final_binary_clahe_splits"
)

TRAIN_CSV = DATA_DIR / "train_final_clahe.csv"
VAL_CSV = DATA_DIR / "validation_final_clahe.csv"

MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = (
    MODEL_DIR
    / "melanoma_swin_final_clahe.pth"
)

HISTORY_PATH = (
    RESULTS_DIR
    / "swin_final_clahe_training_history.csv"
)

IMAGE_SIZE = 224

BATCH_SIZE = 8

EPOCHS = 10

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-4

RANDOM_SEED = 42

MODEL_NAME = "swin_tiny_patch4_window7_224"

THRESHOLD = 0.5

# CPU-friendly:
NUM_WORKERS = 0

PIN_MEMORY = False


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 90)
print("MELONMA - SWIN TRANSFORMER FINAL CLAHE TRAINING")
print("=" * 90)

print("\nPyTorch version:", torch.__version__)
print("TorchVision version:", __import__("torchvision").__version__)
print("timm version:", timm.__version__)

print("\nTrain CSV:")
print(TRAIN_CSV)

print("\nValidation CSV:")
print(VAL_CSV)

print("\nModel output:")
print(MODEL_PATH)

print("\nSwin model:")
print(MODEL_NAME)

print("\nImage size:")
print(f"{IMAGE_SIZE} x {IMAGE_SIZE}")

print("\nBatch size:")
print(BATCH_SIZE)

print("\nEpochs:")
print(EPOCHS)

print("\nLearning rate:")
print(LEARNING_RATE)

print("\nCUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("Training device: CUDA")
else:
    print("Training will run on CPU.")


# ============================================================
# CHECK FILES
# ============================================================

if not TRAIN_CSV.exists():
    raise FileNotFoundError(
        f"Training CSV not found:\n{TRAIN_CSV}"
    )

if not VAL_CSV.exists():
    raise FileNotFoundError(
        f"Validation CSV not found:\n{VAL_CSV}"
    )


# ============================================================
# LOAD DATA
# ============================================================

train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VAL_CSV)


# ============================================================
# DATASET INFORMATION
# ============================================================

print("\n" + "=" * 90)
print("DATASET INFORMATION")
print("=" * 90)

print(
    "\nTraining images   :",
    len(train_df)
)

print(
    "Validation images :",
    len(val_df)
)


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = [
    "image_path",
    "binary_label",
    "binary_diagnosis",
    "image_id",
]

for column in required_columns:

    if column not in train_df.columns:
        raise ValueError(
            f"Missing training column: {column}"
        )

    if column not in val_df.columns:
        raise ValueError(
            f"Missing validation column: {column}"
        )


# ============================================================
# LABEL VALIDATION
# ============================================================

valid_labels = {0, 1}

train_labels = set(
    train_df["binary_label"].unique()
)

val_labels = set(
    val_df["binary_label"].unique()
)

if not train_labels.issubset(valid_labels):
    raise ValueError(
        f"Invalid training labels: {train_labels}"
    )

if not val_labels.issubset(valid_labels):
    raise ValueError(
        f"Invalid validation labels: {val_labels}"
    )


# ============================================================
# ABSOLUTE IMAGE PATHS
# ============================================================

def make_absolute_path(path):

    path = Path(str(path))

    if path.is_absolute():
        return str(path)

    return str(BASE_DIR / path)


train_df["image_path"] = (
    train_df["image_path"]
    .apply(make_absolute_path)
)

val_df["image_path"] = (
    val_df["image_path"]
    .apply(make_absolute_path)
)


# ============================================================
# VERIFY IMAGES
# ============================================================

missing_train = train_df[
    ~train_df["image_path"]
    .apply(lambda x: Path(x).exists())
]

missing_val = val_df[
    ~val_df["image_path"]
    .apply(lambda x: Path(x).exists())
]


print(
    "\nMissing training images   :",
    len(missing_train)
)

print(
    "Missing validation images :",
    len(missing_val)
)


if len(missing_train) > 0:

    print(
        "\nFirst missing training images:"
    )

    print(
        missing_train["image_path"]
        .head(10)
        .to_string(index=False)
    )

    raise FileNotFoundError(
        "Training images are missing."
    )


if len(missing_val) > 0:

    print(
        "\nFirst missing validation images:"
    )

    print(
        missing_val["image_path"]
        .head(10)
        .to_string(index=False)
    )

    raise FileNotFoundError(
        "Validation images are missing."
    )


# ============================================================
# CHECK TRAIN / VALIDATION LEAKAGE
# ============================================================

train_ids = set(
    train_df["image_id"]
)

val_ids = set(
    val_df["image_id"]
)

overlap = train_ids & val_ids


print(
    "\nTrain/Validation image overlap:",
    len(overlap)
)


if len(overlap) > 0:

    raise ValueError(
        "Data leakage detected between train and validation."
    )


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\n" + "=" * 90)
print("CLASS DISTRIBUTION")
print("=" * 90)

print("\n=== TRAIN ===")

print(
    train_df["binary_diagnosis"]
    .value_counts()
)

print("\n=== VALIDATION ===")

print(
    val_df["binary_diagnosis"]
    .value_counts()
)


# ============================================================
# CLASS WEIGHT
# ============================================================

negative_count = int(
    (train_df["binary_label"] == 0).sum()
)

positive_count = int(
    (train_df["binary_label"] == 1).sum()
)

positive_weight = (
    negative_count / positive_count
)

print("\n" + "=" * 90)
print("CLASS WEIGHT")
print("=" * 90)

print(
    "Non-melanoma (0):",
    "1.0000"
)

print(
    "Melanoma (1):",
    f"{positive_weight:.4f}"
)


# ============================================================
# TRANSFORMS
# ============================================================

# ImageNet normalization because the Swin model
# uses ImageNet pretrained weights.

IMAGENET_MEAN = [
    0.485,
    0.456,
    0.406,
]

IMAGENET_STD = [
    0.229,
    0.224,
    0.225,
]


# Training augmentation
train_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomVerticalFlip(
        p=0.5
    ),

    transforms.RandomRotation(
        degrees=10
    ),

    transforms.RandomAffine(
        degrees=0,
        translate=(0.03, 0.03),
        scale=(0.90, 1.10)
    ),

    transforms.ColorJitter(
        brightness=0.10,
        contrast=0.10,
        saturation=0.05,
        hue=0.02
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    ),

])


# Validation transform
val_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    ),

])


# ============================================================
# DATASET CLASS
# ============================================================

class MelanomaDataset(Dataset):

    def __init__(
        self,
        dataframe,
        transform=None
    ):

        self.dataframe = (
            dataframe
            .reset_index(drop=True)
            .copy()
        )

        self.transform = transform


    def __len__(self):

        return len(self.dataframe)


    def __getitem__(self, index):

        row = self.dataframe.iloc[index]

        image_path = str(
            row["image_path"]
        )

        label = float(
            row["binary_label"]
        )

        try:

            image = Image.open(
                image_path
            ).convert("RGB")

        except Exception as error:

            raise RuntimeError(
                f"Could not load image:\n"
                f"{image_path}\n"
                f"Error: {error}"
            )


        if self.transform is not None:

            image = self.transform(
                image
            )


        label = torch.tensor(
            label,
            dtype=torch.float32
        )


        return image, label


# ============================================================
# CREATE DATASETS
# ============================================================

train_dataset = MelanomaDataset(
    train_df,
    transform=train_transform
)

val_dataset = MelanomaDataset(
    val_df,
    transform=val_transform
)


# ============================================================
# CREATE DATALOADERS
# ============================================================

train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True,

    num_workers=NUM_WORKERS,

    pin_memory=PIN_MEMORY,

)

val_loader = DataLoader(

    val_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=NUM_WORKERS,

    pin_memory=PIN_MEMORY,

)


print("\n" + "=" * 90)
print("DATALOADERS")
print("=" * 90)

print(
    "\nTraining batches:",
    len(train_loader)
)

print(
    "Validation batches:",
    len(val_loader)
)


# ============================================================
# LOAD PRETRAINED SWIN
# ============================================================

print("\n" + "=" * 90)
print("LOADING PRETRAINED SWIN TRANSFORMER")
print("=" * 90)

print("\nModel:")
print(MODEL_NAME)

print("\nImageNet pretrained weights:")
print("Enabled")


# ============================================================
# IMPORTANT CPU-FRIENDLY DESIGN
# ============================================================
#
# Instead of training all 27.5M Swin parameters,
# we use the pretrained Swin as a frozen feature extractor.
#
# The classifier head is the ONLY trainable component.
#
# This avoids expensive backward propagation through
# the entire Swin Transformer.
#

swin_backbone = timm.create_model(

    MODEL_NAME,

    pretrained=True,

    num_classes=0,

    global_pool="avg",

)


# ============================================================
# FREEZE SWIN BACKBONE
# ============================================================

for parameter in swin_backbone.parameters():

    parameter.requires_grad = False


# Swin output feature dimension
feature_dim = (
    swin_backbone.num_features
)


# ============================================================
# CLASSIFICATION MODEL
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

        # Frozen backbone:
        # no gradient calculation here.

        with torch.no_grad():

            features = self.backbone(
                x
            )

        logits = self.classifier(
            features
        )

        return logits.squeeze(1)


model = SwinMelanomaClassifier(
    swin_backbone,
    feature_dim
)


model = model.to(DEVICE)


# ============================================================
# PARAMETER INFORMATION
# ============================================================

total_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)

trainable_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)

print("\n" + "=" * 90)
print("SWIN MODEL INFORMATION")
print("=" * 90)

print(
    "\nTotal parameters:",
    f"{total_parameters:,}"
)

print(
    "Trainable parameters:",
    f"{trainable_parameters:,}"
)

print(
    "Frozen parameters:",
    f"{total_parameters - trainable_parameters:,}"
)

print(
    "\nFeature dimension:",
    feature_dim
)

print(
    "\nModel device:",
    DEVICE
)


# ============================================================
# LOSS
# ============================================================

pos_weight_tensor = torch.tensor(
    [positive_weight],
    dtype=torch.float32,
    device=DEVICE
)

criterion = nn.BCEWithLogitsLoss(
    pos_weight=pos_weight_tensor
)


print("\n" + "=" * 90)
print("LOSS CONFIGURATION")
print("=" * 90)

print(
    "\nBCEWithLogitsLoss"
)

print(
    "Positive class weight:",
    f"{positive_weight:.4f}"
)


# ============================================================
# OPTIMIZER
# ============================================================

trainable_parameters_list = [

    parameter
    for parameter in model.parameters()
    if parameter.requires_grad

]


optimizer = torch.optim.AdamW(

    trainable_parameters_list,

    lr=LEARNING_RATE,

    weight_decay=WEIGHT_DECAY

)


# ============================================================
# LEARNING RATE SCHEDULER
# ============================================================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(

    optimizer,

    mode="max",

    factor=0.5,

    patience=2,

    min_lr=1e-6

)


# ============================================================
# METRIC FUNCTION
# ============================================================

def calculate_metrics(
    y_true,
    y_prob
):

    y_true = np.asarray(
        y_true
    )

    y_prob = np.asarray(
        y_prob
    )

    y_pred = (
        y_prob >= THRESHOLD
    ).astype(int)


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


    try:

        roc_auc = roc_auc_score(
            y_true,
            y_prob
        )

    except ValueError:

        roc_auc = float("nan")


    try:

        pr_auc = average_precision_score(
            y_true,
            y_prob
        )

    except ValueError:

        pr_auc = float("nan")


    return {

        "accuracy": accuracy,

        "roc_auc": roc_auc,

        "pr_auc": pr_auc,

        "precision": precision,

        "recall": recall,

        "f1": f1,

    }


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch():

    # IMPORTANT:
    # Keep the frozen Swin backbone in eval mode.
    # Only the classifier remains in training mode.

    model.backbone.eval()
    model.classifier.train()


    running_loss = 0.0

    all_labels = []
    all_probs = []


    for batch_index, (
        images,
        labels
    ) in enumerate(train_loader):

        images = images.to(
            DEVICE,
            non_blocking=False
        )

        labels = labels.to(
            DEVICE,
            non_blocking=False
        )


        optimizer.zero_grad(
            set_to_none=True
        )


        logits = model(
            images
        )


        loss = criterion(
            logits,
            labels
        )


        loss.backward()


        optimizer.step()


        running_loss += (
            loss.item()
            * images.size(0)
        )


        probabilities = torch.sigmoid(
            logits
        )


        all_labels.extend(
            labels.detach()
            .cpu()
            .numpy()
            .tolist()
        )

        all_probs.extend(
            probabilities.detach()
            .cpu()
            .numpy()
            .tolist()
        )


        # Progress every 100 batches
        if (
            (batch_index + 1) % 100 == 0
            or
            (batch_index + 1) == len(train_loader)
        ):

            print(
                f"  Batch "
                f"{batch_index + 1}/"
                f"{len(train_loader)} "
                f"| Loss: "
                f"{loss.item():.4f}"
            )


    epoch_loss = (
        running_loss
        / len(train_dataset)
    )


    metrics = calculate_metrics(
        all_labels,
        all_probs
    )


    return epoch_loss, metrics


# ============================================================
# VALIDATION
# ============================================================

def validate():

    model.backbone.eval()
    model.classifier.eval()


    running_loss = 0.0

    all_labels = []
    all_probs = []


    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(
                DEVICE
            )

            labels = labels.to(
                DEVICE
            )


            logits = model(
                images
            )


            loss = criterion(
                logits,
                labels
            )


            running_loss += (
                loss.item()
                * images.size(0)
            )


            probabilities = torch.sigmoid(
                logits
            )


            all_labels.extend(
                labels.cpu()
                .numpy()
                .tolist()
            )

            all_probs.extend(
                probabilities.cpu()
                .numpy()
                .tolist()
            )


    epoch_loss = (
        running_loss
        / len(val_dataset)
    )


    metrics = calculate_metrics(
        all_labels,
        all_probs
    )


    return epoch_loss, metrics


# ============================================================
# TRAINING HEADER
# ============================================================

print("\n" + "=" * 90)
print("STARTING SWIN TRANSFORMER TRAINING")
print("=" * 90)

print("\nEpochs:", EPOCHS)

print("\nBatch size:", BATCH_SIZE)

print(
    "\nLearning rate:",
    LEARNING_RATE
)

print(
    "\nWeight decay:",
    WEIGHT_DECAY
)

print(
    "\nPretrained: ImageNet"
)

print(
    "\nInput: 224 x 224"
)

print(
    "\nNormalization: ImageNet mean/std"
)

print(
    "\nAugmentation: Training only"
)

print(
    "\nClass imbalance:",
    "Positive class weighting"
)

print(
    "\nBackbone:",
    "Swin-Tiny"
)

print(
    "\nBackbone frozen:",
    True
)

print(
    "\nTrainable parameters:",
    f"{trainable_parameters:,}"
)

print(
    "\nStarting training...\n"
)


# ============================================================
# TRAINING LOOP
# ============================================================

history = []

best_val_auc = -np.inf

best_epoch = 0

training_start = time.time()


for epoch in range(1, EPOCHS + 1):

    epoch_start = time.time()


    print("\n" + "-" * 90)

    print(
        f"EPOCH {epoch}/{EPOCHS}"
    )

    print("-" * 90)


    train_loss, train_metrics = (
        train_one_epoch()
    )


    val_loss, val_metrics = (
        validate()
    )


    scheduler.step(
        val_metrics["roc_auc"]
    )


    current_lr = (
        optimizer.param_groups[0]["lr"]
    )


    epoch_time = (
        time.time()
        - epoch_start
    )


    print("\nTRAINING RESULTS")

    print(
        f"Train Loss      : "
        f"{train_loss:.4f}"
    )

    print(
        f"Train Accuracy  : "
        f"{train_metrics['accuracy']:.4f}"
    )

    print(
        f"Train ROC-AUC   : "
        f"{train_metrics['roc_auc']:.4f}"
    )

    print(
        f"Train PR-AUC    : "
        f"{train_metrics['pr_auc']:.4f}"
    )

    print(
        f"Train Precision : "
        f"{train_metrics['precision']:.4f}"
    )

    print(
        f"Train Recall    : "
        f"{train_metrics['recall']:.4f}"
    )

    print(
        f"Train F1        : "
        f"{train_metrics['f1']:.4f}"
    )


    print("\nVALIDATION RESULTS")

    print(
        f"Val Loss        : "
        f"{val_loss:.4f}"
    )

    print(
        f"Val Accuracy    : "
        f"{val_metrics['accuracy']:.4f}"
    )

    print(
        f"Val ROC-AUC     : "
        f"{val_metrics['roc_auc']:.4f}"
    )

    print(
        f"Val PR-AUC      : "
        f"{val_metrics['pr_auc']:.4f}"
    )

    print(
        f"Val Precision   : "
        f"{val_metrics['precision']:.4f}"
    )

    print(
        f"Val Recall      : "
        f"{val_metrics['recall']:.4f}"
    )

    print(
        f"Val F1          : "
        f"{val_metrics['f1']:.4f}"
    )

    print(
        f"\nLearning rate   : "
        f"{current_lr:.7f}"
    )

    print(
        f"Epoch time      : "
        f"{epoch_time / 60:.2f} minutes"
    )


    # ========================================================
    # SAVE HISTORY
    # ========================================================

    history_row = {

        "epoch": epoch,

        "train_loss": train_loss,

        "train_accuracy":
            train_metrics["accuracy"],

        "train_roc_auc":
            train_metrics["roc_auc"],

        "train_pr_auc":
            train_metrics["pr_auc"],

        "train_precision":
            train_metrics["precision"],

        "train_recall":
            train_metrics["recall"],

        "train_f1":
            train_metrics["f1"],

        "val_loss": val_loss,

        "val_accuracy":
            val_metrics["accuracy"],

        "val_roc_auc":
            val_metrics["roc_auc"],

        "val_pr_auc":
            val_metrics["pr_auc"],

        "val_precision":
            val_metrics["precision"],

        "val_recall":
            val_metrics["recall"],

        "val_f1":
            val_metrics["f1"],

        "learning_rate":
            current_lr,

        "epoch_time_minutes":
            epoch_time / 60,

    }


    history.append(
        history_row
    )


    history_df = pd.DataFrame(
        history
    )

    history_df.to_csv(
        HISTORY_PATH,
        index=False
    )


    # ========================================================
    # BEST MODEL CHECKPOINT
    # ========================================================

    if (
        val_metrics["roc_auc"]
        > best_val_auc
    ):

        best_val_auc = (
            val_metrics["roc_auc"]
        )

        best_epoch = epoch


        checkpoint = {

            "epoch": epoch,

            "model_name":
                MODEL_NAME,

            "model_state_dict":
                model.state_dict(),

            "classifier_state_dict":
                model.classifier.state_dict(),

            "backbone_frozen":
                True,

            "image_size":
                IMAGE_SIZE,

            "threshold":
                THRESHOLD,

            "positive_weight":
                positive_weight,

            "val_roc_auc":
                best_val_auc,

            "val_accuracy":
                val_metrics["accuracy"],

            "val_pr_auc":
                val_metrics["pr_auc"],

            "val_precision":
                val_metrics["precision"],

            "val_recall":
                val_metrics["recall"],

            "val_f1":
                val_metrics["f1"],

        }


        torch.save(
            checkpoint,
            MODEL_PATH
        )


        print(
            "\n*** BEST SWIN MODEL SAVED ***"
        )

        print(
            "Best validation ROC-AUC:",
            f"{best_val_auc:.4f}"
        )

        print(
            "Saved to:",
            MODEL_PATH
        )


# ============================================================
# TRAINING COMPLETE
# ============================================================

total_training_time = (
    time.time()
    - training_start
)


print("\n" + "=" * 90)
print("SWIN TRANSFORMER TRAINING COMPLETED")
print("=" * 90)


print(
    "\nBest epoch:",
    best_epoch
)

print(
    "\nBest validation ROC-AUC:",
    f"{best_val_auc:.4f}"
)

print(
    "\nTotal training time:",
    f"{total_training_time / 60:.2f} minutes"
)

print(
    "\nBest model saved:"
)

print(
    MODEL_PATH
)

print(
    "\nTraining history saved:"
)

print(
    HISTORY_PATH
)


print("\n" + "=" * 90)
print("FINAL STATUS")
print("=" * 90)

print(
    "\nSTATUS: PASS"
)

print(
    "Swin Transformer final CLAHE "
    "training completed successfully."
)

print("=" * 90)
