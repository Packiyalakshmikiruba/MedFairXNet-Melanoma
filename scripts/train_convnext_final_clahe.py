from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.utils.class_weight import compute_class_weight


# =========================================================
# CONFIGURATION
# =========================================================

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
    / "melanoma_convnext_final_clahe.keras"
)

HISTORY_PATH = (
    RESULTS_DIR
    / "convnext_final_clahe_training_history.csv"
)

IMAGE_SIZE = (224, 224)

BATCH_SIZE = 8

# CPU training is very slow on your system.
# Start with 5 epochs.
EPOCHS = 5

RANDOM_SEED = 42


# =========================================================
# REPRODUCIBILITY
# =========================================================

tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# =========================================================
# HEADER
# =========================================================

print("=" * 80)
print("MELONMA - CONVNEXT FINAL CLAHE TRAINING")
print("=" * 80)

print("\nTensorFlow version:", tf.__version__)

print("\nTrain CSV:")
print(TRAIN_CSV)

print("\nValidation CSV:")
print(VAL_CSV)

print("\nModel output:")
print(MODEL_PATH)

print("\nImage preprocessing:")
print("Resize       : 224 x 224")
print("Pixel range  : 0 - 255")
print("Normalization: NONE")
print("Reason       : ConvNeXt built-in preprocessing")


# =========================================================
# CHECK FILES
# =========================================================

if not TRAIN_CSV.exists():
    raise FileNotFoundError(
        f"Training CSV not found:\n{TRAIN_CSV}"
    )

if not VAL_CSV.exists():
    raise FileNotFoundError(
        f"Validation CSV not found:\n{VAL_CSV}"
    )


# =========================================================
# LOAD METADATA
# =========================================================

train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VAL_CSV)


print("\n" + "=" * 80)
print("DATASET INFORMATION")
print("=" * 80)

print("\nTraining images   :", len(train_df))
print("Validation images :", len(val_df))


# =========================================================
# REQUIRED COLUMNS
# =========================================================

required_columns = [
    "image_path",
    "binary_label",
    "binary_diagnosis",
    "image_id",
]

for column in required_columns:

    if column not in train_df.columns:
        raise ValueError(
            f"Missing column in training CSV: {column}"
        )

    if column not in val_df.columns:
        raise ValueError(
            f"Missing column in validation CSV: {column}"
        )


# =========================================================
# LABEL VALIDATION
# =========================================================

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


# =========================================================
# IMAGE PATH
# =========================================================

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


# =========================================================
# VERIFY TRAINING IMAGES
# =========================================================

missing_train = train_df[
    ~train_df["image_path"]
    .apply(lambda x: Path(x).exists())
]


# =========================================================
# VERIFY VALIDATION IMAGES
# =========================================================

missing_val = val_df[
    ~val_df["image_path"]
    .apply(lambda x: Path(x).exists())
]


print("\nMissing training images   :", len(missing_train))
print("Missing validation images :", len(missing_val))


if len(missing_train) > 0:

    print("\nFirst missing training images:")

    print(
        missing_train["image_path"]
        .head(10)
        .to_string(index=False)
    )

    raise FileNotFoundError(
        "Training images are missing."
    )


if len(missing_val) > 0:

    print("\nFirst missing validation images:")

    print(
        missing_val["image_path"]
        .head(10)
        .to_string(index=False)
    )

    raise FileNotFoundError(
        "Validation images are missing."
    )


# =========================================================
# CHECK SPLIT LEAKAGE
# =========================================================

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


# =========================================================
# CLASS DISTRIBUTION
# =========================================================

print("\n" + "=" * 80)
print("CLASS DISTRIBUTION")
print("=" * 80)

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


# =========================================================
# CLASS WEIGHTS
# =========================================================

classes = np.array([0, 1])

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=train_df["binary_label"]
)

class_weights = {
    int(cls): float(weight)
    for cls, weight in zip(classes, weights)
}


print("\n" + "=" * 80)
print("CLASS WEIGHTS")
print("=" * 80)

print(
    "Non-melanoma (0):",
    f"{class_weights[0]:.4f}"
)

print(
    "Melanoma (1):",
    f"{class_weights[1]:.4f}"
)


# =========================================================
# IMAGE LOADING
# =========================================================
#
# IMPORTANT:
#
# ConvNeXtTiny from tf.keras.applications has built-in
# preprocessing.
#
# Therefore:
#
#     image = 0 - 255
#
# DO NOT:
#
#     image / 255.0
#
# DO NOT:
#
#     (image - mean) / std
#
# =========================================================

def load_image(path, label):

    image = tf.io.read_file(path)

    image = tf.image.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    image.set_shape(
        [None, None, 3]
    )

    image = tf.image.resize(
        image,
        IMAGE_SIZE,
        method=tf.image.ResizeMethod.BILINEAR
    )

    image = tf.cast(
        image,
        tf.float32
    )

    # -----------------------------------------------------
    # IMPORTANT:
    # Keep image in 0-255 range.
    # ConvNeXt handles preprocessing internally.
    # -----------------------------------------------------

    label = tf.cast(
        label,
        tf.float32
    )

    return image, label


# =========================================================
# DATASET CREATION
# =========================================================

def create_dataset(
    dataframe,
    training=False
):

    paths = (
        dataframe["image_path"]
        .astype(str)
        .to_numpy()
    )

    labels = (
        dataframe["binary_label"]
        .astype(np.float32)
        .to_numpy()
    )

    dataset = tf.data.Dataset.from_tensor_slices(
        (
            paths,
            labels
        )
    )

    if training:

        dataset = dataset.shuffle(
            buffer_size=len(dataframe),
            seed=RANDOM_SEED,
            reshuffle_each_iteration=True
        )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.batch(
        BATCH_SIZE
    )

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset


print("\n" + "=" * 80)
print("CREATING DATASETS")
print("=" * 80)

train_dataset = create_dataset(
    train_df,
    training=True
)

val_dataset = create_dataset(
    val_df,
    training=False
)

print("\nTraining dataset created.")
print("Validation dataset created.")


# =========================================================
# DATA AUGMENTATION
# =========================================================

data_augmentation = tf.keras.Sequential(
    [

        tf.keras.layers.RandomFlip(
            mode="horizontal_and_vertical"
        ),

        tf.keras.layers.RandomRotation(
            0.05
        ),

        tf.keras.layers.RandomZoom(
            0.10
        ),

        tf.keras.layers.RandomTranslation(
            height_factor=0.03,
            width_factor=0.03
        ),

        tf.keras.layers.RandomContrast(
            0.10
        ),

    ],
    name="train_only_augmentation"
)


# =========================================================
# LOAD CONVNEXT
# =========================================================

print("\n" + "=" * 80)
print("LOADING IMAGENET CONVNEXT")
print("=" * 80)

base_model = (
    tf.keras.applications.ConvNeXtTiny(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3),

        # IMPORTANT:
        # ConvNeXt performs preprocessing internally.
        include_preprocessing=True
    )
)

# ---------------------------------------------------------
# Stage 1:
# Freeze ImageNet backbone.
# ---------------------------------------------------------

base_model.trainable = False


print("\nConvNeXtTiny loaded.")

print(
    "Base model trainable:",
    base_model.trainable
)

print(
    "ConvNeXt preprocessing:",
    "BUILT-IN"
)


# =========================================================
# BUILD MODEL
# =========================================================

inputs = tf.keras.Input(
    shape=(224, 224, 3),
    name="image_input"
)


# ---------------------------------------------------------
# Augmentation
# ---------------------------------------------------------

x = data_augmentation(
    inputs
)


# ---------------------------------------------------------
# ConvNeXt
# ---------------------------------------------------------

x = base_model(
    x,
    training=False
)


# ---------------------------------------------------------
# Global pooling
# ---------------------------------------------------------

x = tf.keras.layers.GlobalAveragePooling2D(
    name="global_average_pooling"
)(
    x
)


# ---------------------------------------------------------
# Batch normalization
# ---------------------------------------------------------

x = tf.keras.layers.BatchNormalization(
    name="classifier_batch_norm"
)(
    x
)


# ---------------------------------------------------------
# Dropout
# ---------------------------------------------------------

x = tf.keras.layers.Dropout(
    0.35,
    name="classifier_dropout"
)(
    x
)


# ---------------------------------------------------------
# Binary classification
# ---------------------------------------------------------

outputs = tf.keras.layers.Dense(
    1,
    activation="sigmoid",
    name="melanoma_probability"
)(
    x
)


# ---------------------------------------------------------
# Final model
# ---------------------------------------------------------

model = tf.keras.Model(
    inputs=inputs,
    outputs=outputs,
    name="ConvNeXtTiny_Melanoma"
)


# =========================================================
# COMPILE
# =========================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-4
    ),

    loss=tf.keras.losses.BinaryCrossentropy(),

    metrics=[

        tf.keras.metrics.BinaryAccuracy(
            name="accuracy"
        ),

        tf.keras.metrics.AUC(
            name="auc",
            curve="ROC"
        ),

        tf.keras.metrics.AUC(
            name="pr_auc",
            curve="PR"
        ),

        tf.keras.metrics.Precision(
            name="precision"
        ),

        tf.keras.metrics.Recall(
            name="recall"
        ),

    ]
)


# =========================================================
# MODEL SUMMARY
# =========================================================

print("\n" + "=" * 80)
print("CONVNEXT MODEL SUMMARY")
print("=" * 80)

model.summary()


# =========================================================
# CALLBACKS
# =========================================================

callbacks = [

    # -----------------------------------------------------
    # Save best model according to validation ROC-AUC
    # -----------------------------------------------------

    tf.keras.callbacks.ModelCheckpoint(

        filepath=MODEL_PATH,

        monitor="val_auc",

        mode="max",

        save_best_only=True,

        verbose=1
    ),


    # -----------------------------------------------------
    # Early stopping
    # -----------------------------------------------------

    tf.keras.callbacks.EarlyStopping(

        monitor="val_auc",

        mode="max",

        patience=2,

        restore_best_weights=True,

        verbose=1
    ),


    # -----------------------------------------------------
    # Reduce learning rate
    # -----------------------------------------------------

    tf.keras.callbacks.ReduceLROnPlateau(

        monitor="val_auc",

        mode="max",

        factor=0.5,

        patience=1,

        min_lr=1e-6,

        verbose=1
    ),

]


# =========================================================
# TRAINING
# =========================================================

print("\n" + "=" * 80)
print("STARTING CONVNEXT TRAINING")
print("=" * 80)

print("\nEpochs    :", EPOCHS)
print("Batch size:", BATCH_SIZE)
print("Learning rate:", "1e-4")

print("\nIMPORTANT:")
print("Training preprocessing:")
print("  Resize -> Keep 0-255 -> ConvNeXt built-in preprocessing")

print("\nClass weights:")
print(class_weights)

print("\nStarting training...\n")


history = model.fit(

    train_dataset,

    validation_data=val_dataset,

    epochs=EPOCHS,

    class_weight=class_weights,

    callbacks=callbacks,

    verbose=1
)


# =========================================================
# SAVE TRAINING HISTORY
# =========================================================

history_df = pd.DataFrame(
    history.history
)

history_df.insert(
    0,
    "epoch",
    np.arange(
        1,
        len(history_df) + 1
    )
)

history_df.to_csv(
    HISTORY_PATH,
    index=False
)


# =========================================================
# FIND BEST EPOCH
# =========================================================

best_epoch_index = int(
    np.argmax(
        history.history["val_auc"]
    )
)

best_epoch = (
    best_epoch_index + 1
)

best_val_auc = (
    history.history["val_auc"]
    [best_epoch_index]
)

best_val_accuracy = (
    history.history["val_accuracy"]
    [best_epoch_index]
)

best_val_recall = (
    history.history["val_recall"]
    [best_epoch_index]
)

best_val_pr_auc = (
    history.history["val_pr_auc"]
    [best_epoch_index]
)


# =========================================================
# FINAL STATUS
# =========================================================

print("\n" + "=" * 80)
print("TRAINING COMPLETED")
print("=" * 80)

print("\nBest epoch:")
print(best_epoch)

print(
    "\nBest validation ROC-AUC:",
    f"{best_val_auc:.4f}"
)

print(
    "Best validation PR-AUC:",
    f"{best_val_pr_auc:.4f}"
)

print(
    "Best validation accuracy:",
    f"{best_val_accuracy:.4f}"
)

print(
    "Best validation recall:",
    f"{best_val_recall:.4f}"
)

print("\nBest model saved:")
print(MODEL_PATH)

print("\nTraining history saved:")
print(HISTORY_PATH)

print("\n" + "=" * 80)
print("STATUS: PASS")
print("ConvNeXt Final CLAHE binary training completed.")
print("=" * 80)