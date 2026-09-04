from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.utils.class_weight import compute_class_weight


# ============================================================
# MELONMA RESEARCH PROJECT
# EfficientNetV2B0 + PH2 + CLAHE
# IMPROVED TRAINING V2
#
# Main improvements:
# 1. Correct EfficientNetV2 preprocessing
# 2. Data augmentation
# 3. Stage-1 classifier training
# 4. Stage-2 backbone fine-tuning
# 5. Class-balanced training
# 6. AdamW optimizer
# 7. Early stopping
# 8. Reduce learning rate
# 9. Best model selected using validation AUC
# ============================================================


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = (
    Path(__file__).resolve().parent.parent
)


TRAIN_CSV = (
    BASE_DIR
    / "data"
    / "processed"
    / "PH2"
    / "splits"
    / "train_ph2_clahe.csv"
)


VAL_CSV = (
    BASE_DIR
    / "data"
    / "processed"
    / "PH2"
    / "splits"
    / "validation_ph2_clahe.csv"
)


MODEL_DIR = (
    BASE_DIR
    / "models"
)


RESULTS_DIR = (
    BASE_DIR
    / "results"
)


MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# NEW MODEL NAME
# ============================================================

MODEL_PATH = (
    MODEL_DIR
    / "melanoma_efficientnetv2_ph2_clahe_v2.keras"
)


HISTORY_PATH = (
    RESULTS_DIR
    / "efficientnetv2_ph2_clahe_v2_training_history.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = (
    224,
    224
)


BATCH_SIZE = 8


# Stage 1
STAGE1_EPOCHS = 8


# Stage 2
STAGE2_EPOCHS = 25


SEED = 42


# Fine-tune last portion of EfficientNetV2
FINE_TUNE_LAST_LAYERS = 80


# ============================================================
# RANDOM SEEDS
# ============================================================

tf.keras.utils.set_random_seed(
    SEED
)


# ============================================================
# HEADER
# ============================================================

print("=" * 100)
print("MELONMA - IMPROVED EFFICIENTNETV2 PH2 CLAHE TRAINING V2")
print("=" * 100)


print()
print("Project root:")
print(BASE_DIR)


print()
print("Train CSV:")
print(TRAIN_CSV)


print()
print("Validation CSV:")
print(VAL_CSV)


print()
print("Output model:")
print(MODEL_PATH)


print()
print("Image size:")
print(IMAGE_SIZE)


print()
print("Batch size:")
print(BATCH_SIZE)


print()
print("Stage 1 epochs:")
print(STAGE1_EPOCHS)


print()
print("Stage 2 epochs:")
print(STAGE2_EPOCHS)


# ============================================================
# VALIDATE FILES
# ============================================================

print()
print("=" * 100)
print("VALIDATING INPUT FILES")
print("=" * 100)


if not TRAIN_CSV.exists():

    raise FileNotFoundError(
        f"Train CSV not found:\n{TRAIN_CSV}"
    )


if not VAL_CSV.exists():

    raise FileNotFoundError(
        f"Validation CSV not found:\n{VAL_CSV}"
    )


print()
print("Train CSV: OK")
print("Validation CSV: OK")


# ============================================================
# LOAD DATAFRAMES
# ============================================================

print()
print("=" * 100)
print("LOADING DATA")
print("=" * 100)


train_df = pd.read_csv(
    TRAIN_CSV
)


val_df = pd.read_csv(
    VAL_CSV
)


print()
print(
    "Training samples:",
    len(train_df)
)


print(
    "Validation samples:",
    len(val_df)
)


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = [
    "clahe_image_path",
    "binary_label",
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


print()
print("Required columns: OK")


# ============================================================
# VALIDATE LABELS
# ============================================================

print()
print("=" * 100)
print("VALIDATING LABELS")
print("=" * 100)


allowed_labels = {
    0,
    1
}


train_labels = set(
    train_df["binary_label"].astype(int).unique()
)


val_labels = set(
    val_df["binary_label"].astype(int).unique()
)


if not train_labels.issubset(
    allowed_labels
):

    raise ValueError(
        f"Invalid training labels: {train_labels}"
    )


if not val_labels.issubset(
    allowed_labels
):

    raise ValueError(
        f"Invalid validation labels: {val_labels}"
    )


print()
print("Labels are valid.")
print("0 = Non-melanoma")
print("1 = Melanoma")


# ============================================================
# CHECK IMAGE PATHS
# ============================================================

print()
print("=" * 100)
print("CHECKING IMAGE FILES")
print("=" * 100)


train_missing = []


for _, row in train_df.iterrows():

    path = Path(
        str(
            row["clahe_image_path"]
        )
    )

    if not path.exists():

        train_missing.append(
            str(path)
        )


val_missing = []


for _, row in val_df.iterrows():

    path = Path(
        str(
            row["clahe_image_path"]
        )
    )

    if not path.exists():

        val_missing.append(
            str(path)
        )


print()
print(
    "Missing training images:",
    len(train_missing)
)


print(
    "Missing validation images:",
    len(val_missing)
)


if train_missing:

    print()
    print("First missing training files:")

    for path in train_missing[:10]:

        print(path)

    raise FileNotFoundError(
        "Training images are missing."
    )


if val_missing:

    print()
    print("First missing validation files:")

    for path in val_missing[:10]:

        print(path)

    raise FileNotFoundError(
        "Validation images are missing."
    )


print()
print("All training images exist.")
print("All validation images exist.")


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print()
print("=" * 100)
print("CLASS DISTRIBUTION")
print("=" * 100)


train_counts = (
    train_df["binary_label"]
    .value_counts()
    .sort_index()
)


val_counts = (
    val_df["binary_label"]
    .value_counts()
    .sort_index()
)


print()
print("TRAINING")


print(
    "Non-melanoma:",
    int(train_counts.get(0, 0))
)


print(
    "Melanoma    :",
    int(train_counts.get(1, 0))
)


print()
print("VALIDATION")


print(
    "Non-melanoma:",
    int(val_counts.get(0, 0))
)


print(
    "Melanoma    :",
    int(val_counts.get(1, 0))
)


# ============================================================
# CLASS WEIGHTS
# ============================================================

print()
print("=" * 100)
print("CALCULATING CLASS WEIGHTS")
print("=" * 100)


classes = np.array(
    [0, 1]
)


class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=train_df[
        "binary_label"
    ].astype(int).values
)


class_weights = {

    int(classes[i]):
    float(class_weights_array[i])

    for i in range(
        len(classes)
    )
}


print()
print(
    "Class weights:"
)


print(
    class_weights
)


# ============================================================
# IMAGE LOADER
# ============================================================

def load_image(
    path,
    label
):

    # --------------------------------------------------------
    # READ IMAGE
    # --------------------------------------------------------

    image = tf.io.read_file(
        path
    )


    # --------------------------------------------------------
    # DECODE PNG
    # --------------------------------------------------------

    image = tf.image.decode_png(
        image,
        channels=3
    )


    # --------------------------------------------------------
    # RESIZE
    # --------------------------------------------------------

    image = tf.image.resize(
        image,
        IMAGE_SIZE,
        method="bilinear"
    )


    # --------------------------------------------------------
    # FLOAT32
    #
    # IMPORTANT:
    #
    # EfficientNetV2 with
    # include_preprocessing=True
    # expects pixel values in 0-255.
    #
    # Therefore DO NOT perform:
    #
    # image / 255
    #
    # and DO NOT perform:
    #
    # ImageNet mean/std normalization.
    # --------------------------------------------------------

    image = tf.cast(
        image,
        tf.float32
    )


    label = tf.cast(
        label,
        tf.float32
    )


    return (
        image,
        label
    )


# ============================================================
# DATA AUGMENTATION
# ============================================================

data_augmentation = tf.keras.Sequential(

    [

        tf.keras.layers.RandomFlip(
            mode="horizontal"
        ),

        tf.keras.layers.RandomRotation(
            factor=0.08
        ),

        tf.keras.layers.RandomZoom(
            height_factor=0.10,
            width_factor=0.10
        ),

        tf.keras.layers.RandomTranslation(
            height_factor=0.05,
            width_factor=0.05
        ),

        tf.keras.layers.RandomContrast(
            factor=0.10
        ),

    ],

    name="medical_data_augmentation"

)


# ============================================================
# TRAIN IMAGE AUGMENTATION FUNCTION
# ============================================================

def augment_image(
    image,
    label
):

    image = data_augmentation(
        image,
        training=True
    )


    return (
        image,
        label
    )


# ============================================================
# TRAIN DATASET
# ============================================================

print()
print("=" * 100)
print("CREATING TRAIN DATASET")
print("=" * 100)


train_dataset = (
    tf.data.Dataset.from_tensor_slices(

        (
            train_df[
                "clahe_image_path"
            ].astype(str).values,

            train_df[
                "binary_label"
            ].astype(np.float32).values,

        )

    )

    .shuffle(
        buffer_size=len(train_df),
        seed=SEED,
        reshuffle_each_iteration=True
    )

    .map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    .map(
        augment_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    .batch(
        BATCH_SIZE
    )

    .prefetch(
        tf.data.AUTOTUNE
    )
)


# ============================================================
# VALIDATION DATASET
# ============================================================

print()
print("Creating validation dataset...")


val_dataset = (

    tf.data.Dataset.from_tensor_slices(

        (
            val_df[
                "clahe_image_path"
            ].astype(str).values,

            val_df[
                "binary_label"
            ].astype(np.float32).values,

        )

    )

    .map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    .batch(
        BATCH_SIZE
    )

    .prefetch(
        tf.data.AUTOTUNE
    )

)


print()
print("Datasets created successfully.")


# ============================================================
# BUILD EFFICIENTNETV2B0
# ============================================================

print()
print("=" * 100)
print("BUILDING EFFICIENTNETV2B0")
print("=" * 100)


base_model = (
    tf.keras.applications.EfficientNetV2B0(

        include_top=False,

        weights="imagenet",

        input_shape=(
            IMAGE_SIZE[0],
            IMAGE_SIZE[1],
            3
        ),

        include_preprocessing=True,

    )
)


# ============================================================
# STAGE 1 - FREEZE BACKBONE
# ============================================================

base_model.trainable = False


print()
print("Stage 1:")
print("EfficientNetV2 backbone = FROZEN")
print("Classification head = TRAINABLE")


# ============================================================
# MODEL INPUT
# ============================================================

inputs = tf.keras.Input(

    shape=(
        IMAGE_SIZE[0],
        IMAGE_SIZE[1],
        3
    ),

    name="input_image"

)


# ============================================================
# AUGMENTATION INSIDE MODEL
#
# We already augment in tf.data.
# Therefore no second augmentation here.
# ============================================================

x = base_model(
    inputs,
    training=False
)


# ============================================================
# GLOBAL AVERAGE POOLING
# ============================================================

x = tf.keras.layers.GlobalAveragePooling2D(
    name="global_average_pooling"
)(x)


# ============================================================
# DROPOUT
# ============================================================

x = tf.keras.layers.Dropout(
    0.35,
    name="dropout"
)(x)


# ============================================================
# CLASSIFICATION HEAD
# ============================================================

outputs = tf.keras.layers.Dense(

    1,

    activation="sigmoid",

    name="melanoma_probability"

)(x)


# ============================================================
# CREATE MODEL
# ============================================================

model = tf.keras.Model(

    inputs=inputs,

    outputs=outputs,

    name="EfficientNetV2B0_PH2_CLAHE_V2"

)


# ============================================================
# STAGE 1 COMPILE
# ============================================================

model.compile(

    optimizer=tf.keras.optimizers.AdamW(

        learning_rate=1e-3,

        weight_decay=1e-4

    ),

    loss="binary_crossentropy",

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


# ============================================================
# MODEL SUMMARY
# ============================================================

print()
print("=" * 100)
print("MODEL SUMMARY - STAGE 1")
print("=" * 100)


model.summary()


# ============================================================
# STAGE 1 CALLBACKS
# ============================================================

stage1_best_path = (

    MODEL_DIR
    / "melanoma_efficientnetv2_ph2_clahe_v2_stage1.keras"

)


stage1_callbacks = [

    tf.keras.callbacks.ModelCheckpoint(

        stage1_best_path,

        monitor="val_auc",

        mode="max",

        save_best_only=True,

        verbose=1

    ),

    tf.keras.callbacks.EarlyStopping(

        monitor="val_auc",

        mode="max",

        patience=4,

        restore_best_weights=True,

        verbose=1

    ),

    tf.keras.callbacks.ReduceLROnPlateau(

        monitor="val_auc",

        mode="max",

        factor=0.5,

        patience=2,

        min_lr=1e-6,

        verbose=1

    ),

]


# ============================================================
# STAGE 1 TRAINING
# ============================================================

print()
print("=" * 100)
print("STAGE 1 TRAINING")
print("=" * 100)


history_stage1 = model.fit(

    train_dataset,

    validation_data=val_dataset,

    epochs=STAGE1_EPOCHS,

    class_weight=class_weights,

    callbacks=stage1_callbacks,

    verbose=1

)


# ============================================================
# STAGE 1 BEST RESULT
# ============================================================

stage1_history_df = pd.DataFrame(
    history_stage1.history
)


stage1_best_epoch = (
    stage1_history_df[
        "val_auc"
    ].idxmax()
    + 1
)


stage1_best_auc = (
    stage1_history_df[
        "val_auc"
    ].max()
)


print()
print("=" * 100)
print("STAGE 1 COMPLETED")
print("=" * 100)


print()
print(
    "Best Stage 1 epoch:",
    stage1_best_epoch
)


print(
    "Best Stage 1 validation AUC:",
    f"{stage1_best_auc:.4f}"
)


# ============================================================
# STAGE 2 - FINE TUNING
# ============================================================

print()
print("=" * 100)
print("STAGE 2 - FINE TUNING")
print("=" * 100)


# ------------------------------------------------------------
# Unfreeze backbone
# ------------------------------------------------------------

base_model.trainable = True


print()
print(
    "Total EfficientNetV2 layers:",
    len(base_model.layers)
)


# ------------------------------------------------------------
# Freeze earlier layers
# ------------------------------------------------------------

fine_tune_from = max(

    0,

    len(base_model.layers)
    - FINE_TUNE_LAST_LAYERS

)


for layer in base_model.layers[

    :fine_tune_from

]:

    layer.trainable = False


for layer in base_model.layers[

    fine_tune_from:

]:

    layer.trainable = True


# ------------------------------------------------------------
# IMPORTANT:
# BatchNormalization layers remain frozen.
# ------------------------------------------------------------

for layer in base_model.layers:

    if isinstance(
        layer,
        tf.keras.layers.BatchNormalization
    ):

        layer.trainable = False


print()
print(
    "Fine-tuning last layers:",
    FINE_TUNE_LAST_LAYERS
)


# ============================================================
# COUNT TRAINABLE PARAMETERS
# ============================================================

trainable_params = np.sum(

    [
        np.prod(
            variable.shape
        )

        for variable in model.trainable_weights

    ]

)


non_trainable_params = np.sum(

    [
        np.prod(
            variable.shape
        )

        for variable in model.non_trainable_weights

    ]

)


print()
print(
    "Trainable parameters:",
    trainable_params
)


print(
    "Non-trainable parameters:",
    non_trainable_params
)


# ============================================================
# RECOMPILE FOR FINE-TUNING
# ============================================================

model.compile(

    optimizer=tf.keras.optimizers.AdamW(

        learning_rate=1e-5,

        weight_decay=1e-5

    ),

    loss="binary_crossentropy",

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


# ============================================================
# STAGE 2 CALLBACKS
# ============================================================

stage2_callbacks = [

    tf.keras.callbacks.ModelCheckpoint(

        MODEL_PATH,

        monitor="val_auc",

        mode="max",

        save_best_only=True,

        verbose=1

    ),

    tf.keras.callbacks.EarlyStopping(

        monitor="val_auc",

        mode="max",

        patience=6,

        restore_best_weights=True,

        verbose=1

    ),

    tf.keras.callbacks.ReduceLROnPlateau(

        monitor="val_auc",

        mode="max",

        factor=0.5,

        patience=3,

        min_lr=1e-7,

        verbose=1

    ),

]


# ============================================================
# STAGE 2 TRAINING
# ============================================================

print()
print("=" * 100)
print("STARTING STAGE 2 FINE-TUNING")
print("=" * 100)


history_stage2 = model.fit(

    train_dataset,

    validation_data=val_dataset,

    epochs=STAGE2_EPOCHS,

    class_weight=class_weights,

    callbacks=stage2_callbacks,

    verbose=1

)


# ============================================================
# SAVE STAGE HISTORIES
# ============================================================

stage1_history_df.insert(

    0,

    "stage",

    "stage1"

)


stage2_history_df = pd.DataFrame(
    history_stage2.history
)


stage2_history_df.insert(

    0,

    "stage",

    "stage2"

)


# Stage 2 epoch numbering continues
stage2_history_df.insert(

    1,

    "epoch",

    np.arange(

        STAGE1_EPOCHS + 1,

        STAGE1_EPOCHS
        + len(stage2_history_df)
        + 1

    )

)


stage1_history_df.insert(

    1,

    "epoch",

    np.arange(

        1,

        len(stage1_history_df) + 1

    )

)


combined_history = pd.concat(

    [

        stage1_history_df,

        stage2_history_df

    ],

    ignore_index=True

)


combined_history.to_csv(

    HISTORY_PATH,

    index=False

)


# ============================================================
# SAVE FINAL MODEL
# ============================================================

model.save(
    MODEL_PATH
)


# ============================================================
# BEST VALIDATION RESULTS
# ============================================================

best_index = (
    combined_history[
        "val_auc"
    ].idxmax()
)


best_epoch = int(
    combined_history.loc[
        best_index,
        "epoch"
    ]
)


best_val_auc = float(
    combined_history.loc[
        best_index,
        "val_auc"
    ]
)


best_val_accuracy = float(
    combined_history.loc[
        best_index,
        "val_accuracy"
    ]
)


best_val_pr_auc = float(
    combined_history.loc[
        best_index,
        "val_pr_auc"
    ]
)


# ============================================================
# FINAL MODEL INFORMATION
# ============================================================

print()
print("=" * 100)
print("FINAL MODEL INFORMATION")
print("=" * 100)


print()
print(
    "Output model:"
)


print(
    MODEL_PATH
)


print()
print(
    "Model output shape:",
    model.output_shape
)


print()
print(
    "Last layer:",
    model.layers[-1].name
)


print(
    "Last activation:",
    model.layers[-1].activation
)


# ============================================================
# FINAL RESULTS
# ============================================================

print()
print("=" * 100)
print("PH2 EFFICIENTNETV2 TRAINING COMPLETED")
print("=" * 100)


print()
print(
    "Best epoch:",
    best_epoch
)


print(
    "Best validation AUC:",
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


print()
print(
    "Trainable parameters:",
    trainable_params
)


print(
    "Non-trainable parameters:",
    non_trainable_params
)


print()
print(
    "Best model saved:"
)


print(
    MODEL_PATH
)


print()
print(
    "Training history saved:"
)


print(
    HISTORY_PATH
)


# ============================================================
# FINAL STATUS
# ============================================================

print()
print("=" * 100)
print("STATUS: PASS")
print("=" * 100)


print()
print(
    "Improved EfficientNetV2 PH2 CLAHE training completed."
)


print()
print(
    "Next step:"
)


print(
    "Run the updated PH2 evaluation script "
    "using the V2 model."
)


print("=" * 100)