from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.utils.class_weight import compute_class_weight


# =========================================================
# MELONMA
# DenseNet121 Binary Melanoma Classification
# CLAHE + ROI Dataset
# =========================================================


# =========================================================
# CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent


TRAIN_CSV = (
    BASE_DIR
    / "data"
    / "processed"
    / "binary_roi_splits"
    / "train_binary_roi.csv"
)

VAL_CSV = (
    BASE_DIR
    / "data"
    / "processed"
    / "binary_roi_splits"
    / "validation_binary_roi.csv"
)


CLAHE_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "images_clahe"
)


MODEL_DIR = (
    BASE_DIR
    / "models"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


MODEL_PATH = (
    MODEL_DIR
    / "melanoma_densenet121_clahe.keras"
)


HISTORY_PATH = (
    BASE_DIR
    / "results"
    / "densenet121_binary_training_history.csv"
)


IMAGE_SIZE = (224, 224)

BATCH_SIZE = 8

EPOCHS = 15

NUM_CLASSES = 2

RANDOM_SEED = 42


# =========================================================
# REPRODUCIBILITY
# =========================================================

tf.random.set_seed(
    RANDOM_SEED
)

np.random.seed(
    RANDOM_SEED
)


# =========================================================
# INFORMATION
# =========================================================

print("=" * 75)
print("MELONMA - DENSENET121 BINARY MELANOMA TRAINING")
print("=" * 75)

print()
print("TensorFlow version :", tf.__version__)
print("Image size         :", IMAGE_SIZE)
print("Batch size         :", BATCH_SIZE)
print("Epochs             :", EPOCHS)

print()
print("Training CSV:")
print(TRAIN_CSV)

print()
print("Validation CSV:")
print(VAL_CSV)

print()
print("CLAHE image directory:")
print(CLAHE_DIR)

print()
print("Model output:")
print(MODEL_PATH)

print("=" * 75)


# =========================================================
# CHECK INPUT FILES
# =========================================================

if not TRAIN_CSV.exists():

    raise FileNotFoundError(
        f"\nTraining CSV not found:\n{TRAIN_CSV}"
    )


if not VAL_CSV.exists():

    raise FileNotFoundError(
        f"\nValidation CSV not found:\n{VAL_CSV}"
    )


if not CLAHE_DIR.exists():

    raise FileNotFoundError(
        f"\nCLAHE image directory not found:\n{CLAHE_DIR}"
    )


# =========================================================
# LOAD DATA
# =========================================================

train_df = pd.read_csv(
    TRAIN_CSV
)

val_df = pd.read_csv(
    VAL_CSV
)


print()
print("=" * 75)
print("DATASET INFORMATION")
print("=" * 75)

print()
print("Training records   :", len(train_df))
print("Validation records :", len(val_df))


# =========================================================
# REQUIRED COLUMNS
# =========================================================

required_columns = [
    "image_id",
    "image_path",
    "binary_label",
    "binary_diagnosis",
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
# BINARY LABEL VALIDATION
# =========================================================

print()
print("=" * 75)
print("BINARY CLASS DISTRIBUTION")
print("=" * 75)


print()
print("TRAINING:")

print(
    train_df[
        "binary_diagnosis"
    ].value_counts()
)


print()
print("VALIDATION:")

print(
    val_df[
        "binary_diagnosis"
    ].value_counts()
)


# =========================================================
# VALIDATE LABEL VALUES
# =========================================================

valid_labels = {
    0,
    1
}


train_labels = set(
    train_df[
        "binary_label"
    ].unique()
)


val_labels = set(
    val_df[
        "binary_label"
    ].unique()
)


if not train_labels.issubset(
    valid_labels
):

    raise ValueError(
        f"Invalid training labels found: {train_labels}"
    )


if not val_labels.issubset(
    valid_labels
):

    raise ValueError(
        f"Invalid validation labels found: {val_labels}"
    )


# Make sure labels are integer

train_df[
    "binary_label"
] = train_df[
    "binary_label"
].astype(int)


val_df[
    "binary_label"
] = val_df[
    "binary_label"
].astype(int)


# =========================================================
# IMAGE PATH
# =========================================================

def make_clahe_path(image_id):

    """
    Convert image_id into the corresponding
    CLAHE processed image path.
    """

    image_id = str(
        image_id
    )

    return str(
        CLAHE_DIR
        / f"{image_id}.jpg"
    )


train_df[
    "clahe_path"
] = train_df[
    "image_id"
].apply(
    make_clahe_path
)


val_df[
    "clahe_path"
] = val_df[
    "image_id"
].apply(
    make_clahe_path
)


# =========================================================
# VERIFY CLAHE IMAGES
# =========================================================

print()
print("=" * 75)
print("VERIFYING CLAHE IMAGES")
print("=" * 75)


missing_train = train_df[
    ~train_df[
        "clahe_path"
    ].apply(
        lambda x: Path(x).exists()
    )
]


missing_val = val_df[
    ~val_df[
        "clahe_path"
    ].apply(
        lambda x: Path(x).exists()
    )
]


print()
print(
    "Missing training CLAHE images   :",
    len(missing_train)
)


print(
    "Missing validation CLAHE images :",
    len(missing_val)
)


if len(missing_train) > 0:

    print()
    print(
        "First missing training images:"
    )

    print(
        missing_train[
            [
                "image_id",
                "clahe_path"
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    raise FileNotFoundError(
        "Training CLAHE images are missing."
    )


if len(missing_val) > 0:

    print()
    print(
        "First missing validation images:"
    )

    print(
        missing_val[
            [
                "image_id",
                "clahe_path"
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    raise FileNotFoundError(
        "Validation CLAHE images are missing."
    )


# =========================================================
# CLASS WEIGHTS
# =========================================================

print()
print("=" * 75)
print("CLASS WEIGHTS")
print("=" * 75)


classes = np.array(
    [0, 1]
)


weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=train_df[
        "binary_label"
    ].values
)


class_weights = {

    int(cls): float(weight)

    for cls, weight in zip(
        classes,
        weights
    )

}


print()
print(
    "Non-melanoma (0):",
    f"{class_weights[0]:.4f}"
)


print(
    "Melanoma (1)    :",
    f"{class_weights[1]:.4f}"
)


# =========================================================
# IMAGE LOADING
# =========================================================

def load_image(
    path,
    label
):

    image = tf.io.read_file(
        path
    )

    image = tf.image.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    image = tf.image.resize(
        image,
        IMAGE_SIZE
    )

    image = tf.cast(
        image,
        tf.float32
    )

    return (
        image,
        label
    )


# =========================================================
# DATASET CREATION
# =========================================================

def create_dataset(
    df,
    training=False
):

    paths = df[
        "clahe_path"
    ].values

    labels = df[
        "binary_label"
    ].values


    dataset = (
        tf.data.Dataset
        .from_tensor_slices(
            (
                paths,
                labels
            )
        )
    )


    if training:

        dataset = dataset.shuffle(
            buffer_size=len(df),
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


train_dataset = create_dataset(
    train_df,
    training=True
)


val_dataset = create_dataset(
    val_df,
    training=False
)


# =========================================================
# DATA AUGMENTATION
# =========================================================

data_augmentation = tf.keras.Sequential(

    [

        tf.keras.layers.RandomFlip(
            "horizontal_and_vertical"
        ),

        tf.keras.layers.RandomRotation(
            0.05
        ),

        tf.keras.layers.RandomZoom(
            0.10
        ),

        tf.keras.layers.RandomContrast(
            0.10
        ),

    ],

    name="data_augmentation"
)


# =========================================================
# LOAD DENSENET121
# =========================================================

print()
print("=" * 75)
print("LOADING IMAGENET DENSENET121")
print("=" * 75)


base_model = (
    tf.keras.applications.DenseNet121(

        include_top=False,

        weights="imagenet",

        input_shape=(
            224,
            224,
            3
        )
    )
)


# Freeze pretrained layers

base_model.trainable = False


print()
print(
    "DenseNet121 loaded successfully."
)

print(
    "Pretrained layers frozen."
)


# =========================================================
# MODEL
# =========================================================

inputs = tf.keras.Input(
    shape=(
        224,
        224,
        3
    ),
    name="image"
)


x = data_augmentation(
    inputs
)


x = base_model(
    x,
    training=False
)


x = tf.keras.layers.GlobalAveragePooling2D(
    name="global_average_pooling"
)(
    x
)


x = tf.keras.layers.Dropout(
    0.40,
    name="dropout"
)(
    x
)


outputs = tf.keras.layers.Dense(
    1,
    activation="sigmoid",
    name="melanoma_probability"
)(
    x
)


model = tf.keras.Model(
    inputs=inputs,
    outputs=outputs,
    name="Melanoma_DenseNet121"
)


# =========================================================
# COMPILE
# =========================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
    ),

    loss=tf.keras.losses.BinaryCrossentropy(),

    metrics=[
        tf.keras.metrics.BinaryAccuracy(
            name="accuracy"
        ),

        tf.keras.metrics.Precision(
            name="precision"
        ),

        tf.keras.metrics.Recall(
            name="recall"
        ),

        tf.keras.metrics.AUC(
            name="auc"
        ),
    ]
)


# =========================================================
# MODEL SUMMARY
# =========================================================

print()
print("=" * 75)
print("MODEL SUMMARY")
print("=" * 75)


model.summary()


# =========================================================
# CALLBACKS
# =========================================================

HISTORY_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)


callbacks = [

    tf.keras.callbacks.ModelCheckpoint(

        filepath=MODEL_PATH,

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

        monitor="val_loss",

        factor=0.5,

        patience=2,

        min_lr=1e-7,

        verbose=1
    ),

]


# =========================================================
# TRAINING
# =========================================================

print()
print("=" * 75)
print("STARTING DENSENET121 BINARY TRAINING")
print("=" * 75)

print()
print("Task:")
print("Non-melanoma vs Melanoma")

print()
print("Pipeline:")
print("ROI")
print("  ↓")
print("Hair Removal")
print("  ↓")
print("CLAHE")
print("  ↓")
print("DenseNet121")
print("  ↓")
print("Binary Classification")


history = model.fit(

    train_dataset,

    validation_data=val_dataset,

    epochs=EPOCHS,

    class_weight=class_weights,

    callbacks=callbacks

)


# =========================================================
# SAVE TRAINING HISTORY
# =========================================================

history_df = pd.DataFrame(
    history.history
)


history_df.index.name = (
    "epoch"
)


history_df.to_csv(
    HISTORY_PATH
)


# =========================================================
# FINAL RESULTS
# =========================================================

print()
print("=" * 75)
print("TRAINING COMPLETED")
print("=" * 75)


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


print()
print(
    "Best validation AUC:"
)

print(
    f"{max(history.history['val_auc']):.4f}"
)


print()
print(
    "Best validation accuracy:"
)

print(
    f"{max(history.history['val_accuracy']):.4f}"
)


print()
print("=" * 75)
print("STATUS: PASS")
print("DenseNet121 binary melanoma training completed.")
print("=" * 75)