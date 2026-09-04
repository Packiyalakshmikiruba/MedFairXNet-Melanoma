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
    / "melanoma_efficientnetv2_final_clahe.keras"
)

HISTORY_PATH = (
    RESULTS_DIR
    / "efficientnetv2_final_clahe_training_history.csv"
)

IMAGE_SIZE = (224, 224)

BATCH_SIZE = 8

EPOCHS = 15

NUM_CLASSES = 2

RANDOM_SEED = 42


# =========================================================
# REPRODUCIBILITY
# =========================================================

tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# =========================================================
# HEADER
# =========================================================

print("=" * 75)
print("MELONMA - EFFICIENTNETV2 FINAL CLAHE TRAINING")
print("=" * 75)

print("\nTensorFlow version:", tf.__version__)

print("\nTrain CSV:")
print(TRAIN_CSV)

print("\nValidation CSV:")
print(VAL_CSV)

print("\nModel output:")
print(MODEL_PATH)


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


print("\n" + "=" * 75)
print("DATASET INFORMATION")
print("=" * 75)

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
# VERIFY IMAGES
# =========================================================

missing_train = train_df[
    ~train_df["image_path"]
    .apply(lambda x: Path(x).exists())
]

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


print("\nTrain/Validation image overlap:", len(overlap))


if len(overlap) > 0:

    raise ValueError(
        "Data leakage detected between train and validation."
    )


# =========================================================
# CLASS DISTRIBUTION
# =========================================================

print("\n" + "=" * 75)
print("CLASS DISTRIBUTION")
print("=" * 75)

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


print("\n" + "=" * 75)
print("CLASS WEIGHTS")
print("=" * 75)

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
        IMAGE_SIZE
    )

    image = tf.cast(
        image,
        tf.float32
    )

    label = tf.cast(
        label,
        tf.int32
    )

    return image, label


# =========================================================
# DATASET CREATION
# =========================================================

def create_dataset(
    dataframe,
    training=False
):

    paths = dataframe[
        "image_path"
    ].values

    labels = dataframe[
        "binary_label"
    ].values

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
# EFFICIENTNETV2
# =========================================================

print("\n" + "=" * 75)
print("LOADING IMAGENET EFFICIENTNETV2")
print("=" * 75)


base_model = (
    tf.keras.applications.EfficientNetV2B0(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3)
    )
)


base_model.trainable = False


print("\nEfficientNetV2B0 loaded.")

print("Base model trainable:", base_model.trainable)


# =========================================================
# MODEL
# =========================================================

inputs = tf.keras.Input(
    shape=(224, 224, 3),
    name="image_input"
)


x = data_augmentation(
    inputs
)


x = base_model(
    x,
    training=False
)


x = tf.keras.layers.GlobalAveragePooling2D()(
    x
)


x = tf.keras.layers.BatchNormalization()(
    x
)


x = tf.keras.layers.Dropout(
    0.35
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
    name="EfficientNetV2B0_Melanoma"
)


# =========================================================
# COMPILE
# =========================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss="binary_crossentropy",

    metrics=[
        tf.keras.metrics.BinaryAccuracy(
            name="accuracy"
        ),

        tf.keras.metrics.AUC(
            name="auc"
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

print("\n" + "=" * 75)
print("EFFICIENTNETV2 MODEL SUMMARY")
print("=" * 75)

model.summary()


# =========================================================
# CALLBACKS
# =========================================================

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

        monitor="val_auc",

        mode="max",

        factor=0.5,

        patience=2,

        min_lr=1e-6,

        verbose=1
    ),

]


# =========================================================
# TRAINING
# =========================================================

print("\n" + "=" * 75)
print("STARTING EFFICIENTNETV2 TRAINING")
print("=" * 75)

print("\nTraining only:")

print("  Augmentation       : ENABLED")

print("  Validation augment : DISABLED")

print("  Test data           : NOT USED")

print("  Class weighting     : ENABLED")


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

history_df.index.name = "epoch"

history_df.reset_index(
    inplace=True
)

history_df.to_csv(
    HISTORY_PATH,
    index=False
)


# =========================================================
# BEST METRICS
# =========================================================

best_epoch = int(
    np.argmax(
        history.history["val_auc"]
    )
)


best_val_auc = (
    history.history["val_auc"]
    [best_epoch]
)

best_val_accuracy = (
    history.history["val_accuracy"]
    [best_epoch]
)


# =========================================================
# FINAL SUMMARY
# =========================================================

print("\n" + "=" * 75)
print("TRAINING COMPLETED")
print("=" * 75)

print("\nBest epoch:")
print(best_epoch + 1)

print(
    "\nBest validation AUC:",
    f"{best_val_auc:.4f}"
)

print(
    "Best validation accuracy:",
    f"{best_val_accuracy:.4f}"
)

print("\nBest model saved:")

print(MODEL_PATH)

print("\nTraining history saved:")

print(HISTORY_PATH)

print("\n" + "=" * 75)
print("STATUS: PASS")
print("EfficientNetV2 final CLAHE binary training completed.")
print("=" * 75)