from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.utils.class_weight import compute_class_weight


# =========================================================
# CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_CSV = (
    BASE_DIR
    / "data"
    / "processed"
    / "final_binary_clahe_splits"
    / "train_final_clahe.csv"
)

VAL_CSV = (
    BASE_DIR
    / "data"
    / "processed"
    / "final_binary_clahe_splits"
    / "validation_final_clahe.csv"
)

MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = (
    MODEL_DIR
    / "melanoma_resnet101_final_clahe.keras"
)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 8
EPOCHS = 10
NUM_CLASSES = 2
RANDOM_SEED = 42


# =========================================================
# REPRODUCIBILITY
# =========================================================

tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


print("=" * 75)
print("MELANOMA - RESNET101 FINAL CLAHE TRAINING")
print("=" * 75)

print("\nTensorFlow version:", tf.__version__)


# =========================================================
# CHECK INPUT FILES
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
# LOAD DATA
# =========================================================

train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VAL_CSV)

print("\nTrain records:", len(train_df))
print("Validation records:", len(val_df))


# =========================================================
# CHECK REQUIRED COLUMNS
# =========================================================

required_columns = [
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

train_df["binary_label"] = (
    train_df["binary_label"]
    .astype(int)
)

val_df["binary_label"] = (
    val_df["binary_label"]
    .astype(int)
)


# =========================================================
# IMAGE PATH
# =========================================================

def make_absolute_path(path):

    path = Path(path)

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


print(
    "\nMissing training images:",
    len(missing_train)
)

print(
    "Missing validation images:",
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
        "Some training images are missing."
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
        "Some validation images are missing."
    )


# =========================================================
# CLASS DISTRIBUTION
# =========================================================

print("\n" + "=" * 75)
print("TRAINING CLASS DISTRIBUTION")
print("=" * 75)

print(
    train_df["binary_diagnosis"]
    .value_counts()
)


print("\nBinary label distribution:")

print(
    train_df["binary_label"]
    .value_counts()
    .sort_index()
)


print("\nValidation class distribution:")

print(
    val_df["binary_diagnosis"]
    .value_counts()
)


# =========================================================
# CLASS WEIGHTS
# =========================================================

classes = np.unique(
    train_df["binary_label"]
)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=train_df["binary_label"]
)

class_weights = {
    int(cls): float(weight)
    for cls, weight in zip(
        classes,
        weights
    )
}


print("\n" + "=" * 75)
print("CLASS WEIGHTS")
print("=" * 75)

print(
    "Non-melanoma (0):",
    class_weights.get(0)
)

print(
    "Melanoma (1):",
    class_weights.get(1)
)


# =========================================================
# IMAGE LOADING
# =========================================================

def load_image(path, label):

    image = tf.io.read_file(path)

    image = tf.image.decode_jpeg(
        image,
        channels=3
    )

    image = tf.image.resize(
        image,
        IMAGE_SIZE
    )

    image = tf.cast(
        image,
        tf.float32
    )

    return image, label


# =========================================================
# DATASET CREATION
# =========================================================

def create_dataset(
    df,
    training=False
):

    paths = df["image_path"].values
    labels = df["binary_label"].values

    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
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
# TRAINING ONLY
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
# RESNET101
# =========================================================

print("\n" + "=" * 75)
print("LOADING IMAGENET RESNET101")
print("=" * 75)


base_model = tf.keras.applications.ResNet101(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3)
)

base_model.trainable = False


# =========================================================
# MODEL
# =========================================================

inputs = tf.keras.Input(
    shape=(224, 224, 3),
    name="image"
)

x = data_augmentation(inputs)

x = tf.keras.applications.resnet.preprocess_input(
    x
)

x = base_model(
    x,
    training=False
)

x = tf.keras.layers.GlobalAveragePooling2D()(x)

x = tf.keras.layers.Dropout(
    0.30
)(x)

outputs = tf.keras.layers.Dense(
    NUM_CLASSES,
    activation="softmax",
    name="prediction"
)(x)


model = tf.keras.Model(
    inputs=inputs,
    outputs=outputs,
    name="ResNet101_Final_CLAHE"
)


# =========================================================
# COMPILE
# =========================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss="sparse_categorical_crossentropy",

    metrics=[
        "accuracy"
    ]
)


# =========================================================
# MODEL SUMMARY
# =========================================================

print("\n" + "=" * 75)
print("RESNET101 FINAL CLAHE MODEL SUMMARY")
print("=" * 75)

model.summary()


# =========================================================
# CALLBACKS
# =========================================================

callbacks = [

    tf.keras.callbacks.ModelCheckpoint(

        filepath=MODEL_PATH,

        monitor="val_accuracy",

        save_best_only=True,

        verbose=1
    ),

    tf.keras.callbacks.EarlyStopping(

        monitor="val_loss",

        patience=3,

        restore_best_weights=True,

        verbose=1
    ),

    tf.keras.callbacks.ReduceLROnPlateau(

        monitor="val_loss",

        factor=0.5,

        patience=2,

        verbose=1
    ),

]


# =========================================================
# TRAINING
# =========================================================

print("\n" + "=" * 75)
print("STARTING RESNET101 FINAL CLAHE TRAINING")
print("=" * 75)

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

history_path = (
    BASE_DIR
    / "results"
    / "resnet101_final_clahe_training_history.csv"
)

history_path.parent.mkdir(
    parents=True,
    exist_ok=True
)

history_df.to_csv(
    history_path,
    index=False
)


# =========================================================
# COMPLETED
# =========================================================

print("\n" + "=" * 75)
print("RESNET101 FINAL CLAHE TRAINING COMPLETED")
print("=" * 75)

print("\nBest model saved at:")

print(MODEL_PATH)

print("\nTraining history saved at:")

print(history_path)

print("\nTraining completed successfully.")