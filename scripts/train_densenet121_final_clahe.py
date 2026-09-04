from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.utils.class_weight import compute_class_weight


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
    / "melanoma_densenet121_final_clahe.keras"
)

HISTORY_PATH = (
    RESULTS_DIR
    / "densenet121_final_clahe_training_history.csv"
)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 8
EPOCHS = 15
NUM_CLASSES = 2
RANDOM_SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ============================================================
# HEADER
# ============================================================

print("=" * 75)
print("MELONMA - DENSENET121 FINAL CLAHE BINARY TRAINING")
print("=" * 75)

print("\nTensorFlow version:", tf.__version__)

print("\nTraining CSV:")
print(TRAIN_CSV)

print("\nValidation CSV:")
print(VAL_CSV)

print("\nModel output:")
print(MODEL_PATH)


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

print("\n" + "=" * 75)
print("DATASET INFORMATION")
print("=" * 75)

print("\nTraining images   :", len(train_df))
print("Validation images :", len(val_df))


# ============================================================
# REQUIRED COLUMNS
# ============================================================

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


# ============================================================
# LABEL VALIDATION
# ============================================================

train_df["label"] = train_df["binary_label"].astype(int)
val_df["label"] = val_df["binary_label"].astype(int)

valid_labels = {0, 1}

if not set(train_df["label"].unique()).issubset(valid_labels):
    raise ValueError("Invalid training labels detected.")

if not set(val_df["label"].unique()).issubset(valid_labels):
    raise ValueError("Invalid validation labels detected.")


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\n" + "=" * 75)
print("TRAIN CLASS DISTRIBUTION")
print("=" * 75)

print(
    train_df["binary_diagnosis"]
    .value_counts()
)


print("\n" + "=" * 75)
print("VALIDATION CLASS DISTRIBUTION")
print("=" * 75)

print(
    val_df["binary_diagnosis"]
    .value_counts()
)


# ============================================================
# IMAGE PATH VALIDATION
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
        "Training CLAHE images are missing."
    )


if len(missing_val) > 0:

    print("\nFirst missing validation images:")

    print(
        missing_val["image_path"]
        .head(10)
        .to_string(index=False)
    )

    raise FileNotFoundError(
        "Validation CLAHE images are missing."
    )


# ============================================================
# CROSS-SPLIT LEAKAGE CHECK
# ============================================================

train_ids = set(train_df["image_id"])
val_ids = set(val_df["image_id"])

overlap = train_ids & val_ids

print("\nTrain ∩ Validation:", len(overlap))

if len(overlap) > 0:
    raise ValueError(
        "Image leakage detected between train and validation."
    )


# ============================================================
# CLASS WEIGHTS
# ============================================================

classes = np.unique(
    train_df["label"]
)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=train_df["label"]
)

class_weights = {
    int(cls): float(weight)
    for cls, weight in zip(classes, weights)
}


print("\n" + "=" * 75)
print("CLASS WEIGHTS")
print("=" * 75)

for cls, weight in class_weights.items():

    name = (
        "Non-melanoma"
        if cls == 0
        else "Melanoma"
    )

    print(
        f"{name} ({cls}): {weight:.4f}"
    )


# ============================================================
# IMAGE LOADER
# ============================================================

def load_image(path, label):

    image = tf.io.read_file(path)

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

    # ImageNet normalization
    image = tf.keras.applications.densenet.preprocess_input(
        image
    )

    return image, label


# ============================================================
# DATASET CREATION
# ============================================================

def create_dataset(
    dataframe,
    training=False
):

    paths = dataframe["image_path"].values
    labels = dataframe["label"].values

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


# ============================================================
# TRAINING-ONLY AUGMENTATION
# ============================================================

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

        tf.keras.layers.RandomContrast(
            0.10
        ),

    ],

    name="training_augmentation"
)


# ============================================================
# DENSENET121
# ============================================================

print("\n" + "=" * 75)
print("LOADING IMAGENET DENSENET121")
print("=" * 75)


base_model = tf.keras.applications.DenseNet121(

    include_top=False,

    weights="imagenet",

    input_shape=(
        224,
        224,
        3
    )
)

base_model.trainable = False


# ============================================================
# MODEL
# ============================================================

inputs = tf.keras.Input(
    shape=(
        224,
        224,
        3
    )
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


x = tf.keras.layers.Dropout(
    0.30
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
    inputs,
    outputs
)


# ============================================================
# COMPILE
# ============================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-3
    ),

    loss=tf.keras.losses.BinaryCrossentropy(),

    metrics=[

        tf.keras.metrics.BinaryAccuracy(
            name="accuracy"
        ),

        tf.keras.metrics.AUC(
            name="auc"
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

print("\n" + "=" * 75)
print("MODEL SUMMARY")
print("=" * 75)

model.summary()


# ============================================================
# CALLBACKS
# ============================================================

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


# ============================================================
# TRAINING
# ============================================================

print("\n" + "=" * 75)
print("STARTING FINAL CLAHE DENSENET121 TRAINING")
print("=" * 75)

history = model.fit(

    train_dataset,

    validation_data=val_dataset,

    epochs=EPOCHS,

    class_weight=class_weights,

    callbacks=callbacks

)


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

history_df = pd.DataFrame(
    history.history
)

history_df.insert(
    0,
    "epoch",
    range(
        1,
        len(history_df) + 1
    )
)

history_df.to_csv(
    HISTORY_PATH,
    index=False
)


# ============================================================
# BEST METRICS
# ============================================================

best_epoch = (
    history_df["val_auc"]
    .idxmax()
)

best_val_auc = (
    history_df.loc[
        best_epoch,
        "val_auc"
    ]
)

best_val_accuracy = (
    history_df.loc[
        best_epoch,
        "val_accuracy"
    ]
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 75)
print("TRAINING COMPLETED")
print("=" * 75)

print("\nBest model saved:")
print(MODEL_PATH)

print("\nTraining history saved:")
print(HISTORY_PATH)

print(
    f"\nBest validation AUC: "
    f"{best_val_auc:.4f}"
)

print(
    f"Best validation accuracy: "
    f"{best_val_accuracy:.4f}"
)

print("\n" + "=" * 75)
print("STATUS: PASS")
print("Final CLAHE DenseNet121 binary training completed.")
print("=" * 75)