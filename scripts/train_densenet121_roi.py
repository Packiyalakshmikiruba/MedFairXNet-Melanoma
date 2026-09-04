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
    / "roi_splits"
    / "train_roi.csv"
)

VAL_CSV = (
    BASE_DIR
    / "data"
    / "processed"
    / "roi_splits"
    / "validation_roi.csv"
)

MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = (
    MODEL_DIR
    / "skin_lesion_densenet121_roi.keras"
)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 8
EPOCHS = 10
NUM_CLASSES = 7
RANDOM_SEED = 42


# =========================================================
# REPRODUCIBILITY
# =========================================================

tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


print("=" * 70)
print("HAM10000 ROI DENSENET121 TRAINING")
print("=" * 70)

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
# LABEL MAPPING
# =========================================================

label_mapping = {
    "akiec": 0,
    "bcc": 1,
    "bkl": 2,
    "df": 3,
    "mel": 4,
    "nv": 5,
    "vasc": 6,
}

class_names = [
    "akiec",
    "bcc",
    "bkl",
    "df",
    "mel",
    "nv",
    "vasc",
]


train_df["label"] = train_df["dx"].map(label_mapping)
val_df["label"] = val_df["dx"].map(label_mapping)


if train_df["label"].isna().any():
    raise ValueError(
        "Unknown diagnosis found in training dataset."
    )

if val_df["label"].isna().any():
    raise ValueError(
        "Unknown diagnosis found in validation dataset."
    )


train_df["label"] = train_df["label"].astype(int)
val_df["label"] = val_df["label"].astype(int)


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
# VERIFY ROI IMAGES
# =========================================================

missing_train = train_df[
    ~train_df["image_path"]
    .apply(lambda x: Path(x).exists())
]

missing_val = val_df[
    ~val_df["image_path"]
    .apply(lambda x: Path(x).exists())
]


print("\nMissing training ROI images:", len(missing_train))
print("Missing validation ROI images:", len(missing_val))


if len(missing_train) > 0:
    print("\nFirst missing training images:")
    print(
        missing_train["image_path"]
        .head(10)
        .to_string(index=False)
    )

    raise FileNotFoundError(
        "Some training ROI images are missing."
    )


if len(missing_val) > 0:
    print("\nFirst missing validation images:")
    print(
        missing_val["image_path"]
        .head(10)
        .to_string(index=False)
    )

    raise FileNotFoundError(
        "Some validation ROI images are missing."
    )


# =========================================================
# CLASS DISTRIBUTION
# =========================================================

print("\nTraining class distribution:")

print(
    train_df["dx"]
    .value_counts()
    .sort_index()
)


# =========================================================
# CLASS WEIGHTS
# =========================================================

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
    for cls, weight in zip(
        classes,
        weights
    )
}


print("\nClass weights:")

for cls, weight in class_weights.items():

    print(
        f"{class_names[cls]} ({cls}): "
        f"{weight:.4f}"
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

def create_dataset(df, training=False):

    paths = df["image_path"].values
    labels = df["label"].values

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

    dataset = dataset.prefetch(1)

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
            "horizontal"
        ),

        tf.keras.layers.RandomRotation(
            0.05
        ),

        tf.keras.layers.RandomZoom(
            0.10
        ),

    ],
    name="data_augmentation"
)


# =========================================================
# DENSENET121
# =========================================================

print("\n" + "=" * 70)
print("LOADING IMAGENET DENSENET121")
print("=" * 70)


base_model = tf.keras.applications.DenseNet121(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3)
)

base_model.trainable = False


# =========================================================
# MODEL
# =========================================================

inputs = tf.keras.Input(
    shape=(224, 224, 3)
)

x = data_augmentation(inputs)

x = base_model(
    x,
    training=False
)

x = tf.keras.layers.GlobalAveragePooling2D()(x)

x = tf.keras.layers.Dropout(
    0.3
)(x)

outputs = tf.keras.layers.Dense(
    NUM_CLASSES,
    activation="softmax"
)(x)


model = tf.keras.Model(
    inputs,
    outputs
)


# =========================================================
# COMPILE
# =========================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]
)


# =========================================================
# MODEL SUMMARY
# =========================================================

print("\n" + "=" * 70)
print("DENSENET121 ROI MODEL SUMMARY")
print("=" * 70)

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

print("\n" + "=" * 70)
print("STARTING DENSENET121 ROI TRAINING")
print("=" * 70)


history = model.fit(

    train_dataset,

    validation_data=val_dataset,

    epochs=EPOCHS,

    class_weight=class_weights,

    callbacks=callbacks
)


# =========================================================
# COMPLETED
# =========================================================

print("\n" + "=" * 70)
print("DENSENET121 ROI TRAINING COMPLETED")
print("=" * 70)

print("\nBest model saved at:")

print(MODEL_PATH)

print("\nTraining completed successfully.")