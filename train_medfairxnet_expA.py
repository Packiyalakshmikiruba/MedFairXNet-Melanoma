import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    CSVLogger
)
from tensorflow.keras.optimizers import Adam

# ============================================================
# EXPERIMENT A
# MedFairXNet Controlled Training
#
# Architecture:
# EfficientNetV2B0 -> CBAM -> GAP -> BN -> Dropout
# -> Dense(256) -> Dropout -> Sigmoid
#
# Changes from previous experiment:
# 1. No dataset-source/fairness weighting
# 2. Standard Dropout during training
# 3. MC-Dropout capability is NOT forced during training
# 4. Test set is NOT used
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

BASE_DIR = r"C:\Users\HP\Desktop\Melonma"

TRAIN_CSV = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "final_binary_clahe_splits",
    "train_final_clahe.csv"
)

VAL_CSV = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "final_binary_clahe_splits",
    "validation_final_clahe.csv"
)

IMAGE_BASE = BASE_DIR

RESULT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "medfairxnet_expA"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "melanoma_medfairxnet_expA.keras"
)

HISTORY_PATH = os.path.join(
    RESULT_DIR,
    "training_history.csv"
)

LOG_PATH = os.path.join(
    RESULT_DIR,
    "training.log"
)

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

IMG_SIZE = 224
BATCH_SIZE = 32

# Start conservatively.
INITIAL_EPOCHS = 20
FINETUNE_EPOCHS = 30

INITIAL_LR = 1e-3
FINETUNE_LR = 1e-5

DROPOUT_RATE = 0.25

AUTOTUNE = tf.data.AUTOTUNE

print("=" * 70)
print("MedFairXNet Experiment A")
print("=" * 70)

print("Train CSV :", TRAIN_CSV)
print("Val CSV   :", VAL_CSV)
print("Model     :", MODEL_PATH)
print("Results   :", RESULT_DIR)
print("=" * 70)


# ============================================================
# Load CSV
# ============================================================

train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VAL_CSV)

print("\nTRAIN:")
print("Images:", len(train_df))
print(train_df["binary_label"].value_counts())

print("\nVALIDATION:")
print("Images:", len(val_df))
print(val_df["binary_label"].value_counts())


# ============================================================
# Image path resolver
# ============================================================

def resolve_path(path):
    """
    Resolve relative/absolute image paths.
    """

    if pd.isna(path):
        return None

    path = str(path)

    if os.path.isabs(path):
        return path

    return os.path.join(IMAGE_BASE, path)


def get_image_column(df):
    """
    Prefer image_path.
    Fall back to original_roi_path or original_image_path.
    """

    candidates = [
        "image_path",
        "original_roi_path",
        "original_image_path"
    ]

    for col in candidates:
        if col in df.columns:
            valid = df[col].notna().sum()

            if valid > 0:
                print("Using image column:", col)
                return col

    raise ValueError(
        "No usable image path column found in CSV."
    )


IMAGE_COLUMN = get_image_column(train_df)

train_paths = [
    resolve_path(x)
    for x in train_df[IMAGE_COLUMN].tolist()
]

val_paths = [
    resolve_path(x)
    for x in val_df[IMAGE_COLUMN].tolist()
]

train_labels = train_df["binary_label"].astype("float32").values
val_labels = val_df["binary_label"].astype("float32").values


# ============================================================
# Verify paths
# ============================================================

missing_train = [
    p for p in train_paths
    if p is None or not os.path.exists(p)
]

missing_val = [
    p for p in val_paths
    if p is None or not os.path.exists(p)
]

print("\nPath verification:")
print("Missing train:", len(missing_train))
print("Missing val  :", len(missing_val))

if missing_train:
    print("\nExample missing train path:")
    print(missing_train[:5])
    raise FileNotFoundError(
        "Training image paths are invalid."
    )

if missing_val:
    print("\nExample missing validation path:")
    print(missing_val[:5])
    raise FileNotFoundError(
        "Validation image paths are invalid."
    )


# ============================================================
# Dataset pipeline
# ============================================================

def load_image(path, label):
    image = tf.io.read_file(path)

    image = tf.image.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    image.set_shape([None, None, 3])

    image = tf.image.resize(
        image,
        [IMG_SIZE, IMG_SIZE],
        method=tf.image.ResizeMethod.BILINEAR
    )

    image = tf.cast(image, tf.float32) / 255.0

    return image, label


def make_dataset(paths, labels, training=False):

    ds = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    if training:
        ds = ds.shuffle(
            buffer_size=min(len(paths), 10000),
            seed=SEED,
            reshuffle_each_iteration=True
        )

    ds = ds.map(
        load_image,
        num_parallel_calls=AUTOTUNE
    )

    ds = ds.batch(
        BATCH_SIZE,
        drop_remainder=False
    )

    ds = ds.prefetch(AUTOTUNE)

    return ds


train_ds = make_dataset(
    train_paths,
    train_labels,
    training=True
)

val_ds = make_dataset(
    val_paths,
    val_labels,
    training=False
)


# ============================================================
# CBAM
# ============================================================

class ChannelAttention(layers.Layer):

    def __init__(self, ratio=8, **kwargs):
        super().__init__(**kwargs)

        self.ratio = ratio

    def build(self, input_shape):

        channels = int(input_shape[-1])

        hidden = max(channels // self.ratio, 1)

        self.shared_dense1 = layers.Dense(
            hidden,
            activation="relu",
            kernel_initializer="he_normal"
        )

        self.shared_dense2 = layers.Dense(
            channels,
            kernel_initializer="he_normal"
        )

        super().build(input_shape)

    def call(self, inputs):

        avg_pool = tf.reduce_mean(
            inputs,
            axis=[1, 2],
            keepdims=False
        )

        max_pool = tf.reduce_max(
            inputs,
            axis=[1, 2],
            keepdims=False
        )

        avg_out = self.shared_dense2(
            self.shared_dense1(avg_pool)
        )

        max_out = self.shared_dense2(
            self.shared_dense1(max_pool)
        )

        attention = tf.nn.sigmoid(
            avg_out + max_out
        )

        attention = tf.reshape(
            attention,
            [-1, 1, 1, tf.shape(attention)[-1]]
        )

        return inputs * attention


class SpatialAttention(layers.Layer):

    def __init__(self, kernel_size=7, **kwargs):
        super().__init__(**kwargs)

        self.conv = layers.Conv2D(
            1,
            kernel_size,
            padding="same",
            activation="sigmoid",
            kernel_initializer="he_normal"
        )

    def call(self, inputs):

        avg_pool = tf.reduce_mean(
            inputs,
            axis=-1,
            keepdims=True
        )

        max_pool = tf.reduce_max(
            inputs,
            axis=-1,
            keepdims=True
        )

        concat = tf.concat(
            [avg_pool, max_pool],
            axis=-1
        )

        attention = self.conv(concat)

        return inputs * attention


class CBAM(layers.Layer):

    def __init__(self, ratio=8, **kwargs):
        super().__init__(**kwargs)

        self.channel_attention = ChannelAttention(
            ratio=ratio
        )

        self.spatial_attention = SpatialAttention()

    def call(self, inputs):

        x = self.channel_attention(inputs)

        x = self.spatial_attention(x)

        return x


# ============================================================
# Build Model
# ============================================================

def build_model():

    backbone = EfficientNetV2B0(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )

    # Stage 1:
    # Freeze backbone for stable transfer learning.
    backbone.trainable = False

    inputs = layers.Input(
        shape=(IMG_SIZE, IMG_SIZE, 3),
        name="image"
    )

    x = backbone(
        inputs,
        training=False
    )

    x = CBAM(
        ratio=8,
        name="cbam"
    )(x)

    x = layers.GlobalAveragePooling2D(
        name="global_average_pooling"
    )(x)

    x = layers.BatchNormalization(
        name="feature_batch_norm"
    )(x)

    # IMPORTANT:
    # Standard Dropout.
    # It is active only during training.
    x = layers.Dropout(
        DROPOUT_RATE,
        name="dropout_1"
    )(x)

    x = layers.Dense(
        256,
        activation="relu",
        kernel_initializer="he_normal",
        name="dense_256"
    )(x)

    x = layers.BatchNormalization(
        name="dense_batch_norm"
    )(x)

    x = layers.Dropout(
        DROPOUT_RATE,
        name="dropout_2"
    )(x)

    outputs = layers.Dense(
        1,
        activation="sigmoid",
        name="melanoma_probability"
    )(x)

    model = Model(
        inputs,
        outputs,
        name="MedFairXNet_ExpA"
    )

    return model, backbone


model, backbone = build_model()

model.summary()


# ============================================================
# Metrics
# ============================================================

metrics = [
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
        name="auc",
        curve="ROC"
    ),

    tf.keras.metrics.AUC(
        name="pr_auc",
        curve="PR"
    )
]


# ============================================================
# Stage 1 compilation
# ============================================================

model.compile(
    optimizer=Adam(
        learning_rate=INITIAL_LR
    ),

    loss=tf.keras.losses.BinaryCrossentropy(),

    metrics=metrics
)


# ============================================================
# Callbacks
# ============================================================

checkpoint = ModelCheckpoint(
    MODEL_PATH,
    monitor="val_pr_auc",
    mode="max",
    save_best_only=True,
    verbose=1
)

early_stop = EarlyStopping(
    monitor="val_pr_auc",
    mode="max",
    patience=7,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_pr_auc",
    mode="max",
    factor=0.3,
    patience=3,
    min_lr=1e-7,
    verbose=1
)

csv_logger = CSVLogger(
    HISTORY_PATH,
    append=False
)


# ============================================================
# Stage 1 Training
# ============================================================

print("\n")
print("=" * 70)
print("STAGE 1: TRANSFER LEARNING")
print("=" * 70)

history_1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=INITIAL_EPOCHS,
    callbacks=[
        checkpoint,
        early_stop,
        reduce_lr,
        csv_logger
    ],
    verbose=1
)


# ============================================================
# Stage 2: Fine-tuning
# ============================================================

print("\n")
print("=" * 70)
print("STAGE 2: FINE-TUNING")
print("=" * 70)

# Unfreeze backbone
backbone.trainable = True

# Freeze BatchNorm layers inside pretrained backbone.
# This generally gives more stable fine-tuning.
for layer in backbone.layers:

    if isinstance(
        layer,
        layers.BatchNormalization
    ):
        layer.trainable = False


model.compile(
    optimizer=Adam(
        learning_rate=FINETUNE_LR
    ),

    loss=tf.keras.losses.BinaryCrossentropy(),

    metrics=metrics
)


# New callbacks for fine-tuning
checkpoint_ft = ModelCheckpoint(
    MODEL_PATH,
    monitor="val_pr_auc",
    mode="max",
    save_best_only=True,
    verbose=1
)

early_stop_ft = EarlyStopping(
    monitor="val_pr_auc",
    mode="max",
    patience=8,
    restore_best_weights=True,
    verbose=1
)

reduce_lr_ft = ReduceLROnPlateau(
    monitor="val_pr_auc",
    mode="max",
    factor=0.3,
    patience=3,
    min_lr=1e-8,
    verbose=1
)

csv_logger_ft = CSVLogger(
    HISTORY_PATH,
    append=True
)


history_2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=FINETUNE_EPOCHS,
    callbacks=[
        checkpoint_ft,
        early_stop_ft,
        reduce_lr_ft,
        csv_logger_ft
    ],
    verbose=1
)


# ============================================================
# Load best model
# ============================================================

print("\n")
print("=" * 70)
print("LOADING BEST EXPERIMENT-A MODEL")
print("=" * 70)

best_model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "CBAM": CBAM,
        "ChannelAttention": ChannelAttention,
        "SpatialAttention": SpatialAttention
    }
)


# ============================================================
# Final validation evaluation
# ============================================================

print("\n")
print("=" * 70)
print("FINAL VALIDATION EVALUATION")
print("=" * 70)

val_results = best_model.evaluate(
    val_ds,
    verbose=1,
    return_dict=True
)

for key, value in val_results.items():

    print(
        f"{key:15s}: {value:.6f}"
    )


# ============================================================
# Save validation metrics
# ============================================================

val_metrics_path = os.path.join(
    RESULT_DIR,
    "validation_evaluation.csv"
)

pd.DataFrame(
    [val_results]
).to_csv(
    val_metrics_path,
    index=False
)


# ============================================================
# Training completion
# ============================================================

print("\n")
print("=" * 70)
print("EXPERIMENT A COMPLETED")
print("=" * 70)

print("Best model:")
print(MODEL_PATH)

print("\nTraining history:")
print(HISTORY_PATH)

print("\nValidation metrics:")
print(val_metrics_path)

print("\nIMPORTANT:")
print("Test dataset was NOT used during training.")
print("Test dataset was NOT used for threshold tuning.")
print("=" * 70)
