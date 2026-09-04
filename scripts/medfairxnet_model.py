"""
MELONMA - MedFairXNet MODEL ARCHITECTURE
============================================================
Proposed model = EfficientNetV2B0 backbone
                  + CBAM (Channel + Spatial) attention block
                  + Monte-Carlo Dropout head (uncertainty)
                  + dataset-source-aware class weighting hook (fairness)

Why this design (so it is defensible in the paper, not decoration):
- Backbone: EfficientNetV2 was your strongest lightweight baseline
  (ROC-AUC ~0.76), so it is a fair, comparable starting point.
- CBAM attention: lets the model re-weight lesion-relevant channels
  and spatial regions -> produces cleaner Grad-CAM maps for the XAI
  section (this is the "X" in MedFairXNet).
- MC-Dropout head: keeping dropout active at inference time lets you
  run N stochastic forward passes and report predictive mean +
  variance/entropy as an uncertainty/calibration score, without
  needing a second model or extra labels.
- Fairness hook: training script accepts a `subgroup` column
  (e.g. dataset_source: ISIC2018 / HAM10000 / PH2) and can compute
  per-subgroup sample weights so no single source dominates the
  gradient -> supports the fairness section (equal opportunity /
  per-subgroup TPR-FPR gap) later.

This file only DEFINES the model. It does not train or invent
metrics. Train with train_medfairxnet_final_clahe.py, evaluate with
evaluate_medfairxnet_final_clahe.py, using your real CLAHE splits.
"""

import tensorflow as tf
from tensorflow.keras import layers, models


IMAGE_SIZE = (224, 224)
CHANNELS = 3


# ============================================================
# CBAM: CONVOLUTIONAL BLOCK ATTENTION MODULE
# ============================================================

def channel_attention(input_feature, ratio=8, name="ca"):
    channel = input_feature.shape[-1]

    shared_dense_one = layers.Dense(
        channel // ratio, activation="relu",
        kernel_initializer="he_normal", use_bias=True,
        name=f"{name}_dense1",
    )
    shared_dense_two = layers.Dense(
        channel, kernel_initializer="he_normal", use_bias=True,
        name=f"{name}_dense2",
    )

    avg_pool = layers.GlobalAveragePooling2D()(input_feature)
    avg_pool = shared_dense_one(avg_pool)
    avg_pool = shared_dense_two(avg_pool)

    max_pool = layers.GlobalMaxPooling2D()(input_feature)
    max_pool = shared_dense_one(max_pool)
    max_pool = shared_dense_two(max_pool)

    attention = layers.Add()([avg_pool, max_pool])
    attention = layers.Activation("sigmoid")(attention)
    attention = layers.Reshape((1, 1, channel))(attention)

    return layers.Multiply(name=f"{name}_out")([input_feature, attention])


def _spatial_avg_pool(t):
    return tf.reduce_mean(t, axis=-1, keepdims=True)


def _spatial_max_pool(t):
    return tf.reduce_max(t, axis=-1, keepdims=True)


def _spatial_pool_output_shape(input_shape):
    return input_shape[:-1] + (1,)


def spatial_attention(input_feature, kernel_size=7, name="sa"):
    avg_pool = layers.Lambda(
        _spatial_avg_pool,
        output_shape=_spatial_pool_output_shape,
        name=f"{name}_avgpool",
    )(input_feature)
    max_pool = layers.Lambda(
        _spatial_max_pool,
        output_shape=_spatial_pool_output_shape,
        name=f"{name}_maxpool",
    )(input_feature)
    concat = layers.Concatenate(axis=-1)([avg_pool, max_pool])

    attention = layers.Conv2D(
        filters=1, kernel_size=kernel_size, padding="same",
        activation="sigmoid", kernel_initializer="he_normal",
        use_bias=False, name=f"{name}_conv",
    )(concat)

    return layers.Multiply(name=f"{name}_out")([input_feature, attention])


def cbam_block(input_feature, ratio=8, kernel_size=7, name="cbam"):
    x = channel_attention(input_feature, ratio=ratio, name=f"{name}_ca")
    x = spatial_attention(x, kernel_size=kernel_size, name=f"{name}_sa")
    return x


# ============================================================
# MONTE-CARLO DROPOUT LAYER
# (regular Dropout only drops during training; this version stays
#  active during inference too, so we can sample N stochastic
#  predictions per image for uncertainty estimation)
# ============================================================

class MCDropout(layers.Dropout):
    def call(self, inputs, training=None):
        return super().call(inputs, training=True)


# ============================================================
# BUILD MEDFAIRXNET
# ============================================================

def build_medfairxnet(
    image_size=IMAGE_SIZE,
    dropout_rate=0.35,
    l2_reg=1e-5,
    freeze_backbone_until=None,
):
    """
    Returns a compiled-ready (uncompiled) tf.keras.Model.

    image_size            : (H, W)
    dropout_rate           : MC-Dropout rate in the classifier head
    l2_reg                 : L2 weight decay on the final dense layer
    freeze_backbone_until   : None = fully trainable backbone (fine-tune),
                              or int = freeze the first N backbone layers
                              and fine-tune the rest (recommended first
                              run: freeze_backbone_until=200 for a fast,
                              stable training run, then optionally
                              unfreeze for a low-LR fine-tune pass).
    """

    inputs = layers.Input(shape=(*image_size, CHANNELS), name="image_input")

    backbone = tf.keras.applications.EfficientNetV2B0(
        include_top=False,
        weights="imagenet",
        input_tensor=inputs,
        pooling=None,
    )

    if freeze_backbone_until is not None:
        for layer in backbone.layers[:freeze_backbone_until]:
            layer.trainable = False

    x = backbone.output                      # (7, 7, C) feature map
    x = cbam_block(x, ratio=8, kernel_size=7, name="cbam1")

    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.BatchNormalization(name="bn_head")(x)

    x = MCDropout(dropout_rate, name="mc_dropout_1")(x)
    x = layers.Dense(
        256, activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
        name="dense_256",
    )(x)

    x = MCDropout(dropout_rate, name="mc_dropout_2")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="melanoma_prob")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="MedFairXNet")
    return model


# ============================================================
# MC-DROPOUT UNCERTAINTY INFERENCE HELPER
# ============================================================

def mc_dropout_predict(model, x_batch, n_samples=20):
    """
    Runs n_samples stochastic forward passes on the same input batch and
    returns:
        mean_prob      : predictive mean probability (use this as the
                          final prediction, same as a normal predict())
        std_prob       : predictive std-dev (epistemic uncertainty proxy)
        predictive_entropy : entropy of the mean prediction (calibration /
                          confidence proxy, higher = more uncertain)
    Shapes: all arrays of length len(x_batch).

    IMPORTANT: the model is called with training=False here. The custom
    MCDropout layer ignores the incoming training flag and always applies
    dropout internally, so stochastic sampling still happens -- but
    training=False keeps BatchNormalization using its learned running
    statistics instead of noisy per-batch statistics. Calling with
    training=True here would silently corrupt predictions, since it would
    make every BatchNorm layer use small-batch statistics instead of the
    stats learned over the whole training set.
    """
    import numpy as np

    samples = np.stack(
        [model(x_batch, training=False).numpy().reshape(-1) for _ in range(n_samples)],
        axis=0,
    )  # (n_samples, batch)

    mean_prob = samples.mean(axis=0)
    std_prob = samples.std(axis=0)

    eps = 1e-8
    predictive_entropy = -(
        mean_prob * np.log(mean_prob + eps)
        + (1 - mean_prob) * np.log(1 - mean_prob + eps)
    )

    return mean_prob, std_prob, predictive_entropy


if __name__ == "__main__":
    m = build_medfairxnet()
    m.summary()
    print("\nTotal params:", f"{m.count_params():,}")