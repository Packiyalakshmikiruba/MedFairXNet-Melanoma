"""
MELONMA - MEDFAIRXNET GRAD-CAM (Step 9 — XAI)
============================================================
Adapts your existing generate_gradcam_efficientnetv2_ph2.py pattern
to MedFairXNet. Two things are different and simpler here:

1. MedFairXNet was built with `input_tensor=inputs`, so the
   EfficientNetV2 backbone layers sit directly in the model's own
   layer graph (no nested backbone sub-model to search for) --
   we can build the Grad-CAM gradient model in one line.

2. Target layer: `cbam1_sa_out` -- the feature map AFTER the CBAM
   channel+spatial attention block, i.e. what the classifier head
   actually sees. This is the most meaningful layer to explain for
   MedFairXNet specifically, since CBAM is the "X" (explainability)
   component of the design.

Preprocessing here matches evaluate_medfairxnet_final_clahe.py's
load_image function exactly -- using a different preprocessing would
produce meaningless heatmaps/predictions.

Run from the project's scripts/ folder:
    python generate_gradcam_medfairxnet.py
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

from medfairxnet_model import (
    IMAGE_SIZE, MCDropout,
    _spatial_avg_pool, _spatial_max_pool, _spatial_pool_output_shape,
)


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "melanoma_medfairxnet_final_clahe.keras"

TEST_CSV = (
    BASE_DIR / "data" / "processed" / "final_binary_clahe_splits"
    / "test_final_clahe.csv"
)

OUTPUT_DIR = BASE_DIR / "results" / "gradcam_medfairxnet"

TARGET_LAYER_NAME = "cbam1_sa_out"  # post-CBAM attention feature map

MAX_IMAGES = 10
POSITIVE_CLASS = 1
THRESHOLD = 0.50


# ============================================================
# SETUP
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 90)
print("MELONMA - MEDFAIRXNET GRAD-CAM")
print("=" * 90)

print("\nModel:", MODEL_PATH)
print("Test CSV:", TEST_CSV)
print("Output directory:", OUTPUT_DIR)

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found:\n{MODEL_PATH}")
if not TEST_CSV.exists():
    raise FileNotFoundError(f"Test CSV not found:\n{TEST_CSV}")


# ============================================================
# LOAD TEST DATA
# ============================================================

df = pd.read_csv(TEST_CSV)

image_column = next(
    (c for c in ["image_path", "clahe_image_path", "filepath", "file_path", "path"]
     if c in df.columns), None,
)
if image_column is None:
    raise ValueError("Could not find an image path column.")
if "binary_label" not in df.columns:
    raise ValueError("binary_label column not found in test CSV.")

print(f"\nImage column: {image_column}")
print("Label column: binary_label")

BASE_DIR_RESOLVED = BASE_DIR


def make_absolute_path(path):
    p = Path(str(path))
    return str(p) if p.is_absolute() else str((BASE_DIR_RESOLVED / p).resolve())


df[image_column] = df[image_column].apply(make_absolute_path)

missing_images = df[~df[image_column].apply(lambda p: Path(p).exists())]
print(f"\nMissing images: {len(missing_images)}")
if len(missing_images) > 0:
    raise FileNotFoundError(f"{len(missing_images)} image files are missing.")


# ============================================================
# LOAD MODEL
# ============================================================

print("\n" + "=" * 90)
print("LOADING MEDFAIRXNET MODEL")
print("=" * 90)

model = keras.models.load_model(
    MODEL_PATH, compile=False, safe_mode=False,
    custom_objects={
        "MCDropout": MCDropout,
        "_spatial_avg_pool": _spatial_avg_pool,
        "_spatial_max_pool": _spatial_max_pool,
        "_spatial_pool_output_shape": _spatial_pool_output_shape,
    },
)
print("\nModel loaded successfully.")
print(f"Model output shape: {model.output_shape}")


# ============================================================
# GRAD-CAM TARGET LAYER
# ============================================================

try:
    target_layer = model.get_layer(TARGET_LAYER_NAME)
except ValueError:
    raise RuntimeError(
        f"Could not find layer '{TARGET_LAYER_NAME}' in the model. "
        "If you changed the CBAM layer names in medfairxnet_model.py, "
        "update TARGET_LAYER_NAME at the top of this script."
    )

print("\nSelected Grad-CAM layer:")
print(f"Layer name  : {target_layer.name}")
print(f"Layer type  : {target_layer.__class__.__name__}")
print(f"Output shape: {target_layer.output.shape}")


# ============================================================
# BUILD GRADIENT MODEL
# (single-graph model -- no nested-backbone reconstruction needed,
#  unlike the EfficientNetV2 PH2 script)
# ============================================================

grad_model = keras.models.Model(
    inputs=model.inputs,
    outputs=[target_layer.output, model.output],
)
print("\nGradient model created successfully.")


# ============================================================
# PREPROCESS IMAGE
# (must exactly match evaluate_medfairxnet_final_clahe.py's load_image)
# ============================================================

def load_image_for_model(image_path):
    image_bytes = tf.io.read_file(image_path)
    image = tf.image.decode_image(image_bytes, channels=3, expand_animations=False)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    mean = tf.constant([0.485, 0.456, 0.406])
    std = tf.constant([0.229, 0.224, 0.225])
    image = (image - mean) / std
    return tf.expand_dims(image, axis=0)


def load_image_for_display(image_path):
    """Raw 0-255 image for the 'original' panel (not the normalized tensor)."""
    image_bytes = tf.io.read_file(image_path)
    image = tf.image.decode_image(image_bytes, channels=3, expand_animations=False)
    image = tf.image.resize(image, IMAGE_SIZE)
    return tf.cast(image, tf.float32).numpy()


# ============================================================
# GRAD-CAM
# ============================================================

def generate_gradcam(image_tensor):
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_tensor, training=False)
        tape.watch(conv_outputs)

        if predictions.shape[-1] == 1:
            positive_score = predictions[:, 0]
        else:
            positive_score = predictions[:, POSITIVE_CLASS]

    gradients = tape.gradient(positive_score, conv_outputs)
    if gradients is None:
        raise RuntimeError("Gradients are None. Unable to calculate Grad-CAM.")

    pooled_gradients = tf.reduce_mean(gradients, axis=(1, 2))

    conv_outputs = conv_outputs[0]
    pooled_gradients = pooled_gradients[0]

    heatmap = tf.reduce_sum(conv_outputs * pooled_gradients, axis=-1)
    heatmap = tf.maximum(heatmap, 0)

    max_value = tf.reduce_max(heatmap)
    heatmap = tf.where(max_value > 0, heatmap / max_value, heatmap)

    return heatmap.numpy(), predictions.numpy()


# ============================================================
# OVERLAY
# ============================================================

def create_overlay(original_image, heatmap):
    original_uint8 = np.clip(original_image, 0, 255).astype(np.uint8)

    heatmap_resized = tf.image.resize(
        heatmap[..., np.newaxis], IMAGE_SIZE
    ).numpy().squeeze()
    heatmap_resized = np.clip(heatmap_resized, 0, 1)

    cmap = plt.get_cmap("jet")
    heatmap_color = cmap(heatmap_resized)[:, :, :3]
    heatmap_color = (heatmap_color * 255).astype(np.uint8)

    overlay = 0.60 * original_uint8 + 0.40 * heatmap_color
    return np.clip(overlay, 0, 255).astype(np.uint8)


# ============================================================
# SELECT IMAGES (mix of melanoma + non-melanoma)
# ============================================================

print("\n" + "=" * 90)
print("SELECTING TEST IMAGES")
print("=" * 90)

melanoma_df = df[df["binary_label"] == 1]
non_melanoma_df = df[df["binary_label"] == 0]

selected_parts = []
if len(melanoma_df) > 0:
    selected_parts.append(melanoma_df.head(MAX_IMAGES // 2))
if len(non_melanoma_df) > 0:
    selected_parts.append(non_melanoma_df.head(MAX_IMAGES - MAX_IMAGES // 2))

selected_df = (
    pd.concat(selected_parts, ignore_index=True) if selected_parts else df.head(MAX_IMAGES)
).head(MAX_IMAGES)

print(f"\nSelected {len(selected_df)} images for Grad-CAM.")


# ============================================================
# GENERATE
# ============================================================

print("\n" + "=" * 90)
print("GENERATING GRAD-CAM")
print("=" * 90)

results = []
successful = 0
failed = 0

for idx, row in selected_df.iterrows():
    image_path = str(row[image_column])
    image_id = os.path.basename(image_path)
    true_label = int(row["binary_label"])

    print(f"\nProcessing {idx + 1}/{len(selected_df)}: {image_id}")

    try:
        model_input = load_image_for_model(image_path)
        heatmap, prediction = generate_gradcam(model_input)

        probability = float(prediction.reshape(-1)[0])
        predicted_label = int(probability >= THRESHOLD)

        original = load_image_for_display(image_path)
        overlay = create_overlay(original, heatmap)

        safe_id = os.path.splitext(image_id)[0]
        original_output = OUTPUT_DIR / f"{safe_id}_original.png"
        heatmap_output = OUTPUT_DIR / f"{safe_id}_heatmap.png"
        overlay_output = OUTPUT_DIR / f"{safe_id}_gradcam.png"

        plt.figure(figsize=(6, 6))
        plt.imshow(original.astype(np.uint8))
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.savefig(original_output, dpi=150, bbox_inches="tight", pad_inches=0)
        plt.close()

        plt.figure(figsize=(6, 6))
        plt.imshow(heatmap, cmap="jet")
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.savefig(heatmap_output, dpi=150, bbox_inches="tight", pad_inches=0)
        plt.close()

        plt.figure(figsize=(6, 6))
        plt.imshow(overlay)
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.savefig(overlay_output, dpi=150, bbox_inches="tight", pad_inches=0)
        plt.close()

        results.append({
            "image_id": image_id,
            "image_path": image_path,
            "true_label": true_label,
            "true_class": "Melanoma" if true_label == 1 else "Non-melanoma",
            "prediction_probability": probability,
            "predicted_label": predicted_label,
            "predicted_class": "Melanoma" if predicted_label == 1 else "Non-melanoma",
            "threshold": THRESHOLD,
            "gradcam_layer": target_layer.name,
            "original_image": str(original_output),
            "heatmap_image": str(heatmap_output),
            "gradcam_image": str(overlay_output),
        })

        successful += 1
        print(f"True class       : {'Melanoma' if true_label == 1 else 'Non-melanoma'}")
        print(f"Prediction score : {probability:.6f}")
        print(f"Predicted class  : {'Melanoma' if predicted_label == 1 else 'Non-melanoma'}")
        print("Grad-CAM saved successfully.")

    except Exception as e:
        failed += 1
        print(f"ERROR processing {image_id}: {e}")


# ============================================================
# SAVE CSV
# ============================================================

print("\n" + "=" * 90)
print("SAVING GRAD-CAM RESULTS")
print("=" * 90)

results_csv = OUTPUT_DIR / "gradcam_results.csv"

if not results:
    raise RuntimeError("No Grad-CAM results were generated.")

pd.DataFrame(results).to_csv(results_csv, index=False)
print(f"\nResults CSV:\n{results_csv}")


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 90)
print("GRAD-CAM SUMMARY")
print("=" * 90)
print(f"\nImages requested : {len(selected_df)}")
print(f"Successful       : {successful}")
print(f"Failed           : {failed}")
print(f"\nGrad-CAM layer   : {target_layer.name}")
print(f"Layer output     : {target_layer.output.shape}")
print(f"\nOutput directory:\n{OUTPUT_DIR}")

print("\n" + "=" * 90)
print("STATUS: PASS")
print("MedFairXNet Grad-CAM completed successfully.")
print("=" * 90)