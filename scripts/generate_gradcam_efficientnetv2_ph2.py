import os
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = r"C:\Users\HP\Desktop\Melonma"

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "melanoma_efficientnetv2_ph2_clahe.keras"
)

TEST_CSV = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "PH2",
    "splits",
    "test_ph2_clahe.csv"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "gradcam_efficientnetv2_ph2"
)

IMAGE_SIZE = (224, 224)

# Generate Grad-CAM for maximum 10 images
MAX_IMAGES = 10

# Positive class = melanoma
POSITIVE_CLASS = 1

# Prediction threshold
THRESHOLD = 0.50


# ============================================================
# SETUP
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 90)
print("MELONMA - EFFICIENTNETV2 PH2 CLAHE GRAD-CAM")
print("=" * 90)

print("\nModel:")
print(MODEL_PATH)

print("\nTest CSV:")
print(TEST_CSV)

print("\nOutput directory:")
print(OUTPUT_DIR)


# ============================================================
# CHECK FILES
# ============================================================

if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(
        f"\nModel not found:\n{MODEL_PATH}"
    )

if not os.path.isfile(TEST_CSV):
    raise FileNotFoundError(
        f"\nTest CSV not found:\n{TEST_CSV}"
    )


# ============================================================
# LOAD TEST DATA
# ============================================================

print("\n" + "=" * 90)
print("LOADING TEST DATA")
print("=" * 90)

df = pd.read_csv(TEST_CSV)

print(f"\nTest records: {len(df)}")
print(f"Columns: {list(df.columns)}")


# ------------------------------------------------------------
# Find image column
# ------------------------------------------------------------

possible_image_columns = [
    "image_path",
    "clahe_image_path",
    "filepath",
    "file_path",
    "path"
]

image_column = None

for col in possible_image_columns:
    if col in df.columns:
        image_column = col
        break

if image_column is None:
    raise ValueError(
        "Could not find image path column."
    )

# Prefer image_path because model evaluation used it
if "image_path" in df.columns:
    image_column = "image_path"

if "binary_label" not in df.columns:
    raise ValueError(
        "binary_label column not found in test CSV."
    )

print(f"\nImage column: {image_column}")
print("Label column: binary_label")


# ============================================================
# CHECK IMAGE FILES
# ============================================================

missing_images = []

for path in df[image_column].astype(str):
    if not os.path.isfile(path):
        missing_images.append(path)

print(f"\nMissing images: {len(missing_images)}")

if missing_images:
    print("\nFirst missing images:")
    for path in missing_images[:10]:
        print(path)

    raise FileNotFoundError(
        f"\n{len(missing_images)} image files are missing."
    )


# ============================================================
# LOAD MODEL
# ============================================================

print("\n" + "=" * 90)
print("LOADING EFFICIENTNETV2 MODEL")
print("=" * 90)

model = keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("\nModel loaded successfully.")
print(f"Model output shape: {model.output_shape}")


# ============================================================
# MODEL ARCHITECTURE
# ============================================================

print("\n" + "=" * 90)
print("MODEL ARCHITECTURE")
print("=" * 90)

print("\nTop-level model layers:")

for i, layer in enumerate(model.layers):
    print(
        f"{i:03d} | "
        f"{layer.name:40s} | "
        f"{layer.__class__.__name__}"
    )


# ============================================================
# FIND EFFICIENTNET BACKBONE
# ============================================================

print("\n" + "=" * 90)
print("SEARCHING FOR EFFICIENTNETV2 BACKBONE")
print("=" * 90)

backbone = None
backbone_index = None

for i, layer in enumerate(model.layers):

    layer_name = layer.name.lower()

    if (
        "efficientnet" in layer_name
        or "efficientnetv2" in layer_name
    ):
        backbone = layer
        backbone_index = i
        break


if backbone is None:

    # Fallback: search nested models
    for i, layer in enumerate(model.layers):

        if isinstance(layer, keras.Model):

            print(
                f"Nested model candidate: "
                f"{i} -> {layer.name}"
            )

            if len(layer.layers) > 20:
                backbone = layer
                backbone_index = i
                break


if backbone is None:
    raise RuntimeError(
        "\nCould not identify EfficientNetV2 backbone."
    )


print("\nSelected backbone:")
print(f"Name : {backbone.name}")
print(f"Index: {backbone_index}")


# ============================================================
# FIND LAST CONVOLUTIONAL LAYER INSIDE BACKBONE
# ============================================================

print("\n" + "=" * 90)
print("SEARCHING FOR GRAD-CAM LAYER")
print("=" * 90)

target_layer = None

# First preference: known EfficientNetV2 final feature layer
preferred_names = [
    "top_activation",
    "top_conv",
    "block6e_add",
    "block6d_add",
    "block6c_add",
    "block6b_add",
    "block6a_add",
]

for preferred in preferred_names:

    for layer in reversed(backbone.layers):

        if layer.name.lower() == preferred.lower():

            try:
                shape = layer.output.shape

                if len(shape) == 4:
                    target_layer = layer
                    break

            except Exception:
                pass

    if target_layer is not None:
        break


# ------------------------------------------------------------
# Fallback: find last 4D layer
# ------------------------------------------------------------

if target_layer is None:

    for layer in reversed(backbone.layers):

        try:

            output_shape = layer.output.shape

            if (
                len(output_shape) == 4
                and output_shape[1] is not None
                and output_shape[2] is not None
            ):

                target_layer = layer
                break

        except Exception:
            continue


if target_layer is None:
    raise RuntimeError(
        "\nCould not find a suitable 4D convolutional layer "
        "inside EfficientNetV2."
    )


print("\nSelected Grad-CAM layer:")
print(f"Layer name : {target_layer.name}")
print(f"Layer type : {target_layer.__class__.__name__}")
print(f"Output shape: {target_layer.output.shape}")


# ============================================================
# CREATE BACKBONE GRADIENT MODEL
# ============================================================

print("\n" + "=" * 90)
print("BUILDING GRADIENT MODEL")
print("=" * 90)

try:

    backbone_grad_model = keras.models.Model(
        inputs=backbone.input,
        outputs=[
            target_layer.output,
            backbone.output
        ]
    )

    print("\nBackbone gradient model created successfully.")

except Exception as e:

    raise RuntimeError(
        "\nFailed to create backbone gradient model.\n"
        f"Error: {e}"
    )


# ============================================================
# PREPROCESS IMAGE
# ============================================================

def load_image(image_path):

    image_bytes = tf.io.read_file(image_path)

    image = tf.image.decode_image(
        image_bytes,
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

    image = tf.expand_dims(
        image,
        axis=0
    )

    return image


# ============================================================
# FORWARD THROUGH HEAD
# ============================================================

def forward_through_head(backbone_output):

    x = backbone_output

    # Layers after EfficientNetV2 backbone
    for i in range(backbone_index + 1, len(model.layers)):

        layer = model.layers[i]

        x = layer(
            x,
            training=False
        )

    return x


# ============================================================
# GRAD-CAM
# ============================================================

def generate_gradcam(image_tensor):

    with tf.GradientTape() as tape:

        conv_outputs, backbone_output = backbone_grad_model(
            image_tensor,
            training=False
        )

        tape.watch(conv_outputs)

        predictions = forward_through_head(
            backbone_output
        )

        # Binary sigmoid model
        if predictions.shape[-1] == 1:

            positive_score = predictions[:, 0]

        else:

            positive_score = predictions[:, POSITIVE_CLASS]

    gradients = tape.gradient(
        positive_score,
        conv_outputs
    )

    if gradients is None:
        raise RuntimeError(
            "Gradients are None. "
            "Unable to calculate Grad-CAM."
        )

    # Global average pooling over gradients
    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(1, 2)
    )

    # Remove batch dimension
    conv_outputs = conv_outputs[0]
    pooled_gradients = pooled_gradients[0]

    # Weighted feature maps
    heatmap = tf.reduce_sum(
        conv_outputs * pooled_gradients,
        axis=-1
    )

    # ReLU
    heatmap = tf.maximum(
        heatmap,
        0
    )

    # Normalize
    max_value = tf.reduce_max(
        heatmap
    )

    heatmap = tf.where(
        max_value > 0,
        heatmap / max_value,
        heatmap
    )

    return heatmap.numpy(), predictions.numpy()


# ============================================================
# OVERLAY FUNCTION
# ============================================================

def create_overlay(original_image, heatmap):

    original_uint8 = np.clip(
        original_image,
        0,
        255
    ).astype(np.uint8)

    # Resize heatmap
    heatmap_resized = tf.image.resize(
        heatmap[..., np.newaxis],
        IMAGE_SIZE
    ).numpy().squeeze()

    heatmap_resized = np.clip(
        heatmap_resized,
        0,
        1
    )

    # Matplotlib colormap
    cmap = plt.get_cmap("jet")

    heatmap_color = cmap(
        heatmap_resized
    )[:, :, :3]

    heatmap_color = (
        heatmap_color * 255
    ).astype(np.uint8)

    # Blend
    overlay = (
        0.60 * original_uint8
        + 0.40 * heatmap_color
    )

    overlay = np.clip(
        overlay,
        0,
        255
    ).astype(np.uint8)

    return overlay


# ============================================================
# SELECT IMAGES
# ============================================================

print("\n" + "=" * 90)
print("SELECTING TEST IMAGES")
print("=" * 90)

# Try to include melanoma and non-melanoma examples
melanoma_df = df[
    df["binary_label"] == 1
]

non_melanoma_df = df[
    df["binary_label"] == 0
]

selected_parts = []

if len(melanoma_df) > 0:
    selected_parts.append(
        melanoma_df.head(MAX_IMAGES // 2)
    )

if len(non_melanoma_df) > 0:
    selected_parts.append(
        non_melanoma_df.head(MAX_IMAGES - MAX_IMAGES // 2)
    )

if selected_parts:

    selected_df = pd.concat(
        selected_parts,
        ignore_index=True
    )

else:

    selected_df = df.head(
        MAX_IMAGES
    )


selected_df = selected_df.head(
    MAX_IMAGES
)

print(
    f"\nSelected {len(selected_df)} images "
    f"for Grad-CAM."
)


# ============================================================
# GENERATE RESULTS
# ============================================================

print("\n" + "=" * 90)
print("GENERATING GRAD-CAM")
print("=" * 90)

results = []

successful = 0
failed = 0


for idx, row in selected_df.iterrows():

    image_path = str(
        row[image_column]
    )

    image_id = os.path.basename(
        image_path
    )

    true_label = int(
        row["binary_label"]
    )

    print(
        f"\nProcessing "
        f"{idx + 1}/{len(selected_df)}: "
        f"{image_id}"
    )

    try:

        # ----------------------------------------------------
        # Load image
        # ----------------------------------------------------

        image_tensor = load_image(
            image_path
        )

        # ----------------------------------------------------
        # Generate Grad-CAM
        # ----------------------------------------------------

        heatmap, prediction = generate_gradcam(
            image_tensor
        )

        # ----------------------------------------------------
        # Prediction probability
        # ----------------------------------------------------

        probability = float(
            prediction.reshape(-1)[0]
        )

        predicted_label = int(
            probability >= THRESHOLD
        )

        # ----------------------------------------------------
        # Load original image for visualization
        # ----------------------------------------------------

        original = image_tensor[0].numpy()

        # ----------------------------------------------------
        # Create overlay
        # ----------------------------------------------------

        overlay = create_overlay(
            original,
            heatmap
        )

        # ----------------------------------------------------
        # Save files
        # ----------------------------------------------------

        safe_id = os.path.splitext(
            image_id
        )[0]

        original_output = os.path.join(
            OUTPUT_DIR,
            f"{safe_id}_original.png"
        )

        heatmap_output = os.path.join(
            OUTPUT_DIR,
            f"{safe_id}_heatmap.png"
        )

        overlay_output = os.path.join(
            OUTPUT_DIR,
            f"{safe_id}_gradcam.png"
        )

        # Original
        plt.figure(
            figsize=(6, 6)
        )

        plt.imshow(
            original.astype(np.uint8)
        )

        plt.axis("off")
        plt.tight_layout(pad=0)

        plt.savefig(
            original_output,
            dpi=150,
            bbox_inches="tight",
            pad_inches=0
        )

        plt.close()

        # Heatmap
        plt.figure(
            figsize=(6, 6)
        )

        plt.imshow(
            heatmap,
            cmap="jet"
        )

        plt.axis("off")
        plt.tight_layout(pad=0)

        plt.savefig(
            heatmap_output,
            dpi=150,
            bbox_inches="tight",
            pad_inches=0
        )

        plt.close()

        # Overlay
        plt.figure(
            figsize=(6, 6)
        )

        plt.imshow(
            overlay
        )

        plt.axis("off")
        plt.tight_layout(pad=0)

        plt.savefig(
            overlay_output,
            dpi=150,
            bbox_inches="tight",
            pad_inches=0
        )

        plt.close()

        # ----------------------------------------------------
        # Save result
        # ----------------------------------------------------

        results.append({
            "image_id": image_id,
            "image_path": image_path,
            "true_label": true_label,
            "true_class": (
                "Melanoma"
                if true_label == 1
                else "Non-melanoma"
            ),
            "prediction_probability": probability,
            "predicted_label": predicted_label,
            "predicted_class": (
                "Melanoma"
                if predicted_label == 1
                else "Non-melanoma"
            ),
            "threshold": THRESHOLD,
            "gradcam_layer": target_layer.name,
            "original_image": original_output,
            "heatmap_image": heatmap_output,
            "gradcam_image": overlay_output,
        })

        successful += 1

        print(
            f"True class       : "
            f"{'Melanoma' if true_label == 1 else 'Non-melanoma'}"
        )

        print(
            f"Prediction score : "
            f"{probability:.6f}"
        )

        print(
            f"Predicted class  : "
            f"{'Melanoma' if predicted_label == 1 else 'Non-melanoma'}"
        )

        print(
            "Grad-CAM saved successfully."
        )

    except Exception as e:

        failed += 1

        print(
            f"ERROR processing {image_id}: {e}"
        )


# ============================================================
# SAVE CSV
# ============================================================

print("\n" + "=" * 90)
print("SAVING GRAD-CAM RESULTS")
print("=" * 90)

results_csv = os.path.join(
    OUTPUT_DIR,
    "gradcam_results.csv"
)

if results:

    results_df = pd.DataFrame(
        results
    )

    results_df.to_csv(
        results_csv,
        index=False
    )

    print(
        f"\nResults CSV:\n{results_csv}"
    )

else:

    raise RuntimeError(
        "\nNo Grad-CAM results were generated."
    )


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 90)
print("GRAD-CAM SUMMARY")
print("=" * 90)

print(
    f"\nImages requested : {len(selected_df)}"
)

print(
    f"Successful       : {successful}"
)

print(
    f"Failed           : {failed}"
)

print(
    f"\nGrad-CAM layer   : {target_layer.name}"
)

print(
    f"Layer output     : {target_layer.output.shape}"
)

print(
    f"\nOutput directory:"
)

print(
    OUTPUT_DIR
)

print("\nGenerated files:")

for file_name in sorted(
    os.listdir(OUTPUT_DIR)
):

    print(
        f"  - {file_name}"
    )


print("\n" + "=" * 90)
print("STATUS: PASS")
print("EfficientNetV2 PH2 CLAHE Grad-CAM completed successfully.")
print("=" * 90)