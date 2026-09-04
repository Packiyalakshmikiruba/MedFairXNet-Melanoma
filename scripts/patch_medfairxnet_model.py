"""
MELONMA - PATCH MEDFAIRXNET SAVED MODEL (Lambda output_shape fix)
============================================================
One-time fix. Your MedFairXNet model already finished training and
was saved to melanoma_medfairxnet_final_clahe.keras. That file's
internal config is just missing an `output_shape` hint on two Lambda
layers (cbam1_sa_avgpool, cbam1_sa_maxpool) that Keras needs when
reloading the model on this machine/version. This script edits that
one field directly inside the saved .keras archive -- NO retraining
needed, and the trained weights are untouched.

A .keras file is actually a zip archive containing config.json +
weights. This script:
  1. Makes a timestamped backup copy of the original .keras file.
  2. Unzips it, patches config.json, re-zips it back to the same path.

Run once from the scripts/ folder:
    python patch_medfairxnet_model.py

Then run evaluate_medfairxnet_final_clahe.py as normal.
"""

import json
import shutil
import zipfile
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "melanoma_medfairxnet_final_clahe.keras"

# Spatial size of the CBAM feature map for 224x224 input with the
# EfficientNetV2B0 backbone (confirmed from your model.summary() output:
# cbam1_sa_avgpool / cbam1_sa_maxpool -> (None, 7, 7, 1)).
PATCH_OUTPUT_SHAPE = [7, 7, 1]
TARGET_LAYER_NAMES = {"cbam1_sa_avgpool", "cbam1_sa_maxpool"}

print("=" * 70)
print("PATCHING MEDFAIRXNET SAVED MODEL")
print("=" * 70)
print("\nModel file:", MODEL_PATH)

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found:\n{MODEL_PATH}")

# ------------------------------------------------------------------
# Backup
# ------------------------------------------------------------------
backup_path = MODEL_PATH.with_suffix(".keras.bak")
if not backup_path.exists():
    shutil.copy2(MODEL_PATH, backup_path)
    print("Backup created:", backup_path)
else:
    print("Backup already exists (not overwritten):", backup_path)

# ------------------------------------------------------------------
# Unzip to a temp working folder
# ------------------------------------------------------------------
work_dir = MODEL_PATH.parent / "_medfairxnet_patch_tmp"
if work_dir.exists():
    shutil.rmtree(work_dir)
work_dir.mkdir()

with zipfile.ZipFile(MODEL_PATH, "r") as zf:
    zf.extractall(work_dir)
    original_names = zf.namelist()

config_path = work_dir / "config.json"
if not config_path.exists():
    raise FileNotFoundError(f"config.json not found inside the .keras archive: {config_path}")

with open(config_path, "r", encoding="utf-8") as f:
    config = json.load(f)


# ------------------------------------------------------------------
# Walk the config tree and patch matching Lambda layers
# ------------------------------------------------------------------
patched_count = 0


def patch_layers(layer_list):
    global patched_count
    for layer in layer_list:
        layer_config = layer.get("config", {})
        layer_name = layer_config.get("name")
        if layer.get("class_name") == "Lambda" and layer_name in TARGET_LAYER_NAMES:
            layer_config["output_shape"] = PATCH_OUTPUT_SHAPE
            patched_count += 1
            print(f"Patched layer: {layer_name} -> output_shape={PATCH_OUTPUT_SHAPE}")


# Model config layers usually live at config["config"]["layers"]
model_layers = config.get("config", {}).get("layers", [])
patch_layers(model_layers)

if patched_count == 0:
    raise RuntimeError(
        "No matching Lambda layers were found to patch. "
        "The saved model's structure may differ from what this script expects. "
        "Send the printed layer names back for a follow-up fix."
    )

with open(config_path, "w", encoding="utf-8") as f:
    json.dump(config, f)

print(f"\nTotal layers patched: {patched_count}")

# ------------------------------------------------------------------
# Re-zip back into the .keras file
# ------------------------------------------------------------------
with zipfile.ZipFile(MODEL_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
    for name in original_names:
        file_path = work_dir / name
        zf.write(file_path, arcname=name)

shutil.rmtree(work_dir)

print("\n" + "=" * 70)
print("PATCH COMPLETE")
print("=" * 70)
print("\nModel file updated in place:", MODEL_PATH)
print("Original backed up at:", backup_path)
print("\nNext step: python evaluate_medfairxnet_final_clahe.py")