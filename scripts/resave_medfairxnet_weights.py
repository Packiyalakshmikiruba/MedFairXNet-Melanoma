"""
MELONMA - RESAVE MEDFAIRXNET WITH FIXED ARCHITECTURE (no retraining)
============================================================
Your model already finished training. The only problem is how the
CBAM Lambda layers were serialized -- the saved config uses an
anonymous Python lambda whose reconstructed function loses access to
the `tf` module at load time. medfairxnet_model.py has now been
updated to use named functions instead, which serialize/deserialize
reliably. This script:

  1. Builds a FRESH MedFairXNet model using the fixed architecture
     code (same layer names, so weights line up 1:1).
  2. Extracts the trained weights (model.weights.h5) directly out of
     your existing .keras archive -- this sidesteps loading the
     broken saved architecture entirely.
  3. Loads those trained weights into the fresh model.
  4. Saves the fresh model (fixed architecture + trained weights)
     back to the same .keras path, so evaluate_medfairxnet_final_clahe.py
     will load cleanly from now on.

No GPU time, no retraining -- this only takes a few seconds.

Run once from the scripts/ folder (after replacing medfairxnet_model.py
with the updated version):
    python resave_medfairxnet_weights.py

Then run evaluate_medfairxnet_final_clahe.py as normal.
"""

import shutil
import zipfile
from pathlib import Path

import tensorflow as tf

from medfairxnet_model import build_medfairxnet, IMAGE_SIZE

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "melanoma_medfairxnet_final_clahe.keras"

print("=" * 70)
print("RESAVING MEDFAIRXNET WITH FIXED ARCHITECTURE")
print("=" * 70)
print("\nModel file:", MODEL_PATH)

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found:\n{MODEL_PATH}")

# ------------------------------------------------------------------
# Backup (separate from the earlier .bak so we never lose a good copy)
# ------------------------------------------------------------------
backup_path = MODEL_PATH.with_suffix(".keras.bak2")
if not backup_path.exists():
    shutil.copy2(MODEL_PATH, backup_path)
    print("Backup created:", backup_path)

# ------------------------------------------------------------------
# Step 1: extract the trained weights file straight out of the
# existing .keras archive (a .keras file is just a zip).
# ------------------------------------------------------------------
work_dir = MODEL_PATH.parent / "_medfairxnet_resave_tmp"
if work_dir.exists():
    shutil.rmtree(work_dir)
work_dir.mkdir()

with zipfile.ZipFile(MODEL_PATH, "r") as zf:
    names = zf.namelist()
    weights_name = next((n for n in names if n.endswith(".weights.h5")), None)
    if weights_name is None:
        raise RuntimeError(
            f"Could not find a *.weights.h5 entry inside the .keras archive. "
            f"Archive contents: {names}"
        )
    zf.extract(weights_name, work_dir)

extracted_weights_path = work_dir / weights_name
print("Extracted trained weights:", extracted_weights_path)

# ------------------------------------------------------------------
# Step 2: build a fresh model with the fixed architecture code
# ------------------------------------------------------------------
print("\nBuilding fresh MedFairXNet architecture...")
model = build_medfairxnet(image_size=IMAGE_SIZE, freeze_backbone_until=None)
print("Fresh model built. Total params:", f"{model.count_params():,}")

# ------------------------------------------------------------------
# Step 3: load the trained weights into the fresh model
# ------------------------------------------------------------------
print("\nLoading trained weights into fresh architecture...")
model.load_weights(str(extracted_weights_path))
print("Weights loaded successfully.")

# ------------------------------------------------------------------
# Step 4: save back to the same path with the fixed architecture
# ------------------------------------------------------------------
model.save(str(MODEL_PATH))
print("\nModel re-saved with fixed architecture + trained weights:")
print(MODEL_PATH)

shutil.rmtree(work_dir)

print("\n" + "=" * 70)
print("RESAVE COMPLETE")
print("=" * 70)
print("\nNext step: python evaluate_medfairxnet_final_clahe.py")