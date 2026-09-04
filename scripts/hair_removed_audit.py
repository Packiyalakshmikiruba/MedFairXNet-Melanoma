from pathlib import Path
import cv2
from collections import Counter

INPUT_DIR = Path("data/processed/hair_removed")

VALID_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"
}

expected_shape = (224, 224)
expected_channels = 3

total = 0
valid = 0
invalid = 0

invalid_files = []
shape_counts = Counter()
channel_counts = Counter()

for file in INPUT_DIR.iterdir():

    if not file.is_file():
        continue

    if file.suffix.lower() not in VALID_EXTENSIONS:
        continue

    total += 1

    image = cv2.imread(str(file))

    if image is None:
        invalid += 1
        invalid_files.append((file.name, "Unreadable"))
        continue

    shape = image.shape

    if len(shape) != 3:
        invalid += 1
        invalid_files.append((file.name, f"Shape={shape}"))
        continue

    height, width, channels = shape

    shape_counts[(height, width)] += 1
    channel_counts[channels] += 1

    if (height, width) != expected_shape:
        invalid += 1
        invalid_files.append(
            (file.name, f"Wrong size={height}x{width}")
        )
        continue

    if channels != expected_channels:
        invalid += 1
        invalid_files.append(
            (file.name, f"Wrong channels={channels}")
        )
        continue

    valid += 1


print("=" * 60)
print("HAIR-REMOVED DATASET AUDIT")
print("=" * 60)

print(f"Total images      : {total}")
print(f"Valid images      : {valid}")
print(f"Invalid images    : {invalid}")

print()
print("IMAGE DIMENSIONS")
for shape, count in shape_counts.most_common():
    print(f"{shape}: {count}")

print()
print("CHANNELS")
for channels, count in channel_counts.most_common():
    print(f"{channels} channels: {count}")

print()
print("EXPECTED")
print("Size     : 224 x 224")
print("Channels : 3")

print()
print("=" * 60)

if invalid == 0:
    print("STATUS: PASS")
    print("All images are valid for the next preprocessing stage.")
else:
    print("STATUS: WARNING")
    print("Invalid images found:")
    
    for filename, reason in invalid_files[:50]:
        print(f"{filename} -> {reason}")

    if len(invalid_files) > 50:
        print(f"... and {len(invalid_files) - 50} more")

print("=" * 60)