import os

BASE_DIR = r"C:\Users\HP\Desktop\Melonma\data\raw\ISIC2018"

print("=" * 70)
print("ISIC2018 FOLDER STRUCTURE")
print("=" * 70)

for root, dirs, files in os.walk(BASE_DIR):
    level = root.replace(BASE_DIR, "").count(os.sep)
    indent = "  " * level

    print(f"{indent}[FOLDER] {os.path.basename(root)}")

    for file in files[:10]:
        print(f"{indent}  - {file}")

    if len(files) > 10:
        print(f"{indent}  ... and {len(files) - 10} more files")

print("\n" + "=" * 70)
print("ALL CSV FILES")
print("=" * 70)

for root, dirs, files in os.walk(BASE_DIR):
    for file in files:
        if file.lower().endswith(".csv"):
            print(os.path.join(root, file))

print("\n" + "=" * 70)
print("ALL IMAGE FOLDERS")
print("=" * 70)

for root, dirs, files in os.walk(BASE_DIR):
    image_count = sum(
        1 for f in files
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    )

    if image_count > 0:
        print(f"{root} --> {image_count} images")