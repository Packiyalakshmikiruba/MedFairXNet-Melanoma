from pathlib import Path

PH2_DIR = Path(r"C:\Users\HP\Desktop\Melonma\data\raw\PH2")

print("=" * 60)
print("PH2 FILE INSPECTION")
print("=" * 60)

files = [f for f in PH2_DIR.rglob("*") if f.is_file()]

print("Total files:", len(files))

# Show file extensions
extensions = {}

for file in files:
    ext = file.suffix.lower()

    if ext == "":
        ext = "[NO EXTENSION]"

    extensions[ext] = extensions.get(ext, 0) + 1

print("\nFile extensions:")

for ext, count in sorted(extensions.items()):
    print(f"{ext:15} : {count}")

print("\nFirst 30 files:")

for file in files[:30]:
    print(file.relative_to(PH2_DIR))