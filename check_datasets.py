from pathlib import Path

# Project path
PROJECT_DIR = Path(r"C:\Users\HP\Desktop\Melonma")
RAW_DIR = PROJECT_DIR / "data" / "raw"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def count_images(folder):
    if not folder.exists():
        return 0

    return sum(
        1 for file in folder.rglob("*")
        if file.suffix.lower() in IMAGE_EXTENSIONS
    )


def show_dataset(dataset_name):
    folder = RAW_DIR / dataset_name

    print("\n" + "=" * 60)
    print(f"DATASET: {dataset_name}")
    print("=" * 60)

    if not folder.exists():
        print("❌ Folder not found:", folder)
        return

    all_files = list(folder.rglob("*"))

    image_files = [
        f for f in all_files
        if f.suffix.lower() in IMAGE_EXTENSIONS
    ]

    csv_files = [
        f for f in all_files
        if f.suffix.lower() == ".csv"
    ]

    print("Folder:", folder)
    print("Total files:", len(all_files))
    print("Image files:", len(image_files))
    print("CSV files:", len(csv_files))

    if csv_files:
        print("\nCSV files:")
        for csv in csv_files:
            print("  -", csv.name)

    print("\nFirst 10 image files:")
    for image in image_files[:10]:
        print("  -", image.name)


# Check all three datasets
show_dataset("ISIC2018")
show_dataset("HAM10000")
show_dataset("PH2")

print("\n" + "=" * 60)
print("DATASET VERIFICATION COMPLETED")
print("=" * 60)