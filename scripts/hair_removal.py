from pathlib import Path
import cv2


# ============================================================
# Melonma - Hair & Artifact Removal
# DullRazor-style black-hat detection + inpainting
# ============================================================

INPUT_DIR = Path("data/processed/images_roi")
OUTPUT_DIR = Path("data/processed/hair_removed")

# Hair detection parameters
KERNEL_SIZE = 17
THRESHOLD = 10
INPAINT_RADIUS = 3


def remove_hair(image):
    """
    Detect dark hair-like structures using a black-hat
    morphological operation and remove them using inpainting.
    """

    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Morphological kernel
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (KERNEL_SIZE, KERNEL_SIZE)
    )

    # Black-hat operation:
    # highlights dark structures on a lighter background
    blackhat = cv2.morphologyEx(
        gray,
        cv2.MORPH_BLACKHAT,
        kernel
    )

    # Threshold to create hair mask
    _, hair_mask = cv2.threshold(
        blackhat,
        THRESHOLD,
        255,
        cv2.THRESH_BINARY
    )

    # Remove tiny noise
    cleanup_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3)
    )

    hair_mask = cv2.morphologyEx(
        hair_mask,
        cv2.MORPH_OPEN,
        cleanup_kernel
    )

    # Slightly expand detected hair regions
    hair_mask = cv2.dilate(
        hair_mask,
        cleanup_kernel,
        iterations=1
    )

    # Inpaint detected hair regions
    result = cv2.inpaint(
        image,
        hair_mask,
        INPAINT_RADIUS,
        cv2.INPAINT_TELEA
    )

    return result, hair_mask


def process_images():
    """Process all ROI images."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_DIR.exists():
        print(f"[ERROR] Input folder not found:")
        print(f"       {INPUT_DIR}")
        return

    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff"
    }

    files = [
        file for file in INPUT_DIR.iterdir()
        if file.is_file()
        and file.suffix.lower() in image_extensions
    ]

    if not files:
        print("[WARNING] No images found in input folder.")
        return

    print("=" * 60)
    print("Melonma - Hair Removal Pipeline")
    print("=" * 60)
    print(f"Input : {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Images: {len(files)}")
    print("=" * 60)

    success = 0
    failed = 0

    for index, input_file in enumerate(files, start=1):

        try:
            image = cv2.imread(str(input_file))

            if image is None:
                print(
                    f"[{index}/{len(files)}] "
                    f"FAILED: {input_file.name}"
                )
                failed += 1
                continue

            # Hair removal
            result, hair_mask = remove_hair(image)

            # Preserve original filename
            output_file = OUTPUT_DIR / input_file.name

            # Save hair-removed image
            saved = cv2.imwrite(
                str(output_file),
                result
            )

            if saved:
                print(
                    f"[{index}/{len(files)}] "
                    f"OK: {input_file.name}"
                )
                success += 1
            else:
                print(
                    f"[{index}/{len(files)}] "
                    f"FAILED TO SAVE: {input_file.name}"
                )
                failed += 1

        except Exception as error:
            print(
                f"[{index}/{len(files)}] "
                f"ERROR: {input_file.name}"
            )
            print(f"       {error}")
            failed += 1

    print("=" * 60)
    print("PROCESS COMPLETE")
    print("=" * 60)
    print(f"Successful : {success}")
    print(f"Failed     : {failed}")
    print(f"Output     : {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    process_images()