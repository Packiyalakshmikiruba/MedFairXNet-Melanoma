from pathlib import Path

import cv2
from tqdm import tqdm


# =========================================================
# Melonma - CLAHE Contrast Enhancement
# =========================================================
# Pipeline:
#
# ROI Images
#     ↓
# Hair Removal
#     ↓
# CLAHE Enhancement
#
# Input:
# data/processed/hair_removed
#
# Output:
# data/processed/images_clahe
# =========================================================


# =========================================================
# CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "hair_removed"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "images_clahe"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# SUPPORTED IMAGE EXTENSIONS
# =========================================================

VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff"
}


# =========================================================
# CLAHE FUNCTION
# =========================================================

def apply_clahe(image):
    """
    Apply CLAHE to the L channel in LAB color space.

    CLAHE improves local contrast while preserving
    the original color information.

    Steps:
        BGR
         ↓
        LAB
         ↓
        Extract L channel
         ↓
        CLAHE
         ↓
        Merge L + A + B
         ↓
        BGR
    """

    # -----------------------------------------------------
    # BGR → LAB
    # -----------------------------------------------------

    lab = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB
    )

    # -----------------------------------------------------
    # Split LAB channels
    # -----------------------------------------------------

    l_channel, a_channel, b_channel = cv2.split(
        lab
    )

    # -----------------------------------------------------
    # Create CLAHE
    # -----------------------------------------------------

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    # -----------------------------------------------------
    # Apply CLAHE only to L channel
    # -----------------------------------------------------

    enhanced_l = clahe.apply(
        l_channel
    )

    # -----------------------------------------------------
    # Merge channels
    # -----------------------------------------------------

    enhanced_lab = cv2.merge(
        [
            enhanced_l,
            a_channel,
            b_channel
        ]
    )

    # -----------------------------------------------------
    # LAB → BGR
    # -----------------------------------------------------

    enhanced_image = cv2.cvtColor(
        enhanced_lab,
        cv2.COLOR_LAB2BGR
    )

    return enhanced_image


# =========================================================
# FIND INPUT IMAGES
# =========================================================

def get_image_files():
    """
    Find all supported image files from the input directory.
    """

    if not INPUT_DIR.exists():

        raise FileNotFoundError(
            f"Input directory not found:\n{INPUT_DIR}"
        )

    files = [
        file
        for file in INPUT_DIR.iterdir()
        if file.is_file()
        and file.suffix.lower() in VALID_EXTENSIONS
    ]

    return sorted(files)


# =========================================================
# MAIN PROCESS
# =========================================================

def main():

    print("=" * 70)
    print("MELONMA - CLAHE CONTRAST ENHANCEMENT")
    print("=" * 70)

    print("\nInput directory:")
    print(INPUT_DIR)

    print("\nOutput directory:")
    print(OUTPUT_DIR)

    # -----------------------------------------------------
    # Find images
    # -----------------------------------------------------

    image_files = get_image_files()

    print("\nInput images:", len(image_files))

    if len(image_files) == 0:

        print(
            "\n[ERROR] No supported images found."
        )

        return

    print("\nSupported extensions:")

    for extension in sorted(VALID_EXTENSIONS):
        print(f"  {extension}")

    print("\n" + "=" * 70)
    print("STARTING CLAHE PROCESSING")
    print("=" * 70)

    # -----------------------------------------------------
    # Counters
    # -----------------------------------------------------

    processed = 0
    failed = 0

    failed_files = []

    # -----------------------------------------------------
    # Process images
    # -----------------------------------------------------

    for image_path in tqdm(
        image_files,
        desc="Applying CLAHE"
    ):

        try:

            # -------------------------------------------------
            # Read image
            # -------------------------------------------------

            image = cv2.imread(
                str(image_path)
            )

            if image is None:

                failed += 1

                failed_files.append(
                    (
                        image_path.name,
                        "Unable to read image"
                    )
                )

                continue

            # -------------------------------------------------
            # Apply CLAHE
            # -------------------------------------------------

            enhanced = apply_clahe(
                image
            )

            # -------------------------------------------------
            # Preserve original filename
            # -------------------------------------------------

            output_path = (
                OUTPUT_DIR
                / image_path.name
            )

            # -------------------------------------------------
            # Save enhanced image
            # -------------------------------------------------

            success = cv2.imwrite(
                str(output_path),
                enhanced
            )

            if not success:

                failed += 1

                failed_files.append(
                    (
                        image_path.name,
                        "Unable to save image"
                    )
                )

                continue

            processed += 1

        except Exception as error:

            failed += 1

            failed_files.append(
                (
                    image_path.name,
                    str(error)
                )
            )

    # =====================================================
    # FINAL SUMMARY
    # =====================================================

    print("\n" + "=" * 70)
    print("CLAHE ENHANCEMENT COMPLETED")
    print("=" * 70)

    print("\nInput images     :", len(image_files))
    print("Processed images :", processed)
    print("Failed images    :", failed)

    print("\nOutput directory:")
    print(OUTPUT_DIR)

    print("\nMethod:")
    print("LAB color space")
    print("L-channel CLAHE")
    print("clipLimit = 2.0")
    print("tileGridSize = (8, 8)")

    # -----------------------------------------------------
    # Failed files
    # -----------------------------------------------------

    if failed_files:

        print("\n" + "=" * 70)
        print("FAILED FILES")
        print("=" * 70)

        for filename, reason in failed_files[:50]:

            print(
                f"{filename} -> {reason}"
            )

        if len(failed_files) > 50:

            print(
                f"\n... and "
                f"{len(failed_files) - 50} more"
            )

    # -----------------------------------------------------
    # Final status
    # -----------------------------------------------------

    print("\n" + "=" * 70)

    if failed == 0 and processed == len(image_files):

        print("STATUS: PASS")

        print(
            "All hair-removed images were successfully "
            "enhanced using CLAHE."
        )

    else:

        print("STATUS: CHECK REQUIRED")

        print(
            "Some images failed during CLAHE processing."
        )

    print("=" * 70)


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()