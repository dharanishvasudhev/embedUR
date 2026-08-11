import cv2
import numpy as np
import os


# ============================================================
# 1. Configuration
# ============================================================

IMAGE_PATH = "raw_captures/desk_objects1.jpeg"
OUTPUT_DIR = "outputs"

# ------------------------------------------------------------
# IMPORTANT:
# Replace 42 with the last two digits of your employee ID.
# If you don't have one, use the last two digits of your
# phone number.
# ------------------------------------------------------------

REFERENCE_NUMBER = 42


# ============================================================
# 2. Create output directory
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 3. Calculate assignment-specific values
# ============================================================

last_digit = REFERENCE_NUMBER % 10

angle = REFERENCE_NUMBER * 3

scale = 1 + (last_digit / 10)


# ============================================================
# 4. Print reference values
# ============================================================

print("=" * 60)
print("TASK 3 — GEOMETRIC TRANSFORMATIONS")
print("=" * 60)

print(f"Reference number : {REFERENCE_NUMBER:02d}")
print(f"Last digit       : {last_digit}")
print(f"Rotation angle   : {angle} degrees")
print(f"Scale factor     : {scale}")

print("=" * 60)


# ============================================================
# 5. Load image
# ============================================================

image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError(
        f"Could not load image: {IMAGE_PATH}"
    )

height, width = image.shape[:2]

print(f"Original image size: {width} x {height}")


# ============================================================
# 6. TRANSLATION
# ============================================================

# Translation values in pixels
tx = 80
ty = 50

translation_matrix = np.float32([
    [1, 0, tx],
    [0, 1, ty]
])

translated = cv2.warpAffine(
    image,
    translation_matrix,
    (width, height)
)

cv2.imwrite(
    os.path.join(
        OUTPUT_DIR,
        "task3_translation.jpg"
    ),
    translated
)


# ============================================================
# 7. ROTATION
# ============================================================

center = (
    width // 2,
    height // 2
)

rotation_matrix = cv2.getRotationMatrix2D(
    center,
    angle,
    1.0
)

rotated = cv2.warpAffine(
    image,
    rotation_matrix,
    (width, height)
)

cv2.imwrite(
    os.path.join(
        OUTPUT_DIR,
        "task3_rotation.jpg"
    ),
    rotated
)


# ============================================================
# 8. SCALING
# ============================================================

scaled = cv2.resize(
    image,
    None,
    fx=scale,
    fy=scale,
    interpolation=cv2.INTER_LINEAR
)

cv2.imwrite(
    os.path.join(
        OUTPUT_DIR,
        "task3_scaling.jpg"
    ),
    scaled
)


# ============================================================
# 9. AFFINE TRANSFORMATION
# ============================================================

# Three points from the original image
src_points = np.float32([
    [0, 0],
    [width - 1, 0],
    [0, height - 1]
])

# Modified locations of those points
dst_points = np.float32([
    [0.10 * width, 0.10 * height],
    [0.90 * width, 0.05 * height],
    [0.05 * width, 0.90 * height]
])

affine_matrix = cv2.getAffineTransform(
    src_points,
    dst_points
)

affine = cv2.warpAffine(
    image,
    affine_matrix,
    (width, height)
)

cv2.imwrite(
    os.path.join(
        OUTPUT_DIR,
        "task3_affine.jpg"
    ),
    affine
)


# ============================================================
# 10. Combined transformation
# ============================================================

# First rotate
combined = cv2.warpAffine(
    image,
    rotation_matrix,
    (width, height)
)

# Then scale
combined = cv2.resize(
    combined,
    None,
    fx=scale,
    fy=scale,
    interpolation=cv2.INTER_LINEAR
)

# Resize back to original dimensions so it can be
# compared easily with the original
combined = cv2.resize(
    combined,
    (width, height),
    interpolation=cv2.INTER_LINEAR
)

cv2.imwrite(
    os.path.join(
        OUTPUT_DIR,
        "task3_all_transformations.jpg"
    ),
    combined
)


# ============================================================
# 11. Save transformation information
# ============================================================

info_path = os.path.join(
    OUTPUT_DIR,
    "task3_transformation_values.txt"
)

with open(info_path, "w") as f:

    f.write("TASK 3 — GEOMETRIC TRANSFORMATIONS\n")
    f.write("=" * 50 + "\n\n")

    f.write(
        f"Reference number: {REFERENCE_NUMBER:02d}\n"
    )

    f.write(
        f"Last digit: {last_digit}\n"
    )

    f.write(
        f"Rotation angle = {REFERENCE_NUMBER} × 3\n"
    )

    f.write(
        f"Rotation angle = {angle} degrees\n"
    )

    f.write(
        f"Scale = 1 + ({last_digit} / 10)\n"
    )

    f.write(
        f"Scale factor = {scale}\n\n"
    )

    f.write("Translation:\n")
    f.write(f"X translation = {tx} pixels\n")
    f.write(f"Y translation = {ty} pixels\n\n")

    f.write("Affine transformation:\n")
    f.write("Three-point affine transformation applied.\n")


# ============================================================
# 12. Completion message
# ============================================================

print("\nAll Task 3 outputs saved successfully.")

print("\nSaved files:")

print("  outputs/task3_translation.jpg")
print("  outputs/task3_rotation.jpg")
print("  outputs/task3_scaling.jpg")
print("  outputs/task3_affine.jpg")
print("  outputs/task3_all_transformations.jpg")
print("  outputs/task3_transformation_values.txt")