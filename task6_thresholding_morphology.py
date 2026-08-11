import cv2
import numpy as np
import os


# ============================================================
# 1. Configuration
# ============================================================

IMAGE_PATH = "raw_captures/document.jpg"
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. Load image
# ============================================================

image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError(
        f"Could not load image: {IMAGE_PATH}"
    )

print("=" * 65)
print("TASK 6 — THRESHOLDING AND MORPHOLOGY")
print("=" * 65)

height, width = image.shape[:2]

print(f"Image resolution: {width} x {height}")


# ============================================================
# 3. Convert to grayscale
# ============================================================

gray = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2GRAY
)

cv2.imwrite(
    os.path.join(
        OUTPUT_DIR,
        "task6_grayscale.jpg"
    ),
    gray
)


# ============================================================
# 4. OTSU THRESHOLDING
# ============================================================

# THRESH_BINARY_INV is used because text is normally dark
# and the background is normally light.
#
# Result:
#   Text       -> white
#   Background -> black

otsu_threshold, binary = cv2.threshold(
    gray,
    0,
    255,
    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
)

print(f"\nOtsu threshold value: {otsu_threshold:.2f}")


# ============================================================
# 5. Save Otsu result
# ============================================================

cv2.imwrite(
    os.path.join(
        OUTPUT_DIR,
        "task6_otsu_binary.jpg"
    ),
    binary
)


# ============================================================
# 6. Estimate text/component size
# ============================================================

num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
    binary,
    connectivity=8
)


component_heights = []

for i in range(1, num_labels):

    x = stats[i, cv2.CC_STAT_LEFT]
    y = stats[i, cv2.CC_STAT_TOP]

    w = stats[i, cv2.CC_STAT_WIDTH]
    h = stats[i, cv2.CC_STAT_HEIGHT]

    area = stats[i, cv2.CC_STAT_AREA]

    # Ignore tiny noise and very large page regions.
    if (
        area >= 10
        and h >= 3
        and h <= height * 0.15
        and w <= width * 0.25
    ):
        component_heights.append(h)


# ============================================================
# 7. Estimate typical character/component height
# ============================================================

if len(component_heights) > 0:

    median_component_height = float(
        np.median(component_heights)
    )

else:

    # Fallback if connected components are unusual
    median_component_height = max(
        10,
        height * 0.02
    )


print(
    f"Estimated median text component height: "
    f"{median_component_height:.2f} pixels"
)


# ============================================================
# 8. Select candidate kernel sizes
# ============================================================

# Closing kernel should be small relative to the text.
#
# A kernel that is too small may not close tiny gaps.
# A kernel that is too large may merge neighboring letters.

estimated_kernel = int(
    round(median_component_height * 0.08)
)

# Make sure kernel is odd
if estimated_kernel % 2 == 0:
    estimated_kernel += 1

# Keep the kernel within practical limits
estimated_kernel = max(
    3,
    min(estimated_kernel, 9)
)


candidate_kernel_sizes = [
    3,
    5,
    7,
    9
]


print(
    f"Image-adaptive candidate kernel: "
    f"{estimated_kernel} x {estimated_kernel}"
)


# ============================================================
# 9. Test multiple closing kernels
# ============================================================

closing_results = {}

for kernel_size in candidate_kernel_sizes:

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (kernel_size, kernel_size)
    )

    closed = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=1
    )

    closing_results[kernel_size] = closed

    cv2.imwrite(
        os.path.join(
            OUTPUT_DIR,
            f"task6_closing_{kernel_size}x{kernel_size}.jpg"
        ),
        closed
    )


# ============================================================
# 10. Measure how much each kernel changes the binary image
# ============================================================

change_scores = {}

binary_bool = binary > 0

for kernel_size, result in closing_results.items():

    result_bool = result > 0

    changed_pixels = np.sum(
        binary_bool != result_bool
    )

    total_pixels = binary.shape[0] * binary.shape[1]

    percentage_changed = (
        changed_pixels / total_pixels
    ) * 100

    change_scores[kernel_size] = percentage_changed


# ============================================================
# 11. Select working kernel
# ============================================================

# Prefer the image-adaptive estimate if available.
# Otherwise use 5x5 as a conservative default.

working_kernel = estimated_kernel

if working_kernel not in candidate_kernel_sizes:

    working_kernel = 5


cleaned = closing_results[
    working_kernel
]


# ============================================================
# 12. Save final cleaned image
# ============================================================

cv2.imwrite(
    os.path.join(
        OUTPUT_DIR,
        "task6_final_closed.jpg"
    ),
    cleaned
)


# ============================================================
# 13. Create Otsu vs closing comparison
# ============================================================

otsu_display = cv2.cvtColor(
    binary,
    cv2.COLOR_GRAY2BGR
)

closed_display = cv2.cvtColor(
    cleaned,
    cv2.COLOR_GRAY2BGR
)

display_width = 700

display_height = int(
    display_width * height / width
)

otsu_display = cv2.resize(
    otsu_display,
    (display_width, display_height)
)

closed_display = cv2.resize(
    closed_display,
    (display_width, display_height)
)


# Labels

cv2.rectangle(
    otsu_display,
    (0, 0),
    (350, 50),
    (255, 255, 255),
    -1
)

cv2.putText(
    otsu_display,
    "OTSU THRESHOLD",
    (10, 35),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (0, 0, 0),
    2,
    cv2.LINE_AA
)


cv2.rectangle(
    closed_display,
    (0, 0),
    (450, 50),
    (255, 255, 255),
    -1
)

cv2.putText(
    closed_display,
    f"CLOSING {working_kernel}x{working_kernel}",
    (10, 35),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (0, 0, 0),
    2,
    cv2.LINE_AA
)


comparison = np.hstack([
    otsu_display,
    closed_display
])


cv2.imwrite(
    os.path.join(
        OUTPUT_DIR,
        "task6_otsu_vs_closing.jpg"
    ),
    comparison
)


# ============================================================
# 14. Find a word-like region for before/after crop
# ============================================================

# To find a word, temporarily connect nearby characters
# horizontally.
#
# This is NOT the final morphology operation.
# It is only used to locate a word-sized region automatically.

word_kernel_width = max(
    15,
    int(median_component_height * 1.5)
)

word_kernel_width = min(
    word_kernel_width,
    80
)

word_kernel = cv2.getStructuringElement(
    cv2.MORPH_RECT,
    (word_kernel_width, 3)
)

word_mask = cv2.morphologyEx(
    binary,
    cv2.MORPH_CLOSE,
    word_kernel,
    iterations=1
)


# ============================================================
# 15. Find candidate word regions
# ============================================================

word_contours, _ = cv2.findContours(
    word_mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)


word_candidates = []

for contour in word_contours:

    x, y, w, h = cv2.boundingRect(
        contour
    )

    area = w * h

    # Word-like region:
    # not extremely tiny and not the whole page
    if (
        w >= 20
        and h >= 5
        and w <= width * 0.50
        and h <= height * 0.15
        and area >= 100
    ):

        word_candidates.append(
            (x, y, w, h, area)
        )


# ============================================================
# 16. Select a word candidate
# ============================================================

if len(word_candidates) > 0:

    # Choose a reasonably large word-sized region.
    word_candidates.sort(
        key=lambda item: item[4],
        reverse=True
    )

    x, y, w, h, area = word_candidates[0]

else:

    # Fallback: central region
    w = int(width * 0.25)
    h = int(height * 0.08)

    x = int(
        (width - w) / 2
    )

    y = int(
        (height - h) / 2
    )


# ============================================================
# 17. Add padding around selected word
# ============================================================

padding_x = max(
    10,
    int(w * 0.15)
)

padding_y = max(
    10,
    int(h * 0.30)
)

x1 = max(
    0,
    x - padding_x
)

y1 = max(
    0,
    y - padding_y
)

x2 = min(
    width,
    x + w + padding_x
)

y2 = min(
    height,
    y + h + padding_y
)


# ============================================================
# 18. Crop original/Otsu/closed images
# ============================================================

original_crop = image[
    y1:y2,
    x1:x2
]

otsu_crop = binary[
    y1:y2,
    x1:x2
]

closed_crop = cleaned[
    y1:y2,
    x1:x2
]


# ============================================================
# 19. Save individual crops
# ============================================================

cv2.imwrite(
    os.path.join(
        OUTPUT_DIR,
        "task6_word_original_crop.jpg"
    ),
    original_crop
)

cv2.imwrite(
    os.path.join(
        OUTPUT_DIR,
        "task6_word_otsu_crop.jpg"
    ),
    otsu_crop
)

cv2.imwrite(
    os.path.join(
        OUTPUT_DIR,
        "task6_word_closed_crop.jpg"
    ),
    closed_crop
)


# ============================================================
# 20. Create before/after word crop
# ============================================================

crop_width = 600

crop_height = int(
    crop_width *
    original_crop.shape[0] /
    original_crop.shape[1]
)


original_crop_display = cv2.resize(
    original_crop,
    (crop_width, crop_height)
)

closed_crop_display = cv2.resize(
    closed_crop,
    (crop_width, crop_height)
)

original_crop_display = cv2.cvtColor(
    original_crop_display,
    cv2.COLOR_BGR2RGB
)

closed_crop_display = cv2.cvtColor(
    closed_crop_display,
    cv2.COLOR_GRAY2RGB
)


# Labels

cv2.rectangle(
    original_crop_display,
    (0, 0),
    (300, 45),
    (255, 255, 255),
    -1
)

cv2.putText(
    original_crop_display,
    "BEFORE",
    (10, 32),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (0, 0, 0),
    2,
    cv2.LINE_AA
)


cv2.rectangle(
    closed_crop_display,
    (0, 0),
    (500, 45),
    (255, 255, 255),
    -1
)

cv2.putText(
    closed_crop_display,
    f"AFTER CLOSING {working_kernel}x{working_kernel}",
    (10, 32),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.65,
    (0, 0, 0),
    2,
    cv2.LINE_AA
)


word_comparison = np.hstack([
    original_crop_display,
    closed_crop_display
])


cv2.imwrite(
    os.path.join(
        OUTPUT_DIR,
        "task6_word_before_after.jpg"
    ),
    cv2.cvtColor(
        word_comparison,
        cv2.COLOR_RGB2BGR
    )
)


# ============================================================
# 21. Create kernel comparison image
# ============================================================

kernel_images = []

for kernel_size in candidate_kernel_sizes:

    result = closing_results[
        kernel_size
    ]

    result_display = cv2.cvtColor(
        result,
        cv2.COLOR_GRAY2BGR
    )

    result_display = cv2.resize(
        result_display,
        (400, int(400 * height / width))
    )

    cv2.rectangle(
        result_display,
        (0, 0),
        (250, 45),
        (255, 255, 255),
        -1
    )

    cv2.putText(
        result_display,
        f"Closing {kernel_size}x{kernel_size}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        2,
        cv2.LINE_AA
    )

    kernel_images.append(
        result_display
    )


row1 = np.hstack([
    kernel_images[0],
    kernel_images[1]
])

row2 = np.hstack([
    kernel_images[2],
    kernel_images[3]
])

kernel_comparison = np.vstack([
    row1,
    row2
])


cv2.imwrite(
    os.path.join(
        OUTPUT_DIR,
        "task6_kernel_comparison.jpg"
    ),
    kernel_comparison
)


# ============================================================
# 22. Generate explanation
# ============================================================

if working_kernel == 3:

    kernel_reason = """
The 3x3 kernel was selected because the estimated text
components are relatively small. A smaller effective
kernel would have little additional ability to close
gaps, while larger kernels risk joining neighboring
characters or thickening the text.
"""

elif working_kernel == 5:

    kernel_reason = """
The 5x5 kernel provides a moderate amount of closing.
It is large enough to bridge small breaks in the binary
text but remains small relative to the estimated text
component size. A 3x3 kernel may not close all small
gaps, while 7x7 or larger kernels can begin to thicken
characters or connect neighboring letters.
"""

elif working_kernel == 7:

    kernel_reason = """
The 7x7 kernel was selected because the estimated text
components are large enough that a smaller kernel may
not effectively close the small gaps present in the
binary text. A larger kernel such as 9x9 risks joining
nearby characters and altering the original font
structure.
"""

else:

    kernel_reason = """
The 9x9 kernel was selected because the image contains
relatively large text components and small kernels were
not sufficient to close the gaps. A larger kernel would
risk merging neighboring characters and significantly
changing the shape of the text.
"""


# ============================================================
# 23. Save complete results
# ============================================================

results_path = os.path.join(
    OUTPUT_DIR,
    "task6_results.txt"
)

with open(results_path, "w") as f:

    f.write(
        "TASK 6 — THRESHOLDING AND MORPHOLOGY\n"
    )

    f.write("=" * 65 + "\n\n")

    f.write(
        f"Image resolution: {width} x {height}\n"
    )

    f.write(
        f"Otsu threshold value: "
        f"{otsu_threshold:.2f}\n\n"
    )

    f.write(
        f"Estimated median text component height: "
        f"{median_component_height:.2f} pixels\n\n"
    )

    f.write(
        "Tested closing kernels:\n"
    )

    for kernel_size in candidate_kernel_sizes:

        f.write(
            f"  {kernel_size}x{kernel_size}: "
            f"{change_scores[kernel_size]:.4f}% "
            f"pixels changed\n"
        )

    f.write("\n")

    f.write(
        f"Selected working kernel: "
        f"{working_kernel}x{working_kernel}\n\n"
    )

    f.write(
        "Why this kernel:\n"
    )

    f.write(
        kernel_reason.strip()
    )

    f.write("\n\n")

    f.write(
        "Word crop coordinates:\n"
    )

    f.write(
        f"x1={x1}, y1={y1}, "
        f"x2={x2}, y2={y2}\n"
    )

    f.write("\nOutput files:\n")

    f.write(
        "task6_grayscale.jpg\n"
    )

    f.write(
        "task6_otsu_binary.jpg\n"
    )

    f.write(
        "task6_final_closed.jpg\n"
    )

    f.write(
        "task6_otsu_vs_closing.jpg\n"
    )

    f.write(
        "task6_word_before_after.jpg\n"
    )

    f.write(
        "task6_kernel_comparison.jpg\n"
    )


# ============================================================
# 24. Completion message
# ============================================================

print("\n" + "=" * 65)
print("TASK 6 COMPLETED")
print("=" * 65)

print(
    f"\nOtsu threshold: {otsu_threshold:.2f}"
)

print(
    f"Estimated text height: "
    f"{median_component_height:.2f} pixels"
)

print(
    f"Selected kernel: "
    f"{working_kernel}x{working_kernel}"
)

print("\nSaved outputs:")

print(
    "  outputs/task6_grayscale.jpg"
)

print(
    "  outputs/task6_otsu_binary.jpg"
)

print(
    "  outputs/task6_final_closed.jpg"
)

print(
    "  outputs/task6_otsu_vs_closing.jpg"
)

print(
    "  outputs/task6_word_original_crop.jpg"
)

print(
    "  outputs/task6_word_otsu_crop.jpg"
)

print(
    "  outputs/task6_word_closed_crop.jpg"
)

print(
    "  outputs/task6_word_before_after.jpg"
)

print(
    "  outputs/task6_kernel_comparison.jpg"
)

print(
    "  outputs/task6_results.txt"
)