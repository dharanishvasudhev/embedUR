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
# 2. Load original document
# ============================================================

image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError(
        f"Could not load image: {IMAGE_PATH}"
    )

print("Original image loaded successfully.")
print("Image shape:", image.shape)


# ============================================================
# 3. SAVE PREDICTION BEFORE RUNNING ANY FILTER
# ============================================================

prediction = """
TASK 4 — NOISE AND FILTERING
========================================

Noise:
Gaussian noise

Mean:
0

Sigma:
25

Prediction BEFORE filtering:
I expect the Bilateral filter to preserve text
edges best because it considers both spatial
distance and intensity differences. Pixels across
strong text/background boundaries have different
intensities, so bilateral filtering should avoid
averaging across the boundary as aggressively as
Mean or Gaussian filtering.

I also expect the Median filter to preserve text
edges reasonably well, although Median filtering
is particularly effective for salt-and-pepper
noise rather than Gaussian noise.

The Gaussian filter should reduce the Gaussian
noise effectively but may blur the text edges.

The Mean filter is expected to reduce noise but
may produce the greatest amount of edge blurring.
"""

prediction_path = os.path.join(
    OUTPUT_DIR,
    "task4_prediction.txt"
)

with open(prediction_path, "w") as f:
    f.write(prediction)

print("\nPrediction saved BEFORE filtering:")
print(prediction)


# ============================================================
# 4. Convert image to float
# ============================================================

image_float = image.astype(np.float32)


# ============================================================
# 5. Generate Gaussian noise
# ============================================================

np.random.seed(42)

mean = 0
sigma = 25

noise = np.random.normal(
    mean,
    sigma,
    image_float.shape
)


# ============================================================
# 6. Add noise
# ============================================================

noisy_image = image_float + noise


# Clip values to valid image range
noisy_image = np.clip(
    noisy_image,
    0,
    255
).astype(np.uint8)


# ============================================================
# 7. Save noisy image
# ============================================================

noisy_path = os.path.join(
    OUTPUT_DIR,
    "task4_gaussian_noisy.jpg"
)

cv2.imwrite(
    noisy_path,
    noisy_image
)

print("Gaussian noisy image saved.")


# ============================================================
# 8. MEAN FILTER
# ============================================================

mean_filtered = cv2.blur(
    noisy_image,
    (5, 5)
)

cv2.imwrite(
    os.path.join(
        OUTPUT_DIR,
        "task4_mean_filter.jpg"
    ),
    mean_filtered
)


# ============================================================
# 9. GAUSSIAN FILTER
# ============================================================

gaussian_filtered = cv2.GaussianBlur(
    noisy_image,
    (5, 5),
    sigmaX=0
)

cv2.imwrite(
    os.path.join(
        OUTPUT_DIR,
        "task4_gaussian_filter.jpg"
    ),
    gaussian_filtered
)


# ============================================================
# 10. MEDIAN FILTER
# ============================================================

median_filtered = cv2.medianBlur(
    noisy_image,
    5
)

cv2.imwrite(
    os.path.join(
        OUTPUT_DIR,
        "task4_median_filter.jpg"
    ),
    median_filtered
)


# ============================================================
# 11. BILATERAL FILTER
# ============================================================

bilateral_filtered = cv2.bilateralFilter(
    noisy_image,
    d=9,
    sigmaColor=75,
    sigmaSpace=75
)

cv2.imwrite(
    os.path.join(
        OUTPUT_DIR,
        "task4_bilateral_filter.jpg"
    ),
    bilateral_filtered
)


# ============================================================
# 12. Create comparison image
# ============================================================

original_display = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)

noisy_display = cv2.cvtColor(
    noisy_image,
    cv2.COLOR_BGR2RGB
)

mean_display = cv2.cvtColor(
    mean_filtered,
    cv2.COLOR_BGR2RGB
)

gaussian_display = cv2.cvtColor(
    gaussian_filtered,
    cv2.COLOR_BGR2RGB
)

median_display = cv2.cvtColor(
    median_filtered,
    cv2.COLOR_BGR2RGB
)

bilateral_display = cv2.cvtColor(
    bilateral_filtered,
    cv2.COLOR_BGR2RGB
)


# ============================================================
# 13. Resize all images to same display size
# ============================================================

display_width = 500

aspect_ratio = image.shape[0] / image.shape[1]

display_height = int(
    display_width * aspect_ratio
)

def resize_for_display(img):
    return cv2.resize(
        img,
        (display_width, display_height)
    )


original_display = resize_for_display(
    original_display
)

noisy_display = resize_for_display(
    noisy_display
)

mean_display = resize_for_display(
    mean_display
)

gaussian_display = resize_for_display(
    gaussian_display
)

median_display = resize_for_display(
    median_display
)

bilateral_display = resize_for_display(
    bilateral_display
)


# ============================================================
# 14. Create labeled comparison
# ============================================================

font = cv2.FONT_HERSHEY_SIMPLEX


def add_label(img, text):

    output = img.copy()

    cv2.rectangle(
        output,
        (0, 0),
        (output.shape[1], 40),
        (255, 255, 255),
        -1
    )

    cv2.putText(
        output,
        text,
        (10, 28),
        font,
        0.7,
        (0, 0, 0),
        2,
        cv2.LINE_AA
    )

    return output


original_display = add_label(
    original_display,
    "Original"
)

noisy_display = add_label(
    noisy_display,
    "Gaussian Noise: sigma=25"
)

mean_display = add_label(
    mean_display,
    "Mean Filter"
)

gaussian_display = add_label(
    gaussian_display,
    "Gaussian Filter"
)

median_display = add_label(
    median_display,
    "Median Filter"
)

bilateral_display = add_label(
    bilateral_display,
    "Bilateral Filter"
)


# ============================================================
# 15. Arrange comparison into 2 x 3 grid
# ============================================================

row1 = np.hstack([
    original_display,
    noisy_display,
    mean_display
])

row2 = np.hstack([
    gaussian_display,
    median_display,
    bilateral_display
])

comparison = np.vstack([
    row1,
    row2
])


# ============================================================
# 16. Save comparison
# ============================================================

comparison_bgr = cv2.cvtColor(
    comparison,
    cv2.COLOR_RGB2BGR
)

comparison_path = os.path.join(
    OUTPUT_DIR,
    "task4_filter_comparison.jpg"
)

cv2.imwrite(
    comparison_path,
    comparison_bgr
)


# ============================================================
# 17. Create grayscale versions for edge analysis
# ============================================================

original_gray = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2GRAY
)

noisy_gray = cv2.cvtColor(
    noisy_image,
    cv2.COLOR_BGR2GRAY
)

mean_gray = cv2.cvtColor(
    mean_filtered,
    cv2.COLOR_BGR2GRAY
)

gaussian_gray = cv2.cvtColor(
    gaussian_filtered,
    cv2.COLOR_BGR2GRAY
)

median_gray = cv2.cvtColor(
    median_filtered,
    cv2.COLOR_BGR2GRAY
)

bilateral_gray = cv2.cvtColor(
    bilateral_filtered,
    cv2.COLOR_BGR2GRAY
)


# ============================================================
# 18. Calculate edge-strength indicator
# ============================================================

def laplacian_variance(img):

    return cv2.Laplacian(
        img,
        cv2.CV_64F
    ).var()


edge_values = {

    "Original":
        laplacian_variance(original_gray),

    "Noisy":
        laplacian_variance(noisy_gray),

    "Mean":
        laplacian_variance(mean_gray),

    "Gaussian":
        laplacian_variance(gaussian_gray),

    "Median":
        laplacian_variance(median_gray),

    "Bilateral":
        laplacian_variance(bilateral_gray)
}


# ============================================================
# 19. Save quantitative information
# ============================================================

results_path = os.path.join(
    OUTPUT_DIR,
    "task4_filter_results.txt"
)

with open(results_path, "w") as f:

    f.write(
        "TASK 4 — NOISE AND FILTERING RESULTS\n"
    )

    f.write("=" * 50 + "\n\n")

    f.write(
        "Gaussian noise parameters:\n"
    )

    f.write(
        "Mean = 0\n"
    )

    f.write(
        "Sigma = 25\n\n"
    )

    f.write(
        "Laplacian variance "
        "(edge/detail indicator):\n\n"
    )

    for name, value in edge_values.items():

        f.write(
            f"{name}: {value:.2f}\n"
        )


# ============================================================
# 20. Completion message
# ============================================================

print("\n" + "=" * 60)
print("TASK 4 COMPLETED")
print("=" * 60)

print("\nSaved outputs:")

print(
    "outputs/task4_prediction.txt"
)

print(
    "outputs/task4_gaussian_noisy.jpg"
)

print(
    "outputs/task4_mean_filter.jpg"
)

print(
    "outputs/task4_gaussian_filter.jpg"
)

print(
    "outputs/task4_median_filter.jpg"
)

print(
    "outputs/task4_bilateral_filter.jpg"
)

print(
    "outputs/task4_filter_comparison.jpg"
)

print(
    "outputs/task4_filter_results.txt"
)

print("\nEdge/detail indicator:")
for name, value in edge_values.items():
    print(f"{name:10s}: {value:.2f}")