import cv2
import numpy as np
import os


# ============================================================
# 1. Configuration
# ============================================================

IMAGE_PATH = "raw_captures/desk_objects1.jpeg"
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------------------------------------------------
# IMPORTANT:
# Enter the actual number of physical objects you placed
# in desk_objects.jpg.
#
# Task 0 required at least 3 objects.
# ------------------------------------------------------------

REAL_OBJECT_COUNT = 3


# ============================================================
# 2. Load image
# ============================================================

image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError(
        f"Could not load image: {IMAGE_PATH}"
    )

print("=" * 60)
print("TASK 5 — EDGE AND CONTOUR DETECTION")
print("=" * 60)

print(f"Actual physical object count: {REAL_OBJECT_COUNT}")


# ============================================================
# 3. Preprocessing
# ============================================================

gray = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2GRAY
)

# Gaussian smoothing reduces small noise before Canny
blurred = cv2.GaussianBlur(
    gray,
    (5, 5),
    0
)


# ============================================================
# PART A
# Initial Canny thresholds: 50, 150
# ============================================================

initial_lower = 50
initial_upper = 150

edges_initial = cv2.Canny(
    blurred,
    initial_lower,
    initial_upper
)

cv2.imwrite(
    os.path.join(
        OUTPUT_DIR,
        "task5_canny_50_150.jpg"
    ),
    edges_initial
)


# ============================================================
# Function to count meaningful closed contours
# ============================================================

def detect_object_contours(
    edge_image,
    min_area
):
    """
    Detect external contours from a Canny edge image.

    Morphological closing connects small gaps in object
    boundaries so that otherwise-open contours can become
    closed.
    """

    kernel = np.ones(
        (5, 5),
        np.uint8
    )

    closed_edges = cv2.morphologyEx(
        edge_image,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    contours, hierarchy = cv2.findContours(
        closed_edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    valid_contours = []

    for contour in contours:

        area = cv2.contourArea(contour)

        if area >= min_area:
            valid_contours.append(contour)

    return valid_contours, closed_edges


# ============================================================
# 4. Determine minimum contour area
# ============================================================

image_area = image.shape[0] * image.shape[1]

# Ignore tiny noise regions.
# This is a starting value rather than a universal threshold.
min_area = max(
    500,
    image_area * 0.001
)


# ============================================================
# 5. Initial contour count
# ============================================================

initial_contours, initial_closed = detect_object_contours(
    edges_initial,
    min_area
)

initial_count = len(initial_contours)

print("\nPART A — INITIAL CANNY")
print(
    f"Thresholds: ({initial_lower}, {initial_upper})"
)
print(
    f"Detected candidate contours: {initial_count}"
)


# ============================================================
# 6. Automatically tune Canny thresholds
# ============================================================

# We search through different threshold combinations.
#
# The search is performed in a fixed order, and the first
# exact match is selected.
#
# This gives us a reproducible "number of tries".

candidate_lower_values = [
    20, 30, 40, 50, 60, 70, 80,
    90, 100, 110, 120, 130
]

candidate_upper_values = [
    80, 100, 120, 150, 180, 200,
    220, 240, 255
]

best_result = None
tries = 0

for lower in candidate_lower_values:

    for upper in candidate_upper_values:

        if upper <= lower:
            continue

        tries += 1

        edges = cv2.Canny(
            blurred,
            lower,
            upper
        )

        contours, closed_edges = detect_object_contours(
            edges,
            min_area
        )

        count = len(contours)

        # Exact match
        if count == REAL_OBJECT_COUNT:

            best_result = {
                "lower": lower,
                "upper": upper,
                "count": count,
                "contours": contours,
                "edges": edges,
                "closed_edges": closed_edges,
                "tries": tries
            }

            break

    if best_result is not None:
        break


# ============================================================
# 7. If exact match was not found, choose closest result
# ============================================================

if best_result is None:

    best_difference = float("inf")

    for lower in candidate_lower_values:

        for upper in candidate_upper_values:

            if upper <= lower:
                continue

            tries += 1

            edges = cv2.Canny(
                blurred,
                lower,
                upper
            )

            contours, closed_edges = detect_object_contours(
                edges,
                min_area
            )

            count = len(contours)

            difference = abs(
                count - REAL_OBJECT_COUNT
            )

            if difference < best_difference:

                best_difference = difference

                best_result = {
                    "lower": lower,
                    "upper": upper,
                    "count": count,
                    "contours": contours,
                    "edges": edges,
                    "closed_edges": closed_edges,
                    "tries": tries
                }


# ============================================================
# 8. Extract final Canny result
# ============================================================

final_lower = best_result["lower"]
final_upper = best_result["upper"]

final_edges = best_result["edges"]

final_contours = best_result["contours"]

final_count = best_result["count"]

tries_used = best_result["tries"]


print("\nPART A — TUNED CANNY")
print(
    f"Final lower threshold: {final_lower}"
)
print(
    f"Final upper threshold: {final_upper}"
)
print(
    f"Detected contour count: {final_count}"
)
print(
    f"Actual object count: {REAL_OBJECT_COUNT}"
)
print(
    f"Number of tries: {tries_used}"
)


# ============================================================
# 9. Save final Canny edge image
# ============================================================

cv2.imwrite(
    os.path.join(
        OUTPUT_DIR,
        "task5_tuned_canny.jpg"
    ),
    final_edges
)


# ============================================================
# PART B — findContours()
# ============================================================

# Use the final Canny image as the input for contour detection.
#
# We already obtained final_contours above using
# cv2.findContours().


# ============================================================
# 10. Draw contours on original image
# ============================================================

annotated = image.copy()

for index, contour in enumerate(
    final_contours,
    start=1
):

    cv2.drawContours(
        annotated,
        [contour],
        -1,
        (0, 255, 0),
        3
    )

    # Calculate center using moments
    moments = cv2.moments(contour)

    if moments["m00"] != 0:

        cx = int(
            moments["m10"] / moments["m00"]
        )

        cy = int(
            moments["m01"] / moments["m00"]
        )

        cv2.putText(
            annotated,
            str(index),
            (cx, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            3,
            cv2.LINE_AA
        )


# ============================================================
# 11. Add count information
# ============================================================

text = (
    f"Detected: {final_count} | "
    f"Actual: {REAL_OBJECT_COUNT}"
)

cv2.rectangle(
    annotated,
    (0, 0),
    (500, 55),
    (255, 255, 255),
    -1
)

cv2.putText(
    annotated,
    text,
    (10, 38),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.9,
    (0, 0, 0),
    2,
    cv2.LINE_AA
)


# ============================================================
# 12. Save annotated image
# ============================================================

annotated_path = os.path.join(
    OUTPUT_DIR,
    "task5_contours.jpg"
)

cv2.imwrite(
    annotated_path,
    annotated
)


# ============================================================
# 13. Create raw vs annotated comparison
# ============================================================

raw_resized = cv2.resize(
    image,
    (600, int(image.shape[0] * 600 / image.shape[1]))
)

annotated_resized = cv2.resize(
    annotated,
    (
        600,
        int(image.shape[0] * 600 / image.shape[1])
    )
)

comparison = np.hstack([
    raw_resized,
    annotated_resized
])


# Labels
cv2.rectangle(
    comparison,
    (0, 0),
    (300, 45),
    (255, 255, 255),
    -1
)

cv2.putText(
    comparison,
    "RAW IMAGE",
    (10, 30),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (0, 0, 0),
    2,
    cv2.LINE_AA
)

width_comparison = comparison.shape[1]

cv2.rectangle(
    comparison,
    (600, 0),
    (900, 45),
    (255, 255, 255),
    -1
)

cv2.putText(
    comparison,
    "ANNOTATED",
    (610, 30),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (0, 0, 0),
    2,
    cv2.LINE_AA
)


# ============================================================
# 14. Save comparison
# ============================================================

cv2.imwrite(
    os.path.join(
        OUTPUT_DIR,
        "task5_raw_vs_annotated.jpg"
    ),
    comparison
)


# ============================================================
# 15. Determine match status
# ============================================================

if final_count == REAL_OBJECT_COUNT:

    status = "MATCH"

else:

    status = "MISMATCH"


# ============================================================
# 16. Save complete results
# ============================================================

results_path = os.path.join(
    OUTPUT_DIR,
    "task5_results.txt"
)

with open(results_path, "w") as f:

    f.write(
        "TASK 5 — EDGE AND CONTOUR DETECTION\n"
    )

    f.write("=" * 60 + "\n\n")

    f.write("PART A — CANNY EDGE DETECTION\n\n")

    f.write(
        f"Initial thresholds: "
        f"({initial_lower}, {initial_upper})\n"
    )

    f.write(
        f"Initial detected contours: "
        f"{initial_count}\n\n"
    )

    f.write(
        f"Final lower threshold: "
        f"{final_lower}\n"
    )

    f.write(
        f"Final upper threshold: "
        f"{final_upper}\n"
    )

    f.write(
        f"Final detected contours: "
        f"{final_count}\n"
    )

    f.write(
        f"Actual object count: "
        f"{REAL_OBJECT_COUNT}\n"
    )

    f.write(
        f"Threshold tries: "
        f"{tries_used}\n\n"
    )

    f.write(
        "PART B — CONTOUR DETECTION\n\n"
    )

    f.write(
        "Method: cv2.findContours()\n"
    )

    f.write(
        f"Detected objects/contours: "
        f"{final_count}\n"
    )

    f.write(
        f"Actual objects: "
        f"{REAL_OBJECT_COUNT}\n"
    )

    f.write(
        f"Result: {status}\n\n"
    )

    if status == "MATCH":

        f.write(
            "The detected contour count matches "
            "the actual number of objects.\n"
        )

    else:

        f.write(
            "The detected contour count does not "
            "match the actual number of objects.\n\n"
        )

        f.write(
            "Possible causes include:\n"
        )

        f.write(
            "- Shadows\n"
        )

        f.write(
            "- Touching objects\n"
        )

        f.write(
            "- Reflections\n"
        )

        f.write(
            "- Broken object boundaries\n"
        )

        f.write(
            "- Background texture\n"
        )


# ============================================================
# 17. Completion
# ============================================================

print("\n" + "=" * 60)
print("TASK 5 COMPLETED")
print("=" * 60)

print("\nOutputs saved:")

print(
    "outputs/task5_canny_50_150.jpg"
)

print(
    "outputs/task5_tuned_canny.jpg"
)

print(
    "outputs/task5_contours.jpg"
)

print(
    "outputs/task5_raw_vs_annotated.jpg"
)

print(
    "outputs/task5_results.txt"
)

print("\nFinal result:")
print(
    f"Actual objects   : {REAL_OBJECT_COUNT}"
)
print(
    f"Detected contours: {final_count}"
)
print(
    f"Thresholds       : ({final_lower}, {final_upper})"
)
print(
    f"Tries             : {tries_used}"
)
print(
    f"Status             : {status}"
)