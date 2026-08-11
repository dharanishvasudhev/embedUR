import cv2
import os


# ============================================================
# 1. Configuration
# ============================================================

IMAGE_PATH = "raw_captures/face.jpg"

OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. Load Haar Cascade
# ============================================================

cascade_path = cv2.data.haarcascades + \
    "haarcascade_frontalface_default.xml"

face_cascade = cv2.CascadeClassifier(
    cascade_path
)

if face_cascade.empty():
    raise RuntimeError(
        "Could not load Haar cascade classifier."
    )


# ============================================================
# 3. Load image
# ============================================================

image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError(
        f"Could not load image: {IMAGE_PATH}"
    )


original = image.copy()


print("=" * 65)
print("TASK 8 — FACE DETECTION")
print("=" * 65)

print(f"\nInput image: {IMAGE_PATH}")

height, width = image.shape[:2]

print(
    f"Image resolution: {width} x {height}"
)


# ============================================================
# 4. Convert to grayscale
# ============================================================

gray = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2GRAY
)

cv2.imwrite(
    os.path.join(
        OUTPUT_DIR,
        "task8_grayscale.jpg"
    ),
    gray
)


# ============================================================
# 5. Improve local contrast
# ============================================================

# CLAHE is used only as an additional detection attempt.
# The original grayscale image is still used for the
# primary detector.

clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)

gray_clahe = clahe.apply(gray)

cv2.imwrite(
    os.path.join(
        OUTPUT_DIR,
        "task8_clahe.jpg"
    ),
    gray_clahe
)


# ============================================================
# 6. Run Haar detector on original grayscale image
# ============================================================

faces_original = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30)
)


print("\nPRIMARY HAAR DETECTION")
print("-" * 40)

print(
    f"Faces detected: {len(faces_original)}"
)


# ============================================================
# 7. Run second detection using CLAHE
# ============================================================

faces_clahe = face_cascade.detectMultiScale(
    gray_clahe,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30)
)


print(
    f"Faces detected with CLAHE: "
    f"{len(faces_clahe)}"
)


# ============================================================
# 8. Draw primary detections
# ============================================================

annotated = image.copy()

for i, (x, y, w, h) in enumerate(
    faces_original,
    start=1
):

    cv2.rectangle(
        annotated,
        (x, y),
        (x + w, y + h),
        (255, 255, 255),
        3
    )

    cv2.putText(
        annotated,
        f"Face {i}",
        (x, max(y - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )


# ============================================================
# 9. Save annotated image
# ============================================================

cv2.imwrite(
    os.path.join(
        OUTPUT_DIR,
        "task8_face_detection.jpg"
    ),
    annotated
)


# ============================================================
# 10. Save individual face crops
# ============================================================

face_crop_paths = []

for i, (x, y, w, h) in enumerate(
    faces_original,
    start=1
):

    # Add some context around the detected face
    padding = int(
        0.20 * max(w, h)
    )

    x1 = max(
        0,
        x - padding
    )

    y1 = max(
        0,
        y - padding
    )

    x2 = min(
        width,
        x + w + padding
    )

    y2 = min(
        height,
        y + h + padding
    )

    face_crop = image[
        y1:y2,
        x1:x2
    ]

    crop_path = os.path.join(
        OUTPUT_DIR,
        f"task8_face_{i}_crop.jpg"
    )

    cv2.imwrite(
        crop_path,
        face_crop
    )

    face_crop_paths.append(
        crop_path
    )


# ============================================================
# 11. Create zoomed face inspection images
# ============================================================

for i, (x, y, w, h) in enumerate(
    faces_original,
    start=1
):

    padding = int(
        0.30 * max(w, h)
    )

    x1 = max(
        0,
        x - padding
    )

    y1 = max(
        0,
        y - padding
    )

    x2 = min(
        width,
        x + w + padding
    )

    y2 = min(
        height,
        y + h + padding
    )

    face_zoom = image[
        y1:y2,
        x1:x2
    ]

    # Enlarge the region for pixel-level inspection
    zoom_width = 600

    zoom_height = int(
        face_zoom.shape[0] *
        zoom_width /
        face_zoom.shape[1]
    )

    face_zoom = cv2.resize(
        face_zoom,
        (zoom_width, zoom_height),
        interpolation=cv2.INTER_CUBIC
    )

    cv2.imwrite(
        os.path.join(
            OUTPUT_DIR,
            f"task8_face_{i}_zoom.jpg"
        ),
        face_zoom
    )


# ============================================================
# 12. Save CLAHE detections separately
# ============================================================

clahe_annotated = image.copy()

for i, (x, y, w, h) in enumerate(
    faces_clahe,
    start=1
):

    cv2.rectangle(
        clahe_annotated,
        (x, y),
        (x + w, y + h),
        (255, 255, 255),
        3
    )

    cv2.putText(
        clahe_annotated,
        f"Face {i}",
        (x, max(y - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )


cv2.imwrite(
    os.path.join(
        OUTPUT_DIR,
        "task8_clahe_detection.jpg"
    ),
    clahe_annotated
)


# ============================================================
# 13. Create comparison image
# ============================================================

display_width = 650

display_height = int(
    height *
    display_width /
    width
)

original_display = cv2.resize(
    annotated,
    (display_width, display_height)
)

clahe_display = cv2.resize(
    clahe_annotated,
    (display_width, display_height)
)


cv2.rectangle(
    original_display,
    (0, 0),
    (360, 50),
    (255, 255, 255),
    -1
)

cv2.putText(
    original_display,
    "ORIGINAL GRAYSCALE DETECTION",
    (10, 35),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.65,
    (0, 0, 0),
    2,
    cv2.LINE_AA
)


cv2.rectangle(
    clahe_display,
    (0, 0),
    (330, 50),
    (255, 255, 255),
    -1
)

cv2.putText(
    clahe_display,
    "CLAHE DETECTION",
    (10, 35),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (0, 0, 0),
    2,
    cv2.LINE_AA
)


comparison = cv2.hconcat([
    original_display,
    clahe_display
])


cv2.imwrite(
    os.path.join(
        OUTPUT_DIR,
        "task8_detection_comparison.jpg"
    ),
    comparison
)


# ============================================================
# 14. Analyze detected faces
# ============================================================

results_path = os.path.join(
    OUTPUT_DIR,
    "task8_results.txt"
)

with open(results_path, "w") as f:

    f.write(
        "TASK 8 — FACE DETECTION\n"
    )

    f.write("=" * 65 + "\n\n")

    f.write(
        f"Input image: {IMAGE_PATH}\n"
    )

    f.write(
        f"Resolution: {width} x {height}\n\n"
    )

    f.write(
        "Haar cascade:\n"
    )

    f.write(
        "haarcascade_frontalface_default.xml\n\n"
    )

    f.write(
        "Detection parameters:\n"
    )

    f.write(
        "scaleFactor = 1.1\n"
    )

    f.write(
        "minNeighbors = 5\n"
    )

    f.write(
        "minSize = (30, 30)\n\n"
    )

    f.write(
        f"Faces detected on original grayscale: "
        f"{len(faces_original)}\n"
    )

    f.write(
        f"Faces detected with CLAHE: "
        f"{len(faces_clahe)}\n\n"
    )

    f.write(
        "PRIMARY DETECTIONS\n"
    )

    f.write(
        "-" * 65 + "\n"
    )

    if len(faces_original) == 0:

        f.write(
            "No faces were detected.\n"
        )

    else:

        for i, (x, y, w, h) in enumerate(
            faces_original,
            start=1
        ):

            f.write(
                f"Face {i}: "
                f"x={x}, y={y}, "
                f"width={w}, height={h}\n"
            )

            f.write(
                f"Face {i} area: "
                f"{w * h} pixels\n"
            )

            f.write(
                f"Face {i} center: "
                f"({x + w // 2}, "
                f"{y + h // 2})\n\n"
            )

    f.write(
        "INTERPRETATION\n"
    )

    f.write(
        "-" * 65 + "\n"
    )

    if len(faces_original) == 0:

        f.write(
            "The Haar cascade did not detect a face in "
            "the normal-light photograph. Inspect the "
            "input image and task8_grayscale.jpg to "
            "determine whether lighting, face angle, "
            "occlusion, small face size, or image quality "
            "may have caused the detector to fail.\n"
        )

    else:

        f.write(
            "The Haar cascade detected one or more face "
            "regions. Inspect task8_face_detection.jpg "
            "and the corresponding zoom images to check "
            "whether every detection is a true face or "
            "whether any false positives are present.\n"
        )


# ============================================================
# 15. Completion
# ============================================================

print("\n" + "=" * 65)
print("TASK 8 COMPLETED")
print("=" * 65)

print(
    f"\nOriginal grayscale detections: "
    f"{len(faces_original)}"
)

print(
    f"CLAHE detections: "
    f"{len(faces_clahe)}"
)

print("\nSaved outputs:")

print(
    "  outputs/task8_grayscale.jpg"
)

print(
    "  outputs/task8_clahe.jpg"
)

print(
    "  outputs/task8_face_detection.jpg"
)

print(
    "  outputs/task8_clahe_detection.jpg"
)

print(
    "  outputs/task8_detection_comparison.jpg"
)

for path in face_crop_paths:
    print(
        f"  {path}"
    )

for i in range(
    1,
    len(faces_original) + 1
):

    print(
        f"  outputs/task8_face_{i}_zoom.jpg"
    )

print(
    "  outputs/task8_results.txt"
)