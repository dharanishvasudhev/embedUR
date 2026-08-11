import cv2
import numpy as np
import os
import glob


# ============================================================
# 1. Configuration
# ============================================================

CALIBRATION_DIR = "raw_captures/calibration"

SCENE_IMAGE = "raw_captures/calibration_scene.jpeg"

OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# Checkerboard dimensions:


CHECKERBOARD = (7, 7)


# ============================================================
# 2. Prepare 3D object points
# ============================================================

# The checkerboard is assumed to lie on the Z = 0 plane.

object_points_template = np.zeros(
    (CHECKERBOARD[0] * CHECKERBOARD[1], 3),
    np.float32
)

object_points_template[:, :2] = np.mgrid[
    0:CHECKERBOARD[0],
    0:CHECKERBOARD[1]
].T.reshape(-1, 2)


# ============================================================
# 3. Storage
# ============================================================

object_points = []

image_points = []

successful_images = []

failed_images = []

corner_visualizations = []


# ============================================================
# 4. Find calibration images
# ============================================================

image_extensions = [
    "*.jpg",
    "*.jpeg",
    "*.png",
    "*.JPG",
    "*.JPEG",
    "*.PNG"
]

calibration_images = []

for extension in image_extensions:

    calibration_images.extend(
        glob.glob(
            os.path.join(
                CALIBRATION_DIR,
                extension
            )
        )
    )

calibration_images = sorted(
    list(set(calibration_images))
)


print("=" * 70)
print("TASK 10 — CAMERA CALIBRATION AND LENS UNDISTORTION")
print("=" * 70)

print(
    f"\nCalibration images found: "
    f"{len(calibration_images)}"
)

if len(calibration_images) < 8:

    raise RuntimeError(
        "Too few calibration images. "
        "Capture at least 8 good checkerboard images; "
        "12–15 are recommended."
    )


# ============================================================
# 5. Process each checkerboard image
# ============================================================

image_size = None

criteria = (
    cv2.TERM_CRITERIA_EPS +
    cv2.TERM_CRITERIA_MAX_ITER,
    30,
    0.001
)


for index, image_path in enumerate(
    calibration_images,
    start=1
):

    print(
        f"\n[{index}/{len(calibration_images)}] "
        f"Processing: {os.path.basename(image_path)}"
    )

    image = cv2.imread(image_path)

    if image is None:

        print("  ERROR: Could not read image.")

        failed_images.append(
            (
                image_path,
                "Image could not be read"
            )
        )

        continue


    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )


    if image_size is None:

        image_size = (
            gray.shape[1],
            gray.shape[0]
        )

    elif image_size != (
        gray.shape[1],
        gray.shape[0]
    ):

        print(
            "  FAILED: Image resolution differs "
            "from the first calibration image."
        )

        failed_images.append(
            (
                image_path,
                "Different image resolution"
            )
        )

        continue


    # --------------------------------------------------------
    # Find chessboard corners
    # --------------------------------------------------------

    found, corners = cv2.findChessboardCorners(
        gray,
        CHECKERBOARD,
        cv2.CALIB_CB_ADAPTIVE_THRESH +
        cv2.CALIB_CB_NORMALIZE_IMAGE
    )


    if not found:

        print(
            "  FAILED: Chessboard corners not detected."
        )

        failed_images.append(
            (
                image_path,
                "Chessboard corners not detected"
            )
        )

        continue


    # --------------------------------------------------------
    # Refine corner locations
    # --------------------------------------------------------

    refined_corners = cv2.cornerSubPix(
        gray,
        corners,
        (11, 11),
        (-1, -1),
        criteria
    )


    # --------------------------------------------------------
    # Store calibration points
    # --------------------------------------------------------

    object_points.append(
        object_points_template.copy()
    )

    image_points.append(
        refined_corners
    )

    successful_images.append(
        image_path
    )


    # --------------------------------------------------------
    # Draw detected corners
    # --------------------------------------------------------

    visualization = image.copy()

    cv2.drawChessboardCorners(
        visualization,
        CHECKERBOARD,
        refined_corners,
        found
    )

    output_name = (
        "task10_corners_" +
        os.path.basename(image_path)
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        output_name
    )

    cv2.imwrite(
        output_path,
        visualization
    )

    corner_visualizations.append(
        output_path
    )


    print(
        "  SUCCESS: 54 corners detected "
        "and refined."
    )


# ============================================================
# 6. Check successful images
# ============================================================

print("\n" + "=" * 70)
print("CORNER DETECTION SUMMARY")
print("=" * 70)

print(
    f"Total images: {len(calibration_images)}"
)

print(
    f"Successful: {len(successful_images)}"
)

print(
    f"Discarded: {len(failed_images)}"
)


if len(successful_images) < 8:

    raise RuntimeError(
        "Not enough successful calibration images. "
        "Capture additional checkerboard images."
    )


# ============================================================
# 7. Camera calibration
# ============================================================

print("\n")
print("=" * 70)
print("RUNNING cv2.calibrateCamera()")
print("=" * 70)


reprojection_error, camera_matrix, distortion_coefficients, rvecs, tvecs = cv2.calibrateCamera(
    object_points,
    image_points,
    image_size,
    None,
    None
)


# ============================================================
# 8. Extract camera parameters
# ============================================================

fx = camera_matrix[0, 0]

fy = camera_matrix[1, 1]

cx = camera_matrix[0, 2]

cy = camera_matrix[1, 2]


# ============================================================
# 9. Calculate manual overall reprojection error
# ============================================================

total_error = 0

total_points = 0

per_image_errors = []


for i in range(
    len(object_points)
):

    projected_points, _ = cv2.projectPoints(
        object_points[i],
        rvecs[i],
        tvecs[i],
        camera_matrix,
        distortion_coefficients
    )


    error = cv2.norm(
        image_points[i],
        projected_points,
        cv2.NORM_L2
    )

    number_of_points = len(
        projected_points
    )

    total_error += error

    total_points += number_of_points


    mean_image_error = (
        error / number_of_points
    )

    per_image_errors.append(
        mean_image_error
    )


manual_reprojection_error = (
    total_error / total_points
)


# ============================================================
# 10. Print calibration results
# ============================================================

print("\nCAMERA MATRIX")
print("-" * 50)

print(camera_matrix)


print("\nDISTORTION COEFFICIENTS")
print("-" * 50)

print(distortion_coefficients)


print("\nCAMERA PARAMETERS")
print("-" * 50)

print(
    f"fx = {fx:.4f}"
)

print(
    f"fy = {fy:.4f}"
)

print(
    f"cx = {cx:.4f}"
)

print(
    f"cy = {cy:.4f}"
)


print("\nREPROJECTION ERROR")
print("-" * 50)

print(
    f"OpenCV calibration error = "
    f"{reprojection_error:.6f}"
)

print(
    f"Manual overall error = "
    f"{manual_reprojection_error:.6f}"
)


# ============================================================
# 11. Save camera parameters
# ============================================================

parameters_file = os.path.join(
    OUTPUT_DIR,
    "task10_camera_parameters.txt"
)


with open(
    parameters_file,
    "w"
) as f:

    f.write(
        "TASK 10 — CAMERA CALIBRATION PARAMETERS\n"
    )

    f.write("=" * 70 + "\n\n")


    f.write(
        "Checkerboard:\n"
    )

    f.write(
        "7 x 7 internal corners\n\n"
    )


    f.write(
        "Calibration images:\n"
    )

    f.write(
        f"Total images: "
        f"{len(calibration_images)}\n"
    )

    f.write(
        f"Successful images: "
        f"{len(successful_images)}\n"
    )

    f.write(
        f"Discarded images: "
        f"{len(failed_images)}\n\n"
    )


    f.write(
        "Camera Matrix:\n"
    )

    f.write(
        str(camera_matrix)
    )

    f.write(
        "\n\n"
    )


    f.write(
        "Distortion coefficients:\n"
    )

    f.write(
        str(distortion_coefficients)
    )

    f.write(
        "\n\n"
    )


    f.write(
        "Focal lengths:\n"
    )

    f.write(
        f"fx = {fx:.6f}\n"
    )

    f.write(
        f"fy = {fy:.6f}\n\n"
    )


    f.write(
        "Principal point:\n"
    )

    f.write(
        f"cx = {cx:.6f}\n"
    )

    f.write(
        f"cy = {cy:.6f}\n\n"
    )


    f.write(
        "Reprojection error:\n"
    )

    f.write(
        f"OpenCV = "
        f"{reprojection_error:.6f}\n"
    )

    f.write(
        f"Manual = "
        f"{manual_reprojection_error:.6f}\n\n"
    )


# ============================================================
# 12. Save failed image report
# ============================================================

failed_report = os.path.join(
    OUTPUT_DIR,
    "task10_discarded_images.txt"
)


with open(
    failed_report,
    "w"
) as f:

    f.write(
        "TASK 10 — DISCARDED CALIBRATION IMAGES\n"
    )

    f.write("=" * 70 + "\n\n")


    if len(failed_images) == 0:

        f.write(
            "No images were discarded.\n"
        )

    else:

        for image_path, reason in failed_images:

            f.write(
                f"{os.path.basename(image_path)}\n"
            )

            f.write(
                f"Reason: {reason}\n\n"
            )


# ============================================================
# 13. Create undistorted scene
# ============================================================

scene = cv2.imread(
    SCENE_IMAGE
)

if scene is None:

    print(
        "\nWARNING: calibration_scene.jpeg "
        "was not found."
    )

    print(
        "Calibration completed, but "
        "undistortion was skipped."
    )

else:

    print("\n")
    print("=" * 70)
    print("UNDISTORTING ORDINARY SCENE")
    print("=" * 70)


    scene_gray = cv2.cvtColor(
        scene,
        cv2.COLOR_BGR2GRAY
    )


    # --------------------------------------------------------
    # Undistort using cv2.undistort()
    # --------------------------------------------------------

    undistorted = cv2.undistort(
        scene,
        camera_matrix,
        distortion_coefficients
    )


    # --------------------------------------------------------
    # Save original and undistorted images
    # --------------------------------------------------------

    cv2.imwrite(
        os.path.join(
            OUTPUT_DIR,
            "task10_scene_original.jpg"
        ),
        scene
    )


    cv2.imwrite(
        os.path.join(
            OUTPUT_DIR,
            "task10_scene_undistorted.jpg"
        ),
        undistorted
    )


    # --------------------------------------------------------
    # Side-by-side comparison
    # --------------------------------------------------------

    comparison = np.hstack([
        scene,
        undistorted
    ])


    cv2.imwrite(
        os.path.join(
            OUTPUT_DIR,
            "task10_undistortion_comparison.jpg"
        ),
        comparison
    )


    print(
        "Saved original scene."
    )

    print(
        "Saved undistorted scene."
    )

    print(
        "Saved side-by-side comparison."
    )


# ============================================================
# 14. Final report
# ============================================================

final_report = os.path.join(
    OUTPUT_DIR,
    "task10_final_report.txt"
)


with open(
    final_report,
    "w"
) as f:

    f.write(
        "TASK 10 — CAMERA CALIBRATION AND "
        "LENS UNDISTORTION\n"
    )

    f.write("=" * 70 + "\n\n")


    f.write(
        "CALIBRATION DATA\n"
    )

    f.write("-" * 70 + "\n")

    f.write(
        f"Checkerboard: 9 x 6 internal corners\n"
    )

    f.write(
        f"Images captured: "
        f"{len(calibration_images)}\n"
    )

    f.write(
        f"Images accepted: "
        f"{len(successful_images)}\n"
    )

    f.write(
        f"Images discarded: "
        f"{len(failed_images)}\n\n"
    )


    f.write(
        "CAMERA PARAMETERS\n"
    )

    f.write("-" * 70 + "\n")

    f.write(
        f"fx = {fx:.6f}\n"
    )

    f.write(
        f"fy = {fy:.6f}\n"
    )

    f.write(
        f"cx = {cx:.6f}\n"
    )

    f.write(
        f"cy = {cy:.6f}\n\n"
    )


    f.write(
        "REPROJECTION ERROR\n"
    )

    f.write("-" * 70 + "\n")

    f.write(
        f"Overall error = "
        f"{manual_reprojection_error:.6f} pixels\n\n"
    )


    f.write(
        "DISCARDED IMAGE REASONS\n"
    )

    f.write("-" * 70 + "\n")


    if len(failed_images) == 0:

        f.write(
            "No images discarded.\n"
        )

    else:

        for image_path, reason in failed_images:

            f.write(
                f"{os.path.basename(image_path)}: "
                f"{reason}\n"
            )


    f.write("\n")

    f.write(
        "INTERPRETATION\n"
    )

    f.write("-" * 70 + "\n")

    f.write(
        "Camera calibration estimates the camera's "
        "intrinsic parameters and lens distortion "
        "from multiple views of a known checkerboard. "
        "The resulting parameters were used with "
        "cv2.undistort() to reduce lens distortion "
        "in the ordinary scene.\n"
    )


# ============================================================
# 15. Completion
# ============================================================

print("\n")
print("=" * 70)
print("TASK 10 COMPLETED")
print("=" * 70)

print(
    f"\nAccepted calibration images: "
    f"{len(successful_images)}"
)

print(
    f"Discarded images: "
    f"{len(failed_images)}"
)

print(
    f"\nfx = {fx:.4f}"
)

print(
    f"fy = {fy:.4f}"
)

print(
    f"cx = {cx:.4f}"
)

print(
    f"cy = {cy:.4f}"
)

print(
    f"\nOverall reprojection error = "
    f"{manual_reprojection_error:.6f} pixels"
)

print("\nSaved reports:")

print(
    "  outputs/task10_camera_parameters.txt"
)

print(
    "  outputs/task10_discarded_images.txt"
)

print(
    "  outputs/task10_final_report.txt"
)

print("\nSaved corner detections:")

for path in corner_visualizations:

    print(
        f"  {path}"
    )

if scene is not None:

    print("\nSaved undistortion results:")

    print(
        "  outputs/task10_scene_original.jpg"
    )

    print(
        "  outputs/task10_scene_undistorted.jpg"
    )

    print(
        "  outputs/task10_undistortion_comparison.jpg"
    )