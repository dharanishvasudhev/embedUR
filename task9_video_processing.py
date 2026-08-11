import cv2
import os


# ============================================================
# 1. Configuration
# ============================================================

INPUT_VIDEO = "raw_captures/video.mp4"

OUTPUT_DIR = "outputs"

OUTPUT_VIDEO = os.path.join(
    OUTPUT_DIR,
    "task9_edge_detection_output.mp4"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. Open input video
# ============================================================

cap = cv2.VideoCapture(INPUT_VIDEO)

if not cap.isOpened():
    raise FileNotFoundError(
        f"Could not open video: {INPUT_VIDEO}"
    )


# ============================================================
# 3. Read video properties
# ============================================================

fps = cap.get(
    cv2.CAP_PROP_FPS
)

frame_width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

frame_height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)

total_frames = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)

duration = (
    total_frames / fps
    if fps > 0
    else 0
)


print("=" * 65)
print("TASK 9 — REAL-TIME VIDEO PROCESSING")
print("=" * 65)

print(
    f"\nInput video: {INPUT_VIDEO}"
)

print(
    f"Resolution: {frame_width} x {frame_height}"
)

print(
    f"FPS: {fps:.2f}"
)

print(
    f"Total frames: {total_frames}"
)

print(
    f"Duration: {duration:.2f} seconds"
)


# ============================================================
# 4. Handle unusual FPS
# ============================================================

if fps <= 0:
    print(
        "\nWarning: FPS could not be read."
    )

    fps = 30.0

    print(
        "Using fallback FPS = 30."
    )


# ============================================================
# 5. Create VideoWriter
# ============================================================

# MP4V is widely supported for .mp4 output.

fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)

writer = cv2.VideoWriter(
    OUTPUT_VIDEO,
    fourcc,
    fps,
    (frame_width, frame_height),
    True
)

if not writer.isOpened():
    cap.release()

    raise RuntimeError(
        "Could not create output video."
    )


# ============================================================
# 6. Frame processing
# ============================================================

frame_count = 0

processed_frames = 0


while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1


    # --------------------------------------------------------
    # Convert current frame to grayscale
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )


    # --------------------------------------------------------
    # Slight Gaussian smoothing
    # --------------------------------------------------------

    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )


    # --------------------------------------------------------
    # Canny edge detection
    # --------------------------------------------------------

    edges = cv2.Canny(
        blurred,
        50,
        150
    )


    # --------------------------------------------------------
    # Convert single-channel edges back to BGR
    # --------------------------------------------------------

    output_frame = cv2.cvtColor(
        edges,
        cv2.COLOR_GRAY2BGR
    )


    # --------------------------------------------------------
    # Add information overlay
    # --------------------------------------------------------

    cv2.putText(
        output_frame,
        "TASK 9 - CANNY EDGE DETECTION",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        output_frame,
        f"Frame: {frame_count}/{total_frames}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        output_frame,
        "Canny thresholds: 50, 150",
        (20, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )


    # --------------------------------------------------------
    # Write processed frame
    # --------------------------------------------------------

    writer.write(
        output_frame
    )

    processed_frames += 1


# ============================================================
# 7. Release resources
# ============================================================

cap.release()

writer.release()


# ============================================================
# 8. Verify output
# ============================================================

output_size = os.path.getsize(
    OUTPUT_VIDEO
)

print("\n" + "=" * 65)
print("TASK 9 COMPLETED")
print("=" * 65)

print(
    f"\nFrames processed: {processed_frames}"
)

print(
    f"Output FPS: {fps:.2f}"
)

print(
    f"Output resolution: "
    f"{frame_width} x {frame_height}"
)

print(
    f"Output video: {OUTPUT_VIDEO}"
)

print(
    f"Output file size: "
    f"{output_size / (1024 * 1024):.2f} MB"
)

print("\nProcessing pipeline:")

print(
    "Video frame"
)

print(
    "    ↓"
)

print(
    "Grayscale"
)

print(
    "    ↓"
)

print(
    "Gaussian Blur"
)

print(
    "    ↓"
)

print(
    "Canny Edge Detection"
)

print(
    "    ↓"
)

print(
    "BGR conversion"
)

print(
    "    ↓"
)

print(
    "VideoWriter"
)

print(
    "    ↓"
)

print(
    "Output MP4"
)