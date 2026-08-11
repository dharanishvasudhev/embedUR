import cv2
import numpy as np
import os


# ============================================================
# 1. Configuration
# ============================================================

IMAGE1_PATH = "raw_captures/desk_objects1.jpeg"
IMAGE2_PATH = "raw_captures/desk_objects1_angle.jpeg"

OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. Load images
# ============================================================

image1 = cv2.imread(IMAGE1_PATH)
image2 = cv2.imread(IMAGE2_PATH)

if image1 is None:
    raise FileNotFoundError(
        f"Could not load: {IMAGE1_PATH}"
    )

if image2 is None:
    raise FileNotFoundError(
        f"Could not load: {IMAGE2_PATH}"
    )


print("=" * 65)
print("TASK 7 — FEATURE MATCHING ACROSS PHOTOS")
print("=" * 65)


# ============================================================
# 3. Convert to grayscale
# ============================================================

gray1 = cv2.cvtColor(
    image1,
    cv2.COLOR_BGR2GRAY
)

gray2 = cv2.cvtColor(
    image2,
    cv2.COLOR_BGR2GRAY
)


# ============================================================
# 4. Create ORB detector
# ============================================================

orb = cv2.ORB_create(
    nfeatures=3000,
    scaleFactor=1.2,
    nlevels=8
)


# ============================================================
# 5. Detect keypoints and descriptors
# ============================================================

keypoints1, descriptors1 = orb.detectAndCompute(
    gray1,
    None
)

keypoints2, descriptors2 = orb.detectAndCompute(
    gray2,
    None
)


print("\nORB KEYPOINTS")
print("-" * 40)

print(
    f"Original photo : {len(keypoints1)} keypoints"
)

print(
    f"New angle      : {len(keypoints2)} keypoints"
)


if descriptors1 is None or descriptors2 is None:
    raise RuntimeError(
        "ORB could not generate descriptors for one "
        "or both images."
    )


# ============================================================
# 6. Save keypoint visualizations
# ============================================================

keypoint_image1 = cv2.drawKeypoints(
    image1,
    keypoints1,
    None,
    color=None,
    flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS
)

keypoint_image2 = cv2.drawKeypoints(
    image2,
    keypoints2,
    None,
    color=None,
    flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS
)

cv2.imwrite(
    os.path.join(
        OUTPUT_DIR,
        "task7_keypoints_original.jpg"
    ),
    keypoint_image1
)

cv2.imwrite(
    os.path.join(
        OUTPUT_DIR,
        "task7_keypoints_angle2.jpg"
    ),
    keypoint_image2
)


# ============================================================
# 7. Brute Force Matcher
# ============================================================

# ORB produces binary descriptors.
# Therefore Hamming distance is appropriate.

bf = cv2.BFMatcher(
    cv2.NORM_HAMMING,
    crossCheck=False
)


# ============================================================
# 8. KNN matching
# ============================================================

knn_matches = bf.knnMatch(
    descriptors1,
    descriptors2,
    k=2
)


print(
    f"\nRaw KNN match pairs: {len(knn_matches)}"
)


# ============================================================
# 9. Lowe's ratio test
# ============================================================

good_matches = []

ratio_threshold = 0.75

for pair in knn_matches:

    if len(pair) < 2:
        continue

    m, n = pair

    if m.distance < ratio_threshold * n.distance:

        good_matches.append(m)


print(
    f"Good matches: {len(good_matches)}"
)


# ============================================================
# 10. Sort good matches by descriptor distance
# ============================================================

good_matches = sorted(
    good_matches,
    key=lambda match: match.distance
)


# ============================================================
# 11. Save ALL match visualization
# ============================================================

all_match_limit = min(
    len(knn_matches),
    100
)

all_simple_matches = []

for pair in knn_matches:

    if len(pair) >= 1:

        all_simple_matches.append(
            pair[0]
        )

all_simple_matches = sorted(
    all_simple_matches,
    key=lambda match: match.distance
)

all_simple_matches = all_simple_matches[
    :all_match_limit
]


all_match_image = cv2.drawMatches(
    image1,
    keypoints1,
    image2,
    keypoints2,
    all_simple_matches,
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

cv2.imwrite(
    os.path.join(
        OUTPUT_DIR,
        "task7_all_matches.jpg"
    ),
    all_match_image
)


# ============================================================
# 12. Save GOOD matches
# ============================================================

good_match_limit = min(
    len(good_matches),
    100
)

good_matches_display = good_matches[
    :good_match_limit
]


good_match_image = cv2.drawMatches(
    image1,
    keypoints1,
    image2,
    keypoints2,
    good_matches_display,
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

cv2.imwrite(
    os.path.join(
        OUTPUT_DIR,
        "task7_good_matches.jpg"
    ),
    good_match_image
)


# ============================================================
# 13. Save TOP 20 matches
# ============================================================

top_matches = good_matches[
    :min(20, len(good_matches))
]


top_match_image = cv2.drawMatches(
    image1,
    keypoints1,
    image2,
    keypoints2,
    top_matches,
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

cv2.imwrite(
    os.path.join(
        OUTPUT_DIR,
        "task7_top_matches.jpg"
    ),
    top_match_image
)


# ============================================================
# 14. Analyze strongest individual matches
# ============================================================

top_match_information = []

for rank, match in enumerate(
    top_matches,
    start=1
):

    kp1 = keypoints1[
        match.queryIdx
    ]

    kp2 = keypoints2[
        match.trainIdx
    ]

    x1, y1 = kp1.pt
    x2, y2 = kp2.pt

    top_match_information.append({
        "rank": rank,
        "distance": float(match.distance),
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2
    })


# ============================================================
# 15. Estimate geometric consistency
# ============================================================

inlier_matches = []

homography_inliers = 0

if len(good_matches) >= 4:

    src_points = np.float32([
        keypoints1[m.queryIdx].pt
        for m in good_matches
    ]).reshape(-1, 1, 2)

    dst_points = np.float32([
        keypoints2[m.trainIdx].pt
        for m in good_matches
    ]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(
        src_points,
        dst_points,
        cv2.RANSAC,
        5.0
    )

    if mask is not None:

        mask = mask.ravel()

        homography_inliers = int(
            np.sum(mask)
        )

        inlier_matches = [
            match
            for match, is_inlier
            in zip(good_matches, mask)
            if is_inlier
        ]


# ============================================================
# 16. Save geometrically consistent matches
# ============================================================

if len(inlier_matches) > 0:

    inlier_display = inlier_matches[
        :min(100, len(inlier_matches))
    ]

    inlier_image = cv2.drawMatches(
        image1,
        keypoints1,
        image2,
        keypoints2,
        inlier_display,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    cv2.imwrite(
        os.path.join(
            OUTPUT_DIR,
            "task7_ransac_inliers.jpg"
        ),
        inlier_image
    )


# ============================================================
# 17. Create side-by-side raw image comparison
# ============================================================

display_width = 600

display_height1 = int(
    image1.shape[0] *
    display_width /
    image1.shape[1]
)

display_height2 = int(
    image2.shape[0] *
    display_width /
    image2.shape[1]
)

display_height = min(
    display_height1,
    display_height2
)

display1 = cv2.resize(
    image1,
    (display_width, display_height)
)

display2 = cv2.resize(
    image2,
    (display_width, display_height)
)

comparison = np.hstack([
    display1,
    display2
])


# Labels

cv2.rectangle(
    comparison,
    (0, 0),
    (300, 50),
    (255, 255, 255),
    -1
)

cv2.putText(
    comparison,
    "ORIGINAL DESK",
    (10, 35),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (0, 0, 0),
    2,
    cv2.LINE_AA
)

cv2.rectangle(
    comparison,
    (display_width, 0),
    (display_width + 350, 50),
    (255, 255, 255),
    -1
)

cv2.putText(
    comparison,
    "NEW ANGLE",
    (display_width + 10, 35),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (0, 0, 0),
    2,
    cv2.LINE_AA
)

cv2.imwrite(
    os.path.join(
        OUTPUT_DIR,
        "task7_match_comparison.jpg"
    ),
    comparison
)


# ============================================================
# 18. Calculate match statistics
# ============================================================

if len(good_matches) > 0:

    distances = [
        match.distance
        for match in good_matches
    ]

    best_distance = min(distances)
    mean_distance = np.mean(distances)
    median_distance = np.median(distances)

else:

    best_distance = None
    mean_distance = None
    median_distance = None


if len(good_matches) > 0:

    inlier_ratio = (
        homography_inliers /
        len(good_matches)
    ) * 100

else:

    inlier_ratio = 0


# ============================================================
# 19. Save results
# ============================================================

results_path = os.path.join(
    OUTPUT_DIR,
    "task7_results.txt"
)

with open(results_path, "w") as f:

    f.write(
        "TASK 7 — FEATURE MATCHING ACROSS PHOTOS\n"
    )

    f.write("=" * 65 + "\n\n")

    f.write(
        "Images:\n"
    )

    f.write(
        "Original: raw_captures/desk_objects.jpg\n"
    )

    f.write(
        "New angle: raw_captures/desk_objects_angle2.jpg\n\n"
    )

    f.write(
        "ORB parameters:\n"
    )

    f.write(
        "nfeatures = 3000\n"
    )

    f.write(
        "scaleFactor = 1.2\n"
    )

    f.write(
        "nlevels = 8\n\n"
    )

    f.write(
        "Keypoints:\n"
    )

    f.write(
        f"Original image: {len(keypoints1)}\n"
    )

    f.write(
        f"New angle image: {len(keypoints2)}\n\n"
    )

    f.write(
        "Matching:\n"
    )

    f.write(
        "Matcher: Brute Force\n"
    )

    f.write(
        "Distance: Hamming\n"
    )

    f.write(
        f"Lowe ratio threshold: {ratio_threshold}\n"
    )

    f.write(
        f"Raw KNN pairs: {len(knn_matches)}\n"
    )

    f.write(
        f"Good matches: {len(good_matches)}\n\n"
    )

    f.write(
        "Match distance statistics:\n"
    )

    if best_distance is not None:

        f.write(
            f"Best distance: {best_distance:.2f}\n"
        )

        f.write(
            f"Mean distance: {mean_distance:.2f}\n"
        )

        f.write(
            f"Median distance: {median_distance:.2f}\n"
        )

    else:

        f.write(
            "No good matches found.\n"
        )

    f.write("\n")

    f.write(
        "Geometric consistency:\n"
    )

    f.write(
        f"RANSAC inliers: "
        f"{homography_inliers}\n"
    )

    f.write(
        f"Inlier ratio: "
        f"{inlier_ratio:.2f}%\n\n"
    )

    f.write(
        "TOP MATCHES\n"
    )

    f.write(
        "-" * 65 + "\n"
    )

    for info in top_match_information:

        f.write(
            f"Rank {info['rank']}: "
            f"distance={info['distance']:.2f}, "
            f"original=({info['x1']:.1f}, "
            f"{info['y1']:.1f}), "
            f"new_angle=({info['x2']:.1f}, "
            f"{info['y2']:.1f})\n"
        )

    f.write("\n")

    f.write(
        "STRONGEST REAL-OBJECT MATCH\n"
    )

    f.write(
        "-" * 65 + "\n"
    )

    f.write(
        "The strongest individual ORB match is represented "
        "by Rank 1 above.\n\n"
    )

    f.write(
        "Use task7_top_matches.jpg together with the "
        "original desk image to identify which physical "
        "object contains the Rank 1 keypoint.\n"
    )


# ============================================================
# 20. Completion
# ============================================================

print("\n" + "=" * 65)
print("TASK 7 COMPLETED")
print("=" * 65)

print("\nRESULTS")

print(
    f"Original keypoints : {len(keypoints1)}"
)

print(
    f"New-angle keypoints: {len(keypoints2)}"
)

print(
    f"Raw KNN pairs      : {len(knn_matches)}"
)

print(
    f"Good matches       : {len(good_matches)}"
)

print(
    f"RANSAC inliers     : {homography_inliers}"
)

print(
    f"Inlier ratio       : {inlier_ratio:.2f}%"
)

if best_distance is not None:

    print(
        f"Best match distance: {best_distance:.2f}"
    )

print("\nSaved outputs:")

print(
    "  outputs/task7_keypoints_original.jpg"
)

print(
    "  outputs/task7_keypoints_angle2.jpg"
)

print(
    "  outputs/task7_all_matches.jpg"
)

print(
    "  outputs/task7_good_matches.jpg"
)

print(
    "  outputs/task7_top_matches.jpg"
)

if len(inlier_matches) > 0:

    print(
        "  outputs/task7_ransac_inliers.jpg"
    )

print(
    "  outputs/task7_match_comparison.jpg"
)

print(
    "  outputs/task7_results.txt"
)