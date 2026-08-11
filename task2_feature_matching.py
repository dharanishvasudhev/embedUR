import cv2
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. Load images
# ============================================================

normal_path = "raw_captures/normal_light.jpg"
lowlight_path = "raw_captures/low_light.jpg"

normal = cv2.imread(normal_path)
lowlight = cv2.imread(lowlight_path)

if normal is None:
    raise FileNotFoundError(f"Could not load: {normal_path}")

if lowlight is None:
    raise FileNotFoundError(f"Could not load: {lowlight_path}")


# ============================================================
# 2. Convert images to grayscale
# ============================================================

normal_gray = cv2.cvtColor(normal, cv2.COLOR_BGR2GRAY)
lowlight_gray = cv2.cvtColor(lowlight, cv2.COLOR_BGR2GRAY)


# ============================================================
# 3. Create ORB detector
# ============================================================

orb = cv2.ORB_create(
    nfeatures=2000
)


# ============================================================
# 4. Detect keypoints and descriptors
# ============================================================

keypoints1, descriptors1 = orb.detectAndCompute(
    normal_gray,
    None
)

keypoints2, descriptors2 = orb.detectAndCompute(
    lowlight_gray,
    None
)

print("Normal-light keypoints:", len(keypoints1))
print("Low-light keypoints:", len(keypoints2))


# ============================================================
# 5. Create Brute-Force matcher
# ============================================================

bf = cv2.BFMatcher(
    cv2.NORM_HAMMING,
    crossCheck=False
)


# ============================================================
# 6. Find two nearest matches
# ============================================================

matches = bf.knnMatch(
    descriptors1,
    descriptors2,
    k=2
)


# ============================================================
# 7. Apply Lowe's ratio test
# ============================================================

good_matches = []

for m, n in matches:

    if m.distance < 0.75 * n.distance:
        good_matches.append(m)


print("Total candidate matches:", len(matches))
print("Good matches:", len(good_matches))


# ============================================================
# 8. Visualize good matches
# ============================================================

match_image = cv2.drawMatches(
    normal,
    keypoints1,
    lowlight,
    keypoints2,
    good_matches,
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

match_image_rgb = cv2.cvtColor(
    match_image,
    cv2.COLOR_BGR2RGB
)

plt.figure(figsize=(18, 9))
plt.imshow(match_image_rgb)
plt.title(f"ORB Good Matches: {len(good_matches)}")
plt.axis("off")
plt.show()


# ============================================================
# 9. Check whether enough matches exist
# ============================================================

if len(good_matches) < 4:
    raise RuntimeError(
        "Not enough good matches to calculate homography."
    )


# ============================================================
# 10. Extract corresponding points
# ============================================================

src_pts = np.float32([
    keypoints2[m.trainIdx].pt
    for m in good_matches
]).reshape(-1, 1, 2)

dst_pts = np.float32([
    keypoints1[m.queryIdx].pt
    for m in good_matches
]).reshape(-1, 1, 2)


# ============================================================
# 11. Estimate homography using RANSAC
# ============================================================

H, mask = cv2.findHomography(
    src_pts,
    dst_pts,
    cv2.RANSAC,
    5.0
)

if H is None:
    raise RuntimeError(
        "Homography estimation failed."
    )


# ============================================================
# 12. Count RANSAC inliers
# ============================================================

inliers = mask.ravel().tolist()

num_inliers = sum(inliers)

print("RANSAC inliers:", num_inliers)
print(
    "Inlier ratio:",
    num_inliers / len(good_matches)
)


# ============================================================
# 13. Warp low-light image onto normal image plane
# ============================================================

height, width = normal.shape[:2]

aligned_lowlight = cv2.warpPerspective(
    lowlight,
    H,
    (width, height)
)


# ============================================================
# 14. Convert images for display
# ============================================================

normal_rgb = cv2.cvtColor(
    normal,
    cv2.COLOR_BGR2RGB
)

aligned_lowlight_rgb = cv2.cvtColor(
    aligned_lowlight,
    cv2.COLOR_BGR2RGB
)


# ============================================================
# 15. Display alignment
# ============================================================

plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
plt.imshow(normal_rgb)
plt.title("Normal-light Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(aligned_lowlight_rgb)
plt.title("Aligned Low-light Image")
plt.axis("off")

plt.tight_layout()
plt.show()


# ============================================================
# 16. Blend - Combination 1
# ============================================================

blend1 = cv2.addWeighted(
    normal,
    0.5,
    aligned_lowlight,
    0.5,
    0
)

blend1_rgb = cv2.cvtColor(
    blend1,
    cv2.COLOR_BGR2RGB
)


# ============================================================
# 17. Blend - Combination 2
# ============================================================

blend2 = cv2.addWeighted(
    normal,
    0.7,
    aligned_lowlight,
    0.3,
    0
)

blend2_rgb = cv2.cvtColor(
    blend2,
    cv2.COLOR_BGR2RGB
)


# ============================================================
# 18. Display both blends
# ============================================================

plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
plt.imshow(blend1_rgb)
plt.title("Blend 1: Normal 0.5 + Low-light 0.5")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(blend2_rgb)
plt.title("Blend 2: Normal 0.7 + Low-light 0.3")
plt.axis("off")

plt.tight_layout()
plt.show()


# ============================================================
# 19. Save outputs
# ============================================================

cv2.imwrite(
    "outputs/task2_matches.jpg",
    match_image
)

cv2.imwrite(
    "outputs/task2_aligned_lowlight.jpg",
    aligned_lowlight
)

cv2.imwrite(
    "outputs/task2_blend_50_50.jpg",
    blend1
)

cv2.imwrite(
    "outputs/task2_blend_70_30.jpg",
    blend2
)

print("\nOutputs saved successfully.")