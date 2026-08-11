import cv2
import matplotlib.pyplot as plt

# --------------------------------------------------
# 1. Load the low-light image
# --------------------------------------------------

image_path = "raw_captures/low_light.jpg"

image_bgr = cv2.imread(image_path)

if image_bgr is None:
    raise FileNotFoundError(
        f"Could not find image at: {image_path}"
    )

# --------------------------------------------------
# 2. Convert BGR to other color spaces
# --------------------------------------------------

image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

# --------------------------------------------------
# 3. Extract HSV channels
# --------------------------------------------------

h_channel = image_hsv[:, :, 0]
s_channel = image_hsv[:, :, 1]
v_channel = image_hsv[:, :, 2]

# --------------------------------------------------
# 4. Display RGB, Grayscale and HSV
# --------------------------------------------------

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(image_rgb)
plt.title("RGB")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(image_gray, cmap="gray")
plt.title("Grayscale")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(image_hsv)
plt.title("HSV")
plt.axis("off")

plt.tight_layout()
plt.show()

# --------------------------------------------------
# 5. Display individual HSV channels
# --------------------------------------------------

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(h_channel, cmap="gray")
plt.title("Hue (H)")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(s_channel, cmap="gray")
plt.title("Saturation (S)")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(v_channel, cmap="gray")
plt.title("Value (V)")
plt.axis("off")

plt.tight_layout()
plt.show()