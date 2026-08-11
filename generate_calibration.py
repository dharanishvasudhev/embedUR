import os
import numpy as np

from torchvision import datasets, transforms


# Image Transform
transform = transforms.Compose([
    transforms.ToTensor()
])


# Load MNIST Test Dataset
test_data = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)


# Create Calibration Folder
os.makedirs("calib", exist_ok=True)


# Save First 50 Samples
num_samples = 50

for i in range(num_samples):

    image, label = test_data[i]

    image = image.numpy()

    np.save(
        f"calib/{i}.npy",
        image
    )

print(f"{num_samples} calibration samples saved.")