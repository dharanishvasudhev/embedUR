import os
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------------------
# Dataset Path
# ---------------------------------------

dataset_path = r"D:\AIML\Week1\PetImages"

# ---------------------------------------
# Metadata List
# ---------------------------------------

metadata = []

# ---------------------------------------
# Read Images One by One
# ---------------------------------------

for label in os.listdir(dataset_path):

    class_folder = os.path.join(dataset_path, label)

    if not os.path.isdir(class_folder):
        continue

    for file in os.listdir(class_folder):

        image_path = os.path.join(class_folder, file)

        try:

            # Skip empty files
            if os.path.getsize(image_path) == 0:
                continue

            image = cv2.imread(image_path)

            if image is None:
                continue

            # Resize image
            image = cv2.resize(image, (224, 224))

            # Normalize (only for verification)
            image = image.astype("float32") / 255.0

            # Save metadata only
            metadata.append({
                "Filename": file,
                "Label": label,
                "Path": image_path,
                "Width": image.shape[1],
                "Height": image.shape[0]
            })

        except Exception:
            continue

# ---------------------------------------
# Create DataFrame
# ---------------------------------------

df = pd.DataFrame(metadata)

print("\nTotal Valid Images :", len(df))

# ---------------------------------------
# Train Validation Split
# ---------------------------------------

train_df, val_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["Label"]
)

# ---------------------------------------
# Save Metadata CSV Files
# ---------------------------------------

train_df.to_csv("train_metadata.csv", index=False)
val_df.to_csv("validation_metadata.csv", index=False)

print("\nTraining Images :", len(train_df))
print("Validation Images :", len(val_df))

print("\nFirst 5 Rows")
print(df.head())

print("\nCSV files created successfully.")