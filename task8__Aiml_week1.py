import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# ---------------------------------------
# Data Preprocessing & Augmentation
# ---------------------------------------

train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

test_gen = ImageDataGenerator(
    rescale=1./255
)

# ---------------------------------------
# Load Training Dataset
# ---------------------------------------

train_data = train_gen.flow_from_directory(
    "casting_data/casting_data/train",
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical"
)

# ---------------------------------------
# Load Test Dataset
# ---------------------------------------

test_data = test_gen.flow_from_directory(
    "casting_data/casting_data/test",
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    shuffle=False
)

# Display detected classes
print("\nClasses Found:")
print(train_data.class_indices)

# ---------------------------------------
# Load Pretrained MobileNet
# ---------------------------------------

base_model = MobileNet(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# ---------------------------------------
# Freeze MobileNet Layers
# ---------------------------------------

for layer in base_model.layers:
    layer.trainable = False

# ---------------------------------------
# Add Custom Classification Layers
# ---------------------------------------

x = base_model.output

x = GlobalAveragePooling2D()(x)

x = Dense(128, activation="relu")(x)

output = Dense(2, activation="softmax")(x)

model = Model(
    inputs=base_model.input,
    outputs=output
)

# ---------------------------------------
# Compile Model
# ---------------------------------------

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ---------------------------------------
# Model Summary
# ---------------------------------------

model.summary()

# ---------------------------------------
# Train Model
# ---------------------------------------

history = model.fit(
    train_data,
    epochs=5,
    validation_data=test_data
)

# ---------------------------------------
# Evaluate Model
# ---------------------------------------

loss, accuracy = model.evaluate(test_data)

print("\nTest Loss :", loss)
print("Test Accuracy :", accuracy)

# ---------------------------------------
# Save Model
# ---------------------------------------

model.save("casting_mobilenet_model.h5")

print("\nModel saved successfully as casting_mobilenet_model.h5")