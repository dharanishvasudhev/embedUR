import time
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.applications import MobileNet

from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import GlobalAveragePooling2D

from tensorflow.keras.models import Model

# --------------------------------------
# Dataset
# --------------------------------------

train_gen = ImageDataGenerator(

rescale=1./255,

validation_split=0.2

)

train_data = train_gen.flow_from_directory(

"data",

target_size=(224,224),

batch_size=32,

subset="training"

)

val_data = train_gen.flow_from_directory(

"data",

target_size=(224,224),

batch_size=32,

subset="validation"

)

# --------------------------------------
# MobileNet
# --------------------------------------

base_model = MobileNet(

weights="imagenet",

include_top=False,

input_shape=(224,224,3)

)

# Freeze Layers

for layer in base_model.layers:

    layer.trainable=False

# --------------------------------------
# New Layers
# --------------------------------------

x = base_model.output

x = GlobalAveragePooling2D()(x)

x = Dense(

128,

activation="relu"

)(x)

output = Dense(

2,

activation="softmax"

)(x)

model = Model(

inputs=base_model.input,

outputs=output

)

# --------------------------------------
# Compile
# --------------------------------------

model.compile(

optimizer="adam",

loss="categorical_crossentropy",

metrics=["accuracy"]

)

# --------------------------------------
# Train
# --------------------------------------

model.fit(

train_data,

epochs=5,

validation_data=val_data

)

# --------------------------------------
# Evaluation
# --------------------------------------

loss,accuracy=model.evaluate(val_data)

print("Validation Accuracy:",accuracy)

# --------------------------------------
# Inference Speed
# --------------------------------------

sample=next(val_data)

image=sample[0]

start=time.time()

model.predict(image)

end=time.time()

print("Inference Time")

print(end-start)