"""
Task 6 - PyTorch (.pth) to Fully INT8 TFLite Conversion

Part 1
-------
1. Imports
2. Logging
3. Load trained PyTorch model
4. Validate calibration data
"""

import os
import sys
import logging
import numpy as np

import torch
import onnx
import tensorflow as tf

from model_definition import SimpleCNN
from onnx2tf import convert


# ==========================================================
# Logging Configuration
# ==========================================================

logging.basicConfig(
    filename="conversion_log.txt",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="w"
)

print("=" * 60)
print("PyTorch -> ONNX -> TensorFlow -> INT8 TFLite")
print("=" * 60)

logging.info("Conversion started.")


# ==========================================================
# File and Folder Paths
# ==========================================================

CHECKPOINT_PATH = "model.pth"

ONNX_PATH = "model.onnx"

SAVED_MODEL_DIR = "saved_model"

TFLITE_PATH = "model_int8.tflite"

CALIB_DIR = "calib"


# ==========================================================
# Step 1 : Load PyTorch Model
# ==========================================================

print("\nStep 1 : Loading PyTorch Model")

logging.info("Loading PyTorch model.")


# Check checkpoint exists

if not os.path.exists(CHECKPOINT_PATH):

    print("ERROR : model.pth not found.")

    logging.error("Checkpoint file missing.")

    sys.exit(1)


# Create model

model = SimpleCNN()


# Load weights

try:

    model.load_state_dict(

        torch.load(
            CHECKPOINT_PATH,
            map_location="cpu"
        )

    )

except Exception as e:

    print("ERROR while loading checkpoint.")

    print(e)

    logging.exception("Checkpoint loading failed.")

    sys.exit(1)


# Switch to evaluation mode

model.eval()

print("Model loaded successfully.")

logging.info("Model loaded successfully.")


# ==========================================================
# Step 2 : Validate Calibration Data
# ==========================================================

print("\nStep 2 : Validating Calibration Data")

logging.info("Checking calibration directory.")


# Folder exists?

if not os.path.isdir(CALIB_DIR):

    print("ERROR : Calibration folder missing.")

    logging.error("Calibration directory not found.")

    sys.exit(1)


# Get all npy files

calibration_files = sorted(

    [

        file

        for file in os.listdir(CALIB_DIR)

        if file.endswith(".npy")

    ]

)


# Folder empty?

if len(calibration_files) == 0:

    print("ERROR : Calibration folder is empty.")

    logging.error("Calibration folder empty.")

    sys.exit(1)


print(f"Found {len(calibration_files)} calibration samples.")

logging.info(f"{len(calibration_files)} calibration files found.")


# ----------------------------------------------------------
# Validate every calibration file
# ----------------------------------------------------------

EXPECTED_SHAPE = (1, 28, 28)


for file in calibration_files:

    filepath = os.path.join(CALIB_DIR, file)

    try:

        sample = np.load(filepath)

    except Exception as e:

        print(f"Cannot read {file}")

        logging.exception(f"Failed reading {file}")

        sys.exit(1)


    # ----------------------------
    # Shape Check
    # ----------------------------

    if sample.shape != EXPECTED_SHAPE:

        print(f"ERROR : {file} has wrong shape.")

        print("Expected :", EXPECTED_SHAPE)

        print("Found    :", sample.shape)

        logging.error(f"{file} invalid shape.")

        sys.exit(1)


    # ----------------------------
    # Dtype Check
    # ----------------------------

    if sample.dtype != np.float32:

        print(f"ERROR : {file} is not float32.")

        logging.error(f"{file} invalid dtype.")

        sys.exit(1)


    # ----------------------------
    # NaN Check
    # ----------------------------

    if np.isnan(sample).any():

        print(f"ERROR : {file} contains NaN.")

        logging.error(f"{file} contains NaN.")

        sys.exit(1)


    # ----------------------------
    # Inf Check
    # ----------------------------

    if np.isinf(sample).any():

        print(f"ERROR : {file} contains Inf.")

        logging.error(f"{file} contains Inf.")

        sys.exit(1)


print("Calibration validation successful.")

logging.info("Calibration validation completed.")


# ==========================================================
# Step 3 : Dummy Input
# ==========================================================

print("\nPreparing dummy input for ONNX export...")

dummy_input = torch.randn(
    1,
    1,
    28,
    28
)

print("Dummy Input Shape :", tuple(dummy_input.shape))

logging.info("Dummy input created.")


print("\nPart 1 Completed Successfully.")

logging.info("Part 1 completed.")

# ==========================================================
# Step 4 : Export PyTorch Model to ONNX
# ==========================================================

print("\nStep 4 : Exporting Model to ONNX")

logging.info("Starting ONNX export.")

try:

    torch.onnx.export(

        model,

        dummy_input,

        ONNX_PATH,

        export_params=True,

        opset_version=13,

        do_constant_folding=True,

        input_names=["input"],

        output_names=["output"],

        dynamic_axes=None

    )

    print("ONNX model exported successfully.")

    logging.info("ONNX export successful.")

except Exception as e:

    print("ERROR : ONNX export failed.")

    print(e)

    logging.exception("ONNX export failed.")

    sys.exit(1)


# ==========================================================
# Step 5 : Validate ONNX Model
# ==========================================================

print("\nStep 5 : Validating ONNX Model")

logging.info("Validating ONNX model.")

try:

    onnx_model = onnx.load(ONNX_PATH)

    onnx.checker.check_model(onnx_model)

    print("ONNX model validation successful.")

    logging.info("ONNX validation successful.")

except Exception as e:

    print("ERROR : Invalid ONNX model.")

    print(e)

    logging.exception("ONNX validation failed.")

    sys.exit(1)


# ==========================================================
# Print ONNX Information
# ==========================================================

print("\nONNX Information")

print("--------------------------------")

print("Inputs :")

for inp in onnx_model.graph.input:

    print(" ", inp.name)

print()

print("Outputs :")

for out in onnx_model.graph.output:

    print(" ", out.name)

print()

logging.info("ONNX graph inspected.")


# ==========================================================
# Step 6 : Remove Old SavedModel (if exists)
# ==========================================================

if os.path.exists(SAVED_MODEL_DIR):

    print("Removing previous SavedModel directory...")

    import shutil

    shutil.rmtree(SAVED_MODEL_DIR)

    logging.info("Previous SavedModel removed.")


# ==========================================================
# Step 7 : Convert ONNX → TensorFlow SavedModel
# ==========================================================

print("\nStep 7 : Converting ONNX to TensorFlow SavedModel")

logging.info("Starting ONNX -> TensorFlow conversion.")

try:

    convert(

        input_onnx_file_path=ONNX_PATH,

        output_folder_path=SAVED_MODEL_DIR,

    )

    print("TensorFlow SavedModel created.")

    logging.info("TensorFlow SavedModel created.")

except Exception as e:

    print("ERROR : TensorFlow conversion failed.")

    print(e)

    logging.exception("ONNX -> TensorFlow failed.")

    sys.exit(1)


# ==========================================================
# Verify SavedModel Exists
# ==========================================================

if not os.path.exists(SAVED_MODEL_DIR):

    print("ERROR : SavedModel folder missing.")

    logging.error("SavedModel missing.")

    sys.exit(1)

print("SavedModel verified.")

logging.info("SavedModel verified.")


# ==========================================================
# Print SavedModel Signature
# ==========================================================

print("\nLoading SavedModel...")

try:

    saved_model = tf.saved_model.load(SAVED_MODEL_DIR)

    print("SavedModel loaded successfully.")

    logging.info("SavedModel loaded.")

except Exception as e:

    print("ERROR : Unable to load SavedModel.")

    print(e)

    logging.exception("SavedModel loading failed.")

    sys.exit(1)


print("\nAvailable Signatures")

print("----------------------")

for key in saved_model.signatures.keys():

    print(key)

logging.info("SavedModel signatures printed.")


print("\nPart 2 Completed Successfully.")

logging.info("Part 2 completed.")

# ==========================================================
# Step 8 : Representative Dataset Generator
# ==========================================================

print("\nStep 8 : Creating Representative Dataset")

logging.info("Creating representative dataset.")

def representative_dataset():

    for file in calibration_files:

        path = os.path.join(CALIB_DIR, file)

        sample = np.load(path).astype(np.float32)

        # sample shape: (1, 28, 28)

        # Convert CHW -> HWC
        sample = np.transpose(sample, (1, 2, 0))

        # Add batch dimension
        sample = np.expand_dims(sample, axis=0)
        
        print(sample.shape)

        # Final shape: (1, 28, 28, 1)
        yield [sample]

print("Representative dataset ready.")

logging.info("Representative dataset ready.")


# ==========================================================
# Step 9 : Convert SavedModel → INT8 TFLite
# ==========================================================

print("\nStep 9 : Creating Fully INT8 TFLite Model")

logging.info("Starting TFLite conversion.")

try:

    converter = tf.lite.TFLiteConverter.from_saved_model(
        SAVED_MODEL_DIR
    )

    converter.optimizations = [
        tf.lite.Optimize.DEFAULT
    ]

    converter.representative_dataset = representative_dataset

    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8
    ]

    converter.inference_input_type = tf.int8

    converter.inference_output_type = tf.int8
    
    print("Starting converter.convert()...")

    tflite_model = converter.convert()

    print("converter.convert() completed.")

    with open(TFLITE_PATH, "wb") as f:
        f.write(tflite_model)

    print("TFLite file written successfully.")

    logging.info("INT8 model generated.")

except Exception as e:

    print("ERROR : TFLite conversion failed.")

    print(e)

    logging.exception("TFLite conversion failed.")

    sys.exit(1)


# ==========================================================
# Step 10 : Verify TFLite Model
# ==========================================================

print("\nStep 10 : Verifying INT8 Model")

logging.info("Verifying model.")

interpreter = tf.lite.Interpreter(
    model_path=TFLITE_PATH
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()

output_details = interpreter.get_output_details()

print("\nInput Tensor")

print(input_details[0])

print("\nOutput Tensor")

print(output_details[0])


# Check input dtype

if input_details[0]["dtype"] != np.int8:

    print("WARNING : Input is NOT INT8")

    logging.warning("Input not INT8.")

else:

    print("Input dtype : INT8")


# Check output dtype

if output_details[0]["dtype"] != np.int8:

    print("WARNING : Output is NOT INT8")

    logging.warning("Output not INT8.")

else:

    print("Output dtype : INT8")


# Print Quantization Parameters

input_scale, input_zero = input_details[0]["quantization"]

output_scale, output_zero = output_details[0]["quantization"]

print("\nInput Scale :", input_scale)

print("Input Zero Point :", input_zero)

print()

print("Output Scale :", output_scale)

print("Output Zero Point :", output_zero)

logging.info("Quantization parameters printed.")


# ==========================================================
# Model Size
# ==========================================================

size = os.path.getsize(TFLITE_PATH) / 1024

print(f"\nModel Size : {size:.2f} KiB")

logging.info(f"Model size : {size:.2f} KiB")


# ==========================================================
# Step 11 : Run One Inference
# ==========================================================

print("\nStep 11 : Running Test Inference")

sample = np.load(
    os.path.join(CALIB_DIR, calibration_files[0])
).astype(np.float32)

# Current shape:
# (1, 28, 28)

# Convert CHW -> HWC
sample = np.transpose(sample, (1, 2, 0))

# Add batch dimension
sample = np.expand_dims(sample, axis=0)

# Shape should be:
print("Input Shape :", sample.shape)

# Quantize input
sample = sample / input_scale + input_zero

sample = np.round(sample)

sample = np.clip(sample, -128, 127)

sample = sample.astype(np.int8)

interpreter.set_tensor(
    input_details[0]["index"],
    sample
)

interpreter.invoke()

output = interpreter.get_tensor(
    output_details[0]["index"]
)

print("\nRaw INT8 Output")

print(output)

logging.info("Inference successful.")

print("\nConversion Completed Successfully.")