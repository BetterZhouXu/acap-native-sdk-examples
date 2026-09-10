"""
Copyright (C) 2025, Axis Communications AB, Lund, Sweden

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""
Validate YOLOv8 OBB tensor metadata and save it to a C header.
"""
import sys

import numpy as np
import tensorflow as tf

if len(sys.argv) > 1:
    model_path = sys.argv[1]
else:
    print("Error: No model path provided as parameter. Please provide a path "
          "as a command-line argument.")
    sys.exit(1)

output_file = "model_params.h"
interpreter = tf.lite.Interpreter(model_path)
interpreter.allocate_tensors()
output_details = interpreter.get_output_details()
input_details  = interpreter.get_input_details()

if len(input_details) != 1 or len(output_details) != 1:
    print(f"Error: Expected one input and one output, got "
          f"{len(input_details)} and {len(output_details)}.")
    sys.exit(1)

input_shape  = input_details[0]["shape"].tolist()
output_shape = output_details[0]["shape"].tolist()

if (
    input_details[0]["dtype"] != tf.float32.as_numpy_dtype
    or len(input_shape) != 4
    or input_shape[0] != 1
    or input_shape[1] != 3
):
    print(f"Error: Expected float32 NCHW input [1, 3, height, width], got "
          f"{input_details[0]['dtype']} {input_shape}.")
    sys.exit(1)
if (
    output_details[0]["dtype"] != tf.float32.as_numpy_dtype
    or len(output_shape) != 3
    or output_shape[0] != 1
):
    print(f"Error: Expected float32 output [1, features, detections], got "
          f"{output_details[0]['dtype']} {output_shape}.")
    sys.exit(1)

model_input_height = input_shape[2]
model_input_width  = input_shape[3]
num_features       = output_shape[1]
num_detections     = output_shape[2]
num_classes        = num_features - 5  # x, y, width, height, classes..., angle

if num_classes <= 0:
    print(f"Error: Invalid YOLOv8 OBB output shape {output_shape}.")
    sys.exit(1)

if len(sys.argv) < 3:
    print("Error: No labels file provided as the second argument.")
    sys.exit(1)

labels_path = sys.argv[2]
with open(labels_path, "r", encoding="utf-8") as labels_file:
    raw_labels = [line.rstrip("\r\n") for line in labels_file]

if not raw_labels or any(not label.strip() for label in raw_labels):
    print(f"Error: {labels_path} must contain one non-empty label per line.")
    sys.exit(1)

# labelparse.c ignores legacy placeholder entries named "n/a".
labels = [label for label in raw_labels if label != "n/a"]
if len(labels) != num_classes:
    print(f"Error: Model outputs {num_classes} classes, but {labels_path} "
          f"contains {len(labels)} effective labels.")
    sys.exit(1)

print(f"Validated YOLOv8 OBB input {input_shape}, output {output_shape}, "
      f"and {len(labels)} labels.")

try:
    interpreter.set_tensor(input_details[0]["index"], np.zeros(input_shape, dtype=np.float32))
    interpreter.invoke()
    smoke_output = interpreter.get_tensor(output_details[0]["index"])
except Exception as error:  # TensorFlow Lite uses several backend-specific exception types.
    print(f"Error: Zero-input TensorFlow Lite inference failed: {error}")
    sys.exit(1)

if list(smoke_output.shape) != output_shape or not np.all(np.isfinite(smoke_output)):
    print(f"Error: Invalid zero-input inference output shape or non-finite values: "
          f"{list(smoke_output.shape)}")
    sys.exit(1)

print("Zero-input TensorFlow Lite inference succeeded.")

with open(output_file, "w") as f:
    f.write(f"#ifndef MODEL_PARAMS_H\n")
    f.write(f"#define MODEL_PARAMS_H\n\n")
    f.write(f"#define MODEL_INPUT_HEIGHT {model_input_height}\n")
    f.write(f"#define MODEL_INPUT_WIDTH {model_input_width}\n\n")
    f.write(f"#define NUM_CLASSES {num_classes}\n")
    f.write(f"#define NUM_DETECTIONS {num_detections}\n")
    f.write(f"#define NUM_FEATURES {num_features}\n\n")
    f.write(f"#endif // MODEL_PARAMS_H\n")

print(f"Model parameters have been saved to {output_file}.")
