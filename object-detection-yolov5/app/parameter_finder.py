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
Check your model quantization parameters and save them to file
"""
import tensorflow as tf
import sys
import os
import numpy as np

if len(sys.argv) > 1:
    model_path = sys.argv[1]
else:
    print("Error: No model path provided as parameter. Please provide a path "
          "as a command-line argument.")
    exit(1)

output_file = "model_params.h"
interpreter = tf.lite.Interpreter(model_path)
interpreter.allocate_tensors()
output_details = interpreter.get_output_details()
input_details  = interpreter.get_input_details()

if len(input_details) != 1 or len(output_details) != 1:
    raise ValueError('Expected exactly one input and one output tensor')
inp, out = input_details[0], output_details[0]
if list(inp['shape']) != [1, 3, 640, 640] or list(out['shape']) != [1, 9, 8400] or \
        inp['dtype'] != np.int8 or out['dtype'] != np.int8:
    raise ValueError('Expected one INT8 NCHW input [1,3,640,640] and one INT8 OBB output [1,9,8400]')

input_scale, input_zero = inp['quantization']
output_scale, output_zero = out['quantization']
if input_scale <= 0 or output_scale <= 0 or any(
        len(t['quantization_parameters']['scales']) != 1 for t in (inp, out)):
    raise ValueError('Input and output must have per-tensor affine quantization')

input_range = os.environ.get('MODEL_INPUT_RANGE', '0_1')
output_coords = os.environ.get('MODEL_OUTPUT_COORDS', 'normalized')
if input_range not in ('0_1', '0_255') or output_coords not in ('normalized', 'pixels'):
    raise ValueError('MODEL_INPUT_RANGE must be 0_1 or 0_255; MODEL_OUTPUT_COORDS must be normalized or pixels')

with open(output_file, "w") as f:
    f.write(f"#ifndef MODEL_PARAMS_H\n")
    f.write(f"#define MODEL_PARAMS_H\n\n")
    f.write("#define MODEL_INPUT_HEIGHT 640\n#define MODEL_INPUT_WIDTH 640\n\n")
    f.write(f"#define INPUT_SCALE {input_scale}f\n#define INPUT_ZERO_POINT {input_zero}\n")
    f.write(f"#define INPUT_DIVISOR {255 if input_range == '0_1' else 1}.0f\n")
    f.write(f"#define OUTPUT_COORDS_NORMALIZED {1 if output_coords == 'normalized' else 0}\n")
    f.write(f"#define QUANTIZATION_SCALE {output_scale}f\n")
    f.write(f"#define QUANTIZATION_ZERO_POINT {output_zero}\n\n")
    f.write("#define NUM_CLASSES 4\n#define NUM_DETECTIONS 8400\n\n")
    f.write(f"#endif // MODEL_PARAMS_H\n")

print(f"Model parameters have been saved to {output_file}.")
