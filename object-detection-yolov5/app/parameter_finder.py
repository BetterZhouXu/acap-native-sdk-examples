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

if len(input_details) != 1 or len(output_details) != 3:
    raise ValueError('Expected one input and three OBB head outputs')
inp = input_details[0]
if list(inp['shape']) != [1, 3, 640, 640] or inp['dtype'] != np.int8:
    raise ValueError('Expected INT8 NCHW input [1,3,640,640]')
for output, shape in zip(output_details, ([1, 4, 8400], [1, 4, 8400], [1, 1, 8400])):
    if list(output['shape']) != shape or output['dtype'] != np.int8:
        raise ValueError(f'Expected INT8 OBB head output {shape}')
for tensor in input_details + output_details:
    if tensor['quantization'][0] <= 0 or \
            len(tensor['quantization_parameters']['scales']) != 1:
        raise ValueError('All tensors must have per-tensor affine quantization')

input_scale, input_zero = inp['quantization']

input_range = os.environ.get('MODEL_INPUT_RANGE', '0_1')
if input_range not in ('0_1', '0_255'):
    raise ValueError('MODEL_INPUT_RANGE must be 0_1 or 0_255')

with open(output_file, "w") as f:
    f.write(f"#ifndef MODEL_PARAMS_H\n")
    f.write(f"#define MODEL_PARAMS_H\n\n")
    f.write("#define MODEL_INPUT_HEIGHT 640\n#define MODEL_INPUT_WIDTH 640\n\n")
    f.write(f"#define INPUT_SCALE {input_scale}f\n#define INPUT_ZERO_POINT {input_zero}\n")
    f.write(f"#define INPUT_DIVISOR {255 if input_range == '0_1' else 1}.0f\n")
    for name, output in zip(('DIST', 'CLASS', 'ANGLE'), output_details):
        scale, zero = output['quantization']
        f.write(f"#define {name}_SCALE {scale}f\n#define {name}_ZERO_POINT {zero}\n")
    f.write("#define NUM_CLASSES 4\n#define NUM_DETECTIONS 8400\n\n")
    f.write(f"#endif // MODEL_PARAMS_H\n")

print(f"Model parameters have been saved to {output_file}.")
