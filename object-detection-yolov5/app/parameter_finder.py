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

import argparse
import math
from pathlib import Path

import numpy as np


def model_header(inputs, outputs, coordinates):
    """Validate metadata without running inference (DLPU custom ops are allowed)."""
    if coordinates not in ("normalized", "pixels"):
        raise ValueError("Coordinates must be normalized or pixels")
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError(f"Expected one input and one output; got {len(inputs)} and {len(outputs)}")
    model_input, model_output = inputs[0], outputs[0]
    shape = tuple(int(n) for n in model_input["shape"])
    expected = "RGB input NHWC [1, height, width, 3] or NCHW [1, 3, height, width]"
    if len(shape) != 4 or shape[0] != 1 or min(shape) <= 0:
        raise ValueError(f"Expected {expected} with fixed positive dimensions; got {shape}")
    if shape[1] == 3 and shape[3] == 3:
        raise ValueError(f"Ambiguous input layout for {shape}: both possible channel axes are 3")
    if shape[3] == 3:
        height, width = shape[1], shape[2]
        nchw = False
    elif shape[1] == 3:
        height, width = shape[2], shape[3]
        nchw = True
    else:
        raise ValueError(f"Expected {expected}; got {shape}")
    input_dtype = np.dtype(model_input["dtype"])
    if input_dtype != np.dtype("uint8"):
        raise ValueError(
            f"Input shape {shape} uses {input_dtype}, but VDO preprocessing requires uint8 RGB. "
            "Export a camera-compatible uint8-input model; float32/int8 inputs need additional "
            "normalization/quantization code and cannot be used by simply removing this check.")
    if tuple(model_output["shape"]) != (1, 9, 18900):
        raise ValueError("Expected channel-major OBB output [1, 9, 18900]; "
                         f"got {tuple(model_output['shape'])}")
    types = {np.dtype("float32"): "OBB_FLOAT32", np.dtype("uint8"): "OBB_UINT8",
             np.dtype("int8"): "OBB_INT8"}
    dtype = np.dtype(model_output["dtype"])
    if dtype not in types:
        raise ValueError(f"Unsupported output datatype: {dtype}")
    scale, zero_point = 1.0, 0
    if dtype != np.dtype("float32"):
        quant = model_output["quantization_parameters"]
        if len(quant["scales"]) != 1 or len(quant["zero_points"]) != 1:
            raise ValueError("Only per-tensor output quantization is supported")
        scale = float(quant["scales"][0])
        zero_point = int(quant["zero_points"][0])
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("Output quantization scale must be finite and positive")
        limits = np.iinfo(dtype)
        if not limits.min <= zero_point <= limits.max:
            raise ValueError("Invalid output zero point")
    return ("#ifndef MODEL_PARAMS_H\n#define MODEL_PARAMS_H\n\n"
            f"#define MODEL_INPUT_HEIGHT {height}\n"
            f"#define MODEL_INPUT_WIDTH {width}\n"
            f"#define MODEL_INPUT_LAYOUT_NCHW {int(nchw)}\n"
            f"#define QUANTIZATION_SCALE {scale!r}f\n"
            f"#define QUANTIZATION_ZERO_POINT {zero_point}\n"
            f"#define MODEL_OUTPUT_DTYPE {types[dtype]}\n"
            f"#define MODEL_COORDINATES_NORMALIZED {int(coordinates == 'normalized')}\n"
            "#define NUM_CLASSES 4\n#define NUM_DETECTIONS 18900\n\n#endif\n")


def main():
    parser = argparse.ArgumentParser(description="Inspect a YOLOv8 OBB TFLite model")
    parser.add_argument("model")
    parser.add_argument("--coordinates", choices=("normalized", "pixels"), required=True)
    parser.add_argument("--output", default="model_params.h")
    args = parser.parse_args()
    import tensorflow as tf  # Only the build-time CLI needs TensorFlow.

    interpreter = tf.lite.Interpreter(model_path=args.model)
    inputs = interpreter.get_input_details()
    outputs = interpreter.get_output_details()
    for kind, tensors in (("Input", inputs), ("Output", outputs)):
        for index, tensor in enumerate(tensors):
            print(f"{kind} {index}: name={tensor['name']!r}, "
                  f"shape={tuple(int(n) for n in tensor['shape'])}, "
                  f"dtype={np.dtype(tensor['dtype']).name}, "
                  f"quantization={tensor.get('quantization')}", flush=True)
    try:
        header = model_header(inputs, outputs, args.coordinates)
    except ValueError as error:
        parser.error(str(error))
    Path(args.output).write_text(header, encoding="utf-8")
    print(f"Model parameters have been saved to {args.output}.")


if __name__ == "__main__":
    main()
