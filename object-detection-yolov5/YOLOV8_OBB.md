# YOLOv8 OBB on ARTPEC-8

This variant replaces the YOLOv5 example's detector with a **user-supplied**
ARTPEC-8-compatible, per-tensor INT8 TensorFlow Lite YOLOv8 OBB model. No model
weights or training labels are included in this repository.

Place these files in this directory before building:

```
model/model.tflite     # one signed INT8 NCHW input [1,3,640,640]
label/labels.txt       # four lines, in model class-channel order
```

The script `app/parameter_finder.py` rejects other input/output shapes,
types or missing per-tensor affine quantization. The single signed INT8
output must be `[1,9,8400]`, channel-major:

```
channel 0..3: center x, center y, width, height
channel 4..7: class 0..3 confidence (no YOLOv5 objectness channel)
channel 8:    angle in radians
```

The image is resized to 640×640 planar RGB by Larod cpu-proc and then
quantized into a separate INT8 model input using its input scale/zero point. The default
assumption is RGB pixels divided by 255 and **normalized** output xywh. If
your exported model expects unnormalized RGB pixels or returns xywh in pixels,
set the corresponding Docker build arguments to `0_255` or `pixels`. These
conventions **cannot be determined from tensor shape/quantization metadata**;
inspect your model/export pipeline. The angle must be in radians and the output
must contain decoded xywh and class confidence, not raw feature-map logits.

```sh
cd object-detection-yolov5
mkdir -p model label
# Copy your model into model/model.tflite and your four labels into label/labels.txt.
docker build --platform=linux/amd64 -t yolov8-obb-artpec8 \
  --build-arg ARCH=aarch64 --build-arg CHIP=artpec8 \
  --build-arg MODEL_INPUT_RANGE=0_1 \
  --build-arg MODEL_OUTPUT_COORDS=normalized .
docker cp $(docker create --platform=linux/amd64 yolov8-obb-artpec8):/opt/app ./build
```

The application runs on `axis-a8-dlpu-tflite`. It dequantizes each output
channel with the model output scale and zero point, selects the highest class
score per candidate, sorts by score, applies class-aware rotated polygon IoU
NMS (up to 300 candidates and 100 drawn detections), and draws quadrilaterals
through the Axis Bounding Box API. The application parameters
`ConfThresholdPercent` and `IouThresholdPercent` control filtering.

**Note:** A single INT8 scale shared by pixel-space coordinates, angle and
probabilities can severely degrade confidence and angle resolution. Check the
actual output scale and predicted values on a test image; prefer a normalized
xywh output if the model can be re-exported. A successful build alone does not
establish accuracy or ARTPEC-8 compatibility; test the model on the device.
