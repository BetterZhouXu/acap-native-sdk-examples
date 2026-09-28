# YOLOv8 OBB on ARTPEC-8

This variant replaces the YOLOv5 example's detector with a **user-supplied**
ARTPEC-8-compatible, per-tensor INT8 TensorFlow Lite YOLOv8 OBB model. The
model and its four labels are supplied under `app/`.

Place these files in this directory before building:

```
app/model/model.tflite     # one signed INT8 NCHW input [1,3,640,640]
app/label/labels.txt       # four lines, in model class-channel order
```

The bundled original model has `[1,9,8400]` output but contains float32
`DEQUANTIZE`, `COS` and `SIN` ops. During the Docker build,
`app/export_obb_raw.py` converts this **specific model** (checked by SHA-256)
into `model/obb_raw.tflite`, without those ops. It fails rather than silently
altering a different model. `app/parameter_finder.py` checks the resulting
per-tensor INT8 input and the three INT8 channel-major outputs:

```
output 0 [1,4,8400]: DFL distances left, top, right, bottom (grid cells)
output 1 [1,4,8400]: class 0..3 confidence (no objectness channel)
output 2 [1,1,8400]: angle in radians
```

The image is resized to 640x640 planar RGB by Larod cpu-proc and then
quantized into a separate INT8 model input using its input scale/zero point.
The default assumes RGB pixels divided by 255; set `MODEL_INPUT_RANGE=0_255`
if your export expects unnormalized pixels. The C code decodes the 80x80,
40x40 and 20x20 anchor grids (strides 8, 16, 32), applies angle sine/cosine
to the DFL box-center offset, and draws rotated boxes. These output semantics
are specific to this model; the original `[1,9,8400]` model is NOT packaged.

```sh
cd object-detection-yolov5
mkdir -p app/model app/label
# Copy your model into app/model/model.tflite and your four labels into app/label/labels.txt.
docker build --platform=linux/amd64 -t yolov8-obb-artpec8 \
  --build-arg ARCH=aarch64 --build-arg CHIP=artpec8 \
  --build-arg MODEL_INPUT_RANGE=0_1 .
docker cp $(docker create --platform=linux/amd64 yolov8-obb-artpec8):/opt/app ./build
```

The app/executable name is `object_detection_yolov8`; the EAP file is named
`object_detection_yolov8_artpec8_1_0_0_<ARCH>.eap`. This is a new package
identity, not an in-place upgrade of `object_detection_yolov5`. Uninstall the
old package separately if it is still on the camera.

The application runs on `axis-a8-dlpu-tflite`. It dequantizes each output
with its own scale and zero point, selects the highest class
score per candidate, sorts by score, applies class-aware rotated polygon IoU
NMS (up to 300 candidates and 100 drawn detections), and draws quadrilaterals
through the Axis Bounding Box API. The application parameters
`ConfThresholdPercent` and `IouThresholdPercent` control filtering.

The three outputs now retain separate quantization scales, avoiding a shared
scale across coordinates, angles, and probabilities. A desktop CPU invocation
of the transformed model succeeds. **This does not prove ARTPEC-8 compatibility**:
other operators may still fail on the camera, so test on the device.

## Collect Larod startup logs

After starting the application and reproducing the failure, collect the full
camera system log with the provided script from your computer:

```sh
./collect_larod_logs.sh https://CAMERA_IP startup.log
```

It prompts for camera credentials and saves both the full log and a filtered
`startup.log.larod` excerpt. Include the Larod/delegate lines **before** the
`failure when invoking interpreter` line when reporting an issue; redact
credentials, IPs, and other sensitive camera details before sharing. For an
HTTPS camera with a self-signed certificate, prefix the command with
`CURL_INSECURE=1` only if you trust the camera/network. For an
SSH-enabled device, `journalctl -b -u larod --no-pager -n 300` may offer
additional service-level errors. Neither Docker nor the local workspace can
read camera logs without access to the camera.
