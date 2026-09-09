*Copyright (C) 2025, Axis Communications AB, Lund, Sweden. All Rights Reserved.*

<!-- omit from toc -->
# Object detection with YOLOv8 OBB

This ACAP example runs a locally supplied Ultralytics YOLOv8 oriented-bounding-box (OBB) TensorFlow Lite model with Larod. It captures video through VDO, converts it to normalized float32 NCHW input, performs class-aware rotated non-maximum suppression, draws quadrilaterals with the Bounding Box API, and publishes detection events through FastCGI.

## Model contract

The application validates this model contract during the Docker build:

- Input: float32 `[1, 3, height, width]` (NCHW)
- Output: float32 `[1, 4 + classes + 1, predictions]`
- Output channels: `center_x`, `center_y`, `width`, `height`, one score per class, then angle in radians
- OBB `center_x`, `center_y`, `width`, and `height` are expected to already be normalized to `[0,1]`

For the supplied model, input `[1,3,960,960]` and output `[1,9,18900]` imply four classes and 18,900 candidate predictions. The model may use INT8 internally while exposing float32 input and output; no application-side output dequantization is performed.

## Local model and labels

Place the files here before building:

```text
object-detection-yolov5/app/model/model.tflite
object-detection-yolov5/app/label/labels.txt
```

`labels.txt` must contain exactly one non-empty label per line, in the same order as the model's class-score channels. The build fails if the effective label count differs from the model class count. Legacy lines containing exactly `n/a` are ignored by the runtime parser and the validator.

Both files are copied from the local build context. No model or labels are downloaded. Their SHA-256 hashes are printed during the Docker build so the exact assets can be verified.

## Detection processing

For every frame, the application:

1. Captures a YUV or RGB frame with VDO.
2. Resizes and converts it to planar RGB.
3. Converts each byte channel value to float32 in `[0,1]`.
4. Runs the model through Larod.
5. Reads the channel-first float output directly.
6. Selects the highest class score for each candidate.
7. Applies the configured confidence threshold.
8. Applies class-aware rotated-IoU NMS.
9. Keeps at most 300 highest-confidence detections.
10. Draws each OBB as a quadrilateral and saves the event as JSON.

The confidence and IoU thresholds remain configurable through the application settings.

## Detection-result API

A frame containing detections is atomically stored at:

```text
/usr/local/packages/detection/localdata/detection-result.json
```

Query the viewer-authenticated FastCGI endpoint:

```sh
curl --anyauth --user '<USER>:<PASSWORD>' \
  'http://<AXIS_DEVICE_IP>/local/detection/events.cgi'
```

Responses:

- `200 OK`: returns one pending event and removes its claimed JSON file after a successful flush.
- `204 No Content`: no event is pending.
- `405 Method Not Allowed`: only `GET` is supported; the pending event is not consumed.
- `500 Internal Server Error`: the result could not be claimed or read.

A failed response retains the event for retry. A new event produced while another is being served is kept separately and cannot be deleted by the first request.

Example response:

```json
{
  "timestampUnixMs": 1788883200123,
  "detections": [
    {
      "label": "forklift",
      "confidence": 0.91,
      "orientedBox": {
        "centerX": 0.51,
        "centerY": 0.46,
        "width": 0.22,
        "height": 0.11,
        "angleRadians": 0.37,
        "corners": [
          [0.43, 0.37],
          [0.64, 0.45],
          [0.59, 0.55],
          [0.39, 0.47]
        ]
      }
    }
  ]
}
```

Coordinates are normalized to the model frame. Corners can extend slightly outside `[0,1]` when an object crosses a frame boundary; display coordinates are clipped before drawing.

## Build

From `object-detection-yolov5`:

```sh
rm -rf build

docker build \
  --no-cache \
  --progress=plain \
  --platform=linux/amd64 \
  --tag detection:1.0 \
  --build-arg ARCH=aarch64 \
  --build-arg CHIP=artpec9 \
  .
```

Supported `CHIP` values correspond to the included manifests: `artpec8`, `artpec9`, and `cpu`. Choose an SDK architecture and Larod device compatible with the local model.

Extract artifacts without mixing them with an older build:

```sh
container_id=$(docker create --platform=linux/amd64 detection:1.0)
rm -rf build
mkdir build
docker cp "$container_id":/opt/app/. ./build/
docker rm "$container_id"
```

Check the build log's `sha256sum` output against local assets if stale files are suspected:

```sh
shasum -a 256 app/model/model.tflite app/label/labels.txt
```

## Host tests

The OBB decoder and atomic result-file handoff have standalone tests that do not require TensorFlow
or the ACAP SDK:

```sh
cd app
make test-host
```

The tests cover channel-first decoding, rotated corners, confidence sorting, class-aware rotated NMS,
the 300-result cap, invalid metadata, atomic publishing, response retry, and consume-on-read cleanup.

## Install and test

Install the generated `.eap` from `build/`, start the application, and query the endpoint shown above. Because the internal application name is `detection`, uninstall an older package with the same application name before installing the rebuilt package.

Application logs are available at:

```text
http://<AXIS_DEVICE_IP>/axis-cgi/admin/systemlog.cgi?appname=detection
```

## License

[Apache License 2.0](../LICENSE)
