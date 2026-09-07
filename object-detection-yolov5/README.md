# YOLOv8 OBB detection with a FastCGI results API

This directory upgrades the original YOLOv5 example to YOLOv8 oriented bounding
boxes. The directory and main source filename are retained, but the installed
application/binary is **detection**, making the endpoint exactly:

**GET http://CAMERA_IP/local/detection/events.cgi**

The VDO/larod loop performs inference, class-aware rotated NMS, and rotated edge
overlays. It saves recent detections to
`/usr/local/packages/detection/localdata/events.jsonl`. A separate FastCGI worker
reads this file, so a slow HTTP client does not block inference. Atomic file
replacement and a separate file lock prevent partial JSON lines and conflicting
updates. After successfully sending a response, the worker removes the returned
records, preserving detections that arrived while the response was being sent.
The response contains axis-aligned bounding boxes, without angles or corner arrays.

## Model and labels

Supply these files before building (no default model is downloaded):

- `app/model/model.tflite`: your camera-compatible YOLOv8 OBB TFLite model.
- `app/label/labels.txt`: exactly four UTF-8 labels, one per line, in model class
  order 0–3. No numeric prefixes, blank lines, or background label. A final newline
  and CRLF are supported. Labels must not exceed 256 bytes.

The model must have:

- One static **uint8 RGB NHWC input** `[1, height, width, 3]` accepting raw RGB
  bytes from the existing VDO preprocessing pipeline. Float32/int8 inputs require
  different input conversion and are rejected rather than silently misinterpreted.
- One tightly packed **channel-major output `[1, 9, 18900]`**. Channels 0–3 are
  `cx, cy, width, height`; channels 4–7 are the four class probabilities; channel
  8 is the already-decoded angle in **radians**. There is **no objectness channel**.
- Output float32, uint8, or int8. Quantized outputs must have per-tensor scale and
  zero point. Build-time inspection does not allocate/run model tensors.

Shape alone cannot establish coordinate units. Set `OBB_COORDINATES=normalized`
for exports whose x/width are normalized by input width and y/height by input
height (common for Ultralytics TFLite exports), or `OBB_COORDINATES=pixels` for
model-input pixel coordinates. Verify your export rather than guessing. The
default is `normalized`.

The decoder still uses the model's angle internally for correct OBB suppression
and overlays; callers do not need to handle it. It converts coordinates to model
pixels **before** rotating corners and computing IoU. Class probabilities and
angles must already be decoded; this is not
a raw-head/logit/DFL decoder. NMS sorts by descending best class probability,
compares only the same class, and computes polygon-intersection IoU. It considers
the best 1,000 candidates and returns at most 100 detections per frame. This uses
exact rotated IoU, not Ultralytics' probabilistic IoU approximation, so suppression
may differ from Ultralytics defaults.

The existing preprocessing resizes the entire image without letterboxing or
cropping. Overlays and normalized bounding boxes map back to that full image. If your
training/export expects letterboxing, adapt preprocessing and the inverse mapping
before deployment. Returned boxes are the axis-aligned envelopes of the oriented
detections, expressed in model-input pixels and normalized full-image coordinates.

## Build and install

From this directory, with the two supplied assets in place:

```sh
docker build --platform=linux/amd64 -t yolov8-obb:1.0 \
  --build-arg ARCH=aarch64 --build-arg CHIP=artpec8 \
  --build-arg OBB_COORDINATES=normalized .
container_id=$(docker create --platform=linux/amd64 yolov8-obb:1.0)
docker cp "${container_id}:/opt/app" ./build
docker rm "$container_id"
```

Choose `CHIP=artpec8`, `artpec9`, or `cpu`, with the matching model and camera
architecture (`ARCH=aarch64` or `armv7hf`). A generic Ultralytics export is **not
automatically compiled for the camera DLPU** by this Dockerfile. Use the appropriate
Axis model-conversion tools and confirm that the resulting model retains the
required I/O contract. No trained model or labels are included in this change.

Install the `.eap` from `build` through the camera's **Apps** page, then start it.
On Apple Silicon, Docker Desktop's Rosetta emulation may need to be disabled for
the SDK/TensorFlow build. The container installs TensorFlow (and its NumPy
dependency) only for model inspection; neither is needed on the camera.

**Upgrade note:** the package name has changed from `send_event` to `detection`.
Stop/uninstall the previous detection package before installing this build, or
the old application may continue running separately. This application does not
send ONVIF events.

The manifests register `events.cgi` as `fastCgi` with `viewer` access. Axis's web
server handles camera authentication, including HTTP Digest when enabled. Viewer,
operator, or administrator credentials may read detections; change `access` to
`admin` in all manifests if tighter access is required. No credentials are stored
in the app and no separate TCP port is opened.

## Polling response contract

Set the polling client's endpoint constant to `local/detection/events.cgi`.
The supplied requests/Digest GET works with this endpoint. Successful responses
are `200 OK`, `Content-Type: application/x-ndjson; charset=utf-8`, and
`Cache-Control: no-store`. Decode **each nonempty response line separately**, not
the complete body as a JSON array. Unsupported methods return 405; local-file read
failures return 500. Query parameters are currently ignored.

Each line is one detected object, with these fields:

| Field | Meaning |
| --- | --- |
| `schema_version` | Integer `2` |
| `event_id` | Per-run UUID plus monotonic detection sequence, separated by `:` |
| `frame_id` | Monotonic inference-frame sequence within the run |
| `timestamp_us` | Camera Unix epoch microseconds when the frame was fetched, not sensor exposure time |
| `class_id` | Integer 0–3 |
| `label` | Corresponding UTF-8 label, JSON-escaped |
| `confidence` | Best class probability, in [0, 1] |
| `model_input` | `[width, height]` in pixels |
| `bbox` | Axis-aligned envelope with `cx`, `cy`, `width`, `height` in model pixels |
| `bbox_normalized` | `[left, top, right, bottom]`, divided by model width/height |

No angle or oriented-corner fields are returned. Boxes are not clamped in JSON,
preserving true geometry for detections crossing an image edge; the overlay clips
edges to the viewport.

`CameraDetectionPayload` was not defined in the supplied polling snippet. Its
parser/model must accept or map the fields above; endpoint compatibility alone
does not guarantee compatibility with that unseen payload class.

### Retention and empty results

- Polls return pending results in chronological insertion order. After successfully
  flushing the response to Axis's HTTP server, GET **clears the returned records**.
  New detections saved during transmission remain for the next poll. If none have
  arrived, the next poll returns 200 with an empty body.
- File updates are locked, but the lock is not held during network transmission.
  Successfully persisted records are removed from the producer's in-memory queue,
  so a later frame cannot republish results consumed by a poll.
- Failed transfers do not clear records. A crash or clearing failure after sending
  may cause duplicates; use `event_id` if deduplication is needed. FastCGI can only
  confirm delivery to the HTTP server, not that the remote client processed the
  body. This is not an exactly-once delivery protocol. Multiple pollers consume
  from the same queue rather than each receiving their own copy.
- At most **1,000 records or 1 MiB**, whichever is reached first, are retained.
  Older records are evicted. Slow polling can therefore lose old detections.
- Startup clears the previous run's file and creates a new session UUID. Before
  any detection, the endpoint returns 200 with an empty body.
- Frames without detections add no records. Unconsumed detections remain available;
  an unchanged response is not evidence that the objects are still present. Use
  `timestamp_us` and `event_id` to determine freshness. Synchronize camera time.
- There is no object tracker: the same physical object on successive frames
  produces different detection events.
- Snapshots are rewritten once per detection-bearing frame (also retried after a
  write failure). Failed writes are logged and the last complete snapshot remains
  readable. For high-rate continuous deployments, consider external storage or
  batching to reduce camera flash wear; this is not a lossless archival queue.
- The FastCGI worker stops when inference exits, including abnormal parent
  termination. If the worker dies, the detection loop fails rather than silently
  continuing without its API.

## Configuration and checks

ACAP settings `ConfThresholdPercent` (default 25) and `IouThresholdPercent`
(default 45) take effect after restarting the application.

After installation, check:

1. The app starts with your model; logs show four classes and 18,900 candidates.
2. Digest-authenticated GET returns an empty 200 before the first detection and
   valid newline-delimited JSON after detection. Unauthenticated requests are
   rejected by Axis; POST returns 405.
3. Rotated overlays and returned axis-aligned boxes agree with the image, including objects
   near the frame boundary. Confirm the coordinate-unit build setting.
4. A successful poll clears returned results; new detections generated during a
   poll remain for the next query. Concurrent reads never see partial lines,
   retention stays bounded, and restarting resets the session/queue.
5. The FastCGI worker stops along with the app and a restart can bind its socket.

Camera/model inference, ACAP cross-compilation, and Digest routing require testing
with the target SDK and camera. The historical YOLOv5 explanation is preserved in
[README.yolov5.md](README.yolov5.md), but its build/parser instructions no longer
apply to this application.

## License

Source: [Apache License 2.0](../LICENSE). Review and update `app/LICENSE` for the
actual supplied model and its licensing terms before distributing the package.
The previous example's model notices are not proof of your model's license.
