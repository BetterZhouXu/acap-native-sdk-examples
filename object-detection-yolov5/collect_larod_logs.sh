#!/bin/sh
# Collect system logs after reproducing a Larod model-load failure on the camera.
set -eu

if [ "$#" -ne 2 ]; then
    printf 'Usage: %s https://CAMERA_IP output.log\n' "$0" >&2
    exit 2
fi

camera=${1%/}
output=$2
printf 'Camera username: '
read -r username
printf 'curl will prompt for the camera password (not saved in this script).\n'

# Set CURL_INSECURE=1 only if the camera uses a self-signed HTTPS certificate.
if [ "${CURL_INSECURE:-0}" = 1 ]; then
    curl --insecure --digest --user "$username" --fail --show-error --silent \
        "$camera/axis-cgi/admin/systemlog.cgi" --output "$output"
else
    curl --digest --user "$username" --fail --show-error --silent \
        "$camera/axis-cgi/admin/systemlog.cgi" --output "$output"
fi

grep -i -E 'larod|tflite|delegate|dlpu|object_detection_yolov8|interpreter' \
    "$output" > "$output.larod" || true
printf 'Saved full camera log: %s\nRelevant lines: %s.larod\n' "$output" "$output"
