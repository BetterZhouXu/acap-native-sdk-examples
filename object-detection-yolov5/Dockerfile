ARG ARCH=aarch64
ARG VERSION=12.5.0
ARG UBUNTU_VERSION=24.04
ARG REPO=axisecp
ARG SDK=acap-native-sdk

FROM ${REPO}/${SDK}:${VERSION}-${ARCH}-ubuntu${UBUNTU_VERSION}

#-------------------------------------------------------------------------------
# Install TensorFlow (only used to inspect the model)
#-------------------------------------------------------------------------------

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Create a virtual environment for installations using pip
RUN python3 -m venv /opt/venv

# hadolint ignore=SC1091,DL3013
RUN . /opt/venv/bin/activate && pip install --no-cache-dir tensorflow

#-------------------------------------------------------------------------------
# Get YOLOv5 model and labels
#-------------------------------------------------------------------------------

# Download pre-trained YOLOv5 model
WORKDIR /opt/app/model
ARG CHIP=artpec8
COPY ./app/model/model.tflite /opt/app/model/model.tflite

# Download labels for model
WORKDIR /opt/app/label
COPY ./app/label/labels.txt /opt/app/label/labels.txt

#-------------------------------------------------------------------------------
# Build ACAP application
#-------------------------------------------------------------------------------

WORKDIR /opt/app
COPY ./app .

# Extract model parameters using the virtual environment
# hadolint ignore=SC1091
RUN . /opt/venv/bin/activate && python parameter_finder.py 'model/model.tflite'

RUN cp /opt/app/manifest.json.${CHIP} /opt/app/manifest.json && \
    . /opt/axis/acapsdk/environment-setup* && acap-build . \
    -a 'label/labels.txt' \
    -a 'model/model.tflite'

RUN apt-get update && apt-get install -y patchelf
RUN patchelf --set-rpath '$ORIGIN/lib' /opt/app/object_detection_yolov5

# (Optional) Copy libbbox and strip binary
RUN mkdir -p /opt/app/lib
RUN cp /opt/axis/acapsdk/sysroots/aarch64/usr/lib/libbbox.so.1 /opt/app/lib/
