ARG ARCH=aarch64
ARG VERSION=12.8.0
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
# Supply the ARTPEC-8-compatible YOLOv8 OBB model and its four labels locally.
#-------------------------------------------------------------------------------

WORKDIR /opt/app/model
ARG CHIP=artpec8
RUN test "$CHIP" = artpec8
COPY ./model/model.tflite ./model.tflite

WORKDIR /opt/app/label
COPY ./label/labels.txt ./labels.txt

#-------------------------------------------------------------------------------
# Build ACAP application
#-------------------------------------------------------------------------------

WORKDIR /opt/app
COPY ./app .

# Extract model parameters using the virtual environment
ARG MODEL_INPUT_RANGE=0_1
ARG MODEL_OUTPUT_COORDS=normalized
# hadolint ignore=SC1091
RUN . /opt/venv/bin/activate && python parameter_finder.py 'model/model.tflite'

RUN cp /opt/app/manifest.json.${CHIP} /opt/app/manifest.json && \
    . /opt/axis/acapsdk/environment-setup* && acap-build . \
    -a 'label/labels.txt' \
    -a 'model/model.tflite'
