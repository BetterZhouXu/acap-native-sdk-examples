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
# Build ACAP application
#-------------------------------------------------------------------------------

WORKDIR /opt/app
COPY ./app .

# Explicit asset copies make model or label changes visible as separate Docker layers.
COPY ./app/model/model.tflite ./model/model.tflite
COPY ./app/label/labels.txt ./label/labels.txt

ARG CHIP=artpec8

# Print hashes in the build log so the packaged inputs can be verified.
RUN test -s model/model.tflite && test -s label/labels.txt && \
    sha256sum model/model.tflite label/labels.txt

# Extract and validate YOLOv8 OBB model parameters.
# hadolint ignore=SC1091
RUN . /opt/venv/bin/activate && \
    python parameter_finder.py 'model/model.tflite' 'label/labels.txt'

RUN cp /opt/app/manifest.json.${CHIP} /opt/app/manifest.json && \
    . /opt/axis/acapsdk/environment-setup* && acap-build . \
    -a 'label/labels.txt' \
    -a 'model/model.tflite'
