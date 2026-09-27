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
RUN . /opt/venv/bin/activate && pip install --no-cache-dir 'tensorflow==2.16.2' 'numpy<2'

#-------------------------------------------------------------------------------
# Build ACAP application
#-------------------------------------------------------------------------------

WORKDIR /opt/app
# Includes app/model/model.tflite and app/label/labels.txt.
COPY ./app .
ARG CHIP=artpec8
RUN test "$CHIP" = artpec8 && test -s model/model.tflite && test -s label/labels.txt

# Extract model parameters using the virtual environment
ARG MODEL_INPUT_RANGE=0_1
# hadolint ignore=SC1091
RUN . /opt/venv/bin/activate && \
    python export_obb_raw.py model/model.tflite model/obb_raw.tflite && \
    python parameter_finder.py model/obb_raw.tflite

RUN cp /opt/app/manifest.json.${CHIP} /opt/app/manifest.json && \
    . /opt/axis/acapsdk/environment-setup* && acap-build . \
    -a 'label/labels.txt' \
    -a 'model/obb_raw.tflite'
