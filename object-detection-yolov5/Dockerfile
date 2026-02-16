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

RUN python3 -m venv /opt/venv
RUN . /opt/venv/bin/activate && pip install --no-cache-dir tensorflow

#-------------------------------------------------------------------------------
# Use Custom Model and Labels
#-------------------------------------------------------------------------------

WORKDIR /opt/app/model
COPY ./app/model/model.tflite model.tflite

WORKDIR /opt/app/label
COPY ./app/label/labels.txt labels.txt

#-------------------------------------------------------------------------------
# Build ACAP application
#-------------------------------------------------------------------------------

WORKDIR /opt/app
COPY ./app .

# Extract model parameters
RUN . /opt/venv/bin/activate && python parameter_finder.py model/model.tflite

# Build ACAP
RUN cp manifest.json.${CHIP} manifest.json && \
    . /opt/axis/acapsdk/environment-setup* && acap-build . \
    -a label/labels.txt \
    -a model/model.tflite
