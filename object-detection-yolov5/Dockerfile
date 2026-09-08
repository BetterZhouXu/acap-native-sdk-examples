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
# Get YOLOv5 model
#-------------------------------------------------------------------------------

# Download pre-trained YOLOv5 model
WORKDIR /opt/app/model
ARG CHIP=artpec8
ARG MODEL_BUCKET=https://acap-ml-models.s3.amazonaws.com/yolov5
RUN if [ "$CHIP" = cpu ] || [ "$CHIP" = artpec8 ]; then \
        curl -L -o model.tflite \
        $MODEL_BUCKET/yolov5n_artpec8_coco_640.tflite ; \
    elif [ "$CHIP" = artpec9 ]; then \
        curl -L -o model.tflite \
        $MODEL_BUCKET/yolov5n_artpec9_coco_640.tflite ; \
    else \
        printf "Error: '%s' is not a valid value for the CHIP variable\n", "$CHIP"; \
        exit 1; \
    fi

#-------------------------------------------------------------------------------
# Build ACAP application
#-------------------------------------------------------------------------------

WORKDIR /opt/app
# This also copies the model-specific labels from app/label/labels.txt.
COPY ./app .

# Extract model parameters using the virtual environment
# hadolint ignore=SC1091
RUN . /opt/venv/bin/activate && python parameter_finder.py 'model/model.tflite'

RUN cp /opt/app/manifest.json.${CHIP} /opt/app/manifest.json && \
    . /opt/axis/acapsdk/environment-setup* && acap-build . \
    -a 'label/labels.txt' \
    -a 'model/model.tflite'
