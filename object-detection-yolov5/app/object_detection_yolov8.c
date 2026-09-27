/**
 * Copyright (C) 2025, Axis Communications AB, Lund, Sweden
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * - object_detection_yolov8 (YOLOv8 OBB variant) -
 *
 * This application loads an INT8 YOLOv8 OBB model on ARTPEC-8. Its channel-major
 * output is decoded into oriented quadrilaterals, class scores and angles.
 *
 * The application expects two arguments on the command line in the
 * following order: MODELFILE LABELSFILE.
 *
 * First argument, MODELFILE, is a string describing path to the model.
 *
 * Second argument, LABELSFILE, is a string describing path to the label txt.
 *
 */

#include "argparse.h"
#include "imgprovider.h"
#include "labelparse.h"
#include "model.h"
#include "model_params.h"  //Generated at build time
#include "panic.h"
#include "vdo-error.h"
#include "vdo-frame.h"
#include "vdo-types.h"
#include <axsdk/axparameter.h>
#include <bbox.h>

#include <math.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <syslog.h>

#define APP_NAME "object_detection_yolov5"

volatile sig_atomic_t running = 1;

static void shutdown(int status) {
    (void)status;
    running = 0;
}

typedef struct model_params {
    int input_width;
    int input_height;
    float quantization_scale;
    float quantization_zero_point;
    int num_classes;
    int num_detections;
} model_params_t;

static int ax_parameter_get_int(AXParameter* handle, const char* name) {
    gchar* str_value = NULL;
    GError* error    = NULL;
    int value;

    // Get the value of the parameter
    if (!ax_parameter_get(handle, name, &str_value, &error)) {
        panic("%s", error->message);
    }

    // Convert the parameter value to int
    if (sscanf(str_value, "%d", &value) != 1) {
        panic("Axparameter %s was not an int", name);
    }

    syslog(LOG_INFO, "Axparameter %s: %s", name, str_value);

    g_free(str_value);

    return value;
}

static bbox_t* setup_bbox(void) {
    // Create box drawers
    bbox_t* bbox = bbox_view_new(1u);
    if (!bbox) {
        panic("Failed to create box drawer");
    }

    bbox_clear(bbox);
    const bbox_color_t red = bbox_color_from_rgb(0xff, 0x00, 0x00);

    bbox_style_outline(bbox);   // Switch to outline style
    bbox_thickness_thin(bbox);  // Switch to thin lines
    bbox_color(bbox, red);      // Switch to red

    return bbox;
}

static unsigned int elapsed_ms(struct timeval* start_ts, struct timeval* end_ts) {
    return (unsigned int)(((end_ts->tv_sec - start_ts->tv_sec) * 1000) +
                          ((end_ts->tv_usec - start_ts->tv_usec) / 1000));
}

typedef struct point { float x, y; } point_t;
typedef struct detection {
    point_t corners[4];
    float score, area;
    int label;
} detection_t;

static float channel(const int8_t* data, int channel_idx, int idx, const model_params_t* p) {
    return (data[channel_idx * p->num_detections + idx] - p->quantization_zero_point) *
           p->quantization_scale;
}

static int compare_scores(const void* a, const void* b) {
    const detection_t* left = a;
    const detection_t* right = b;
    return (right->score > left->score) - (right->score < left->score);
}

static float cross(point_t a, point_t b, point_t p) {
    return (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x);
}

static float clamp_coordinate(float value) {
    return fmaxf(0.0f, fminf(1.0f, value));
}

static float rotated_iou(const detection_t* a, const detection_t* b) {
    point_t poly[8], clipped[8];
    memcpy(poly, a->corners, sizeof(a->corners));
    int count = 4;
    for (int edge = 0; edge < 4 && count; ++edge) {
        point_t start = b->corners[edge], end = b->corners[(edge + 1) % 4];
        int next = 0;
        for (int i = 0; i < count; ++i) {
            point_t p = poly[i], q = poly[(i + 1) % count];
            float cp = cross(start, end, p), cq = cross(start, end, q);
            if ((cp >= 0) != (cq >= 0)) {
                float t = cp / (cp - cq);
                clipped[next++] = (point_t){p.x + t * (q.x - p.x), p.y + t * (q.y - p.y)};
            }
            if (cq >= 0) clipped[next++] = q;
        }
        count = next;
        memcpy(poly, clipped, (size_t)count * sizeof(point_t));
    }
    float twice_area = 0;
    for (int i = 0; i < count; ++i) {
        point_t p = poly[i], q = poly[(i + 1) % count];
        twice_area += p.x * q.y - p.y * q.x;
    }
    float intersection = fabsf(twice_area) / 2.0f;
    return intersection / (a->area + b->area - intersection);
}

static int decode_detections(const int8_t* data, const model_params_t* p,
                             float threshold, detection_t* detections) {
    int count = 0;
    for (int i = 0; i < p->num_detections; ++i) {
        float score = -INFINITY;
        int label = 0;
        for (int c = 0; c < p->num_classes; ++c) {
            float value = channel(data, 4 + c, i, p);  // YOLOv8 has no objectness channel
            if (value > score) { score = value; label = c; }
        }
        if (!isfinite(score) || score < threshold) continue;
        float x = channel(data, 0, i, p) / (OUTPUT_COORDS_NORMALIZED ? 1 : p->input_width);
        float y = channel(data, 1, i, p) / (OUTPUT_COORDS_NORMALIZED ? 1 : p->input_height);
        float w = channel(data, 2, i, p) / (OUTPUT_COORDS_NORMALIZED ? 1 : p->input_width);
        float h = channel(data, 3, i, p) / (OUTPUT_COORDS_NORMALIZED ? 1 : p->input_height);
        float angle = channel(data, 4 + p->num_classes, i, p);
        if (!isfinite(x) || !isfinite(y) || !isfinite(w) || !isfinite(h) ||
            !isfinite(angle) || w <= 0 || h <= 0) continue;
        detection_t* d = &detections[count++];
        d->score = score; d->label = label; d->area = w * h;
        float cs = cosf(angle), sn = sinf(angle);
        for (int k = 0; k < 4; ++k) {
            float dx = (k == 0 || k == 3 ? -w : w) / 2;
            float dy = (k < 2 ? -h : h) / 2;
            d->corners[k] = (point_t){x + dx * cs - dy * sn, y + dx * sn + dy * cs};
        }
    }
    qsort(detections, (size_t)count, sizeof(*detections), compare_scores);
    return count;
}

int main(int argc, char** argv) {
    g_autoptr(GError) vdo_error           = NULL;
    img_provider_t* image_provider        = NULL;
    model_provider_t* model_provider      = NULL;
    model_tensor_output_t* tensor_outputs = NULL;
    bbox_t* bbox                          = NULL;

    // Stop main loop at signal
    signal(SIGTERM, shutdown);
    signal(SIGINT, shutdown);

    args_t args;
    parse_args(argc, argv, &args);

    model_params_t* model_params = (model_params_t*)malloc(sizeof(model_params_t));
    if (model_params == NULL) {
        panic("%s: Unable to allocate model_params_t: %s", __func__, strerror(errno));
    }

    // Comes from model_params.h
    model_params->input_width             = MODEL_INPUT_WIDTH;
    model_params->input_height            = MODEL_INPUT_HEIGHT;
    model_params->quantization_scale      = QUANTIZATION_SCALE;
    model_params->quantization_zero_point = QUANTIZATION_ZERO_POINT;
    model_params->num_classes             = NUM_CLASSES;
    model_params->num_detections          = NUM_DETECTIONS;

    syslog(LOG_INFO,
           "Model input size w/h: %d x %d",
           model_params->input_width,
           model_params->input_height);
    syslog(LOG_INFO, "Quantization scale: %f", model_params->quantization_scale);
    syslog(LOG_INFO, "Quantization zero point: %f", model_params->quantization_zero_point);
    syslog(LOG_INFO, "Number of classes: %d", model_params->num_classes);
    syslog(LOG_INFO, "Number of detections: %d", model_params->num_detections);

    detection_t* detections = calloc((size_t)model_params->num_detections, sizeof(*detections));
    if (!detections) panic("Could not allocate detection buffer");

    // Create a new axparameter instance
    GError* axparameter_error       = NULL;
    AXParameter* axparameter_handle = ax_parameter_new(APP_NAME, &axparameter_error);
    if (axparameter_handle == NULL) {
        panic("%s", axparameter_error->message);
    }

    float conf_threshold = ax_parameter_get_int(axparameter_handle, "ConfThresholdPercent") / 100.0;
    float iou_threshold  = ax_parameter_get_int(axparameter_handle, "IouThresholdPercent") / 100.0;

    ax_parameter_free(axparameter_handle);

    VdoFormat vdo_format = VDO_FORMAT_YUV;
    double vdo_framerate = 30.0;

    if (!g_strcmp0(args.device_name, "a9-dlpu-tflite")) {
        // Possible to run RGB on ARTPEC-9
        vdo_format = VDO_FORMAT_RGB;
    }

    // Choose a valid stream resolution since only certain resolutions are allowed
    unsigned int stream_width  = 0;
    unsigned int stream_height = 0;
    if (!choose_stream_resolution(model_params->input_width,
                                  model_params->input_height,
                                  vdo_format,
                                  "native",
                                  "all",
                                  &stream_width,
                                  &stream_height)) {
        syslog(LOG_ERR, "%s: Failed choosing stream resolution", __func__);
        goto end;
    }
    syslog(LOG_INFO,
           "Creating VDO image provider and creating stream %u x %u",
           stream_width,
           stream_height);

    image_provider = create_img_provider(stream_width, stream_height, 2, vdo_format, vdo_framerate);
    if (!image_provider) {
        panic("%s: Could not create image provider", __func__);
    }

    size_t number_output_tensors = 0;
    model_provider               = create_model_provider(model_params->input_width,
                                           model_params->input_height,
                                           image_provider->width,
                                           image_provider->height,
                                           image_provider->pitch,
                                           image_provider->format,
                                           VDO_FORMAT_PLANAR_RGB,
                                           args.model_file,
                                           args.device_name,
                                           false,
                                           &number_output_tensors,
                                           INPUT_SCALE,
                                           INPUT_ZERO_POINT,
                                           INPUT_DIVISOR);
    if (!model_provider) {
        panic("%s: Could not create model provider", __func__);
    }
    if (number_output_tensors != 1) panic("Expected exactly one OBB output tensor");
    tensor_outputs = calloc(number_output_tensors, sizeof(model_tensor_output_t));
    if (!tensor_outputs) {
        panic("%s: Could not allocate tensor outputs", __func__);
    }

    char** labels = NULL;          // This is the array of label strings. The label
                                   // entries points into the large label_file_data buffer.
    size_t num_labels;             // Number of entries in the labels array.
    char* label_file_data = NULL;  // Buffer holding the complete collection of label strings.

    parse_labels(&labels, &label_file_data, args.labels_file, &num_labels);
    if (num_labels < (size_t)model_params->num_classes) {
        panic("Expected at least %d labels, got %zu", model_params->num_classes, num_labels);
    }

    syslog(LOG_INFO, "Start fetching video frames from VDO");
    if (!img_provider_start(image_provider)) {
        panic("%s: Could not start image provider", __func__);
    }

    bbox = setup_bbox();

    while (running) {
        struct timeval start_ts, end_ts;
        unsigned int preprocessing_ms = 0;
        unsigned int inference_ms     = 0;
        unsigned int total_elapsed_ms = 0;

        g_autoptr(VdoBuffer) vdo_buf = img_provider_get_frame(image_provider);
        if (!vdo_buf) {
            // This can only happen if it is global rotation then
            // the stream has to be restarted because rotation has been changed.
            syslog(
                LOG_INFO,
                "No buffer because of changed global rotation. Application needs to be restarted");
            goto end;
        }
        // If needed convert and scale/crop to correct input format and resolution
        // Its up to the model provider to decide if needed or not
        // If not needed the model_run_preprocessing will return true without
        // any work
        gettimeofday(&start_ts, NULL);
        if (!model_run_preprocessing(model_provider, vdo_buf)) {
            // No power
            if (!vdo_stream_buffer_unref(image_provider->vdo_stream, &vdo_buf, &vdo_error)) {
                if (!vdo_error_is_expected(&vdo_error)) {
                    panic("%s: Unexpexted error: %s", __func__, vdo_error->message);
                }
                g_clear_error(&vdo_error);
            }
            img_provider_flush_all_frames(image_provider);
            continue;
        }
        gettimeofday(&end_ts, NULL);

        preprocessing_ms = (unsigned int)(((end_ts.tv_sec - start_ts.tv_sec) * 1000) +
                                          ((end_ts.tv_usec - start_ts.tv_usec) / 1000));
        syslog(LOG_INFO, "Ran pre-processing for %u ms", preprocessing_ms);

        // Retrieve detections from data
        gettimeofday(&start_ts, NULL);
        if (!model_run_inference(model_provider, vdo_buf)) {
            // No power
            if (!vdo_stream_buffer_unref(image_provider->vdo_stream, &vdo_buf, &vdo_error)) {
                if (!vdo_error_is_expected(&vdo_error)) {
                    panic("%s: Unexpexted error: %s", __func__, vdo_error->message);
                }
                g_clear_error(&vdo_error);
            }
            img_provider_flush_all_frames(image_provider);
            continue;
        }
        gettimeofday(&end_ts, NULL);

        inference_ms = (unsigned int)(((end_ts.tv_sec - start_ts.tv_sec) * 1000) +
                                      ((end_ts.tv_usec - start_ts.tv_usec) / 1000));
        syslog(LOG_INFO, "Ran inference for %u ms", inference_ms);

        total_elapsed_ms = inference_ms + preprocessing_ms;

        // Check if the framerate from vdo should be changed
        img_provider_update_framerate(image_provider, total_elapsed_ms);

        for (size_t i = 0; i < number_output_tensors; i++) {
            if (!model_get_tensor_output_info(model_provider, i, &tensor_outputs[i])) {
                panic("Failed to get output tensor info for %zu", i);
            }
        }

        if (tensor_outputs[0].size != (size_t)(9 * model_params->num_detections)) {
            panic("Unexpected OBB output size %zu", tensor_outputs[0].size);
        }
        const int8_t* tensor_data = tensor_outputs[0].data;
        // Parse the output
        gettimeofday(&start_ts, NULL);
        int count = decode_detections(tensor_data, model_params, conf_threshold, detections);
        gettimeofday(&end_ts, NULL);
        syslog(LOG_INFO, "Ran parsing for %u ms", elapsed_ms(&start_ts, &end_ts));

        bbox_clear(bbox);

        // Bound polygon comparisons and overlay cost on dense frames.
        if (count > 300) count = 300;
        int kept[100], kept_count = 0;
        for (int i = 0; i < count && kept_count < 100; ++i) {
            detection_t* d = &detections[i];
            bool suppressed = false;
            for (int j = 0; j < kept_count; ++j) {
                detection_t* other = &detections[kept[j]];
                if (other->label == d->label && rotated_iou(d, other) > iou_threshold) {
                    suppressed = true;
                    break;
                }
            }
            if (suppressed) continue;
            kept[kept_count++] = i;
            syslog(LOG_INFO, "Object %d: Label=%s, Confidence=%.2f",
                   kept_count, labels[d->label], d->score);
            bbox_coordinates_frame_normalized(bbox);
            bbox_quad(bbox, clamp_coordinate(d->corners[0].x), clamp_coordinate(d->corners[0].y),
                      clamp_coordinate(d->corners[1].x), clamp_coordinate(d->corners[1].y),
                      clamp_coordinate(d->corners[2].x), clamp_coordinate(d->corners[2].y),
                      clamp_coordinate(d->corners[3].x), clamp_coordinate(d->corners[3].y));
        }

        if (!bbox_commit(bbox, 0u)) {
            panic("Failed to commit box drawer");
        }

        // This will allow vdo to fill this buffer with data again
        if (!vdo_stream_buffer_unref(image_provider->vdo_stream, &vdo_buf, &vdo_error)) {
            if (!vdo_error_is_expected(&vdo_error)) {
                panic("%s: Unexpexted error: %s", __func__, vdo_error->message);
            }
            g_clear_error(&vdo_error);
        }
    }

end:
    // Cleanup
    free(model_params);
    free(detections);
    if (image_provider) {
        destroy_img_provider(image_provider);
    }
    if (model_provider) {
        destroy_model_provider(model_provider);
    }
    free(tensor_outputs);
    free(labels);
    free(label_file_data);
    bbox_destroy(bbox);

    syslog(LOG_INFO, "Exit %s", argv[0]);

    return 0;
}

