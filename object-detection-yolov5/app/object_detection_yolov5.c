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
 * - YOLOv8 oriented object detection and JSONL events -
 *
 * Loads a larod YOLOv8 OBB model with output [1, 9, 18900], draws oriented boxes,
 * and serves saved detections at /local/detection/events.cgi.
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
#include "events.h"
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
#include <sys/time.h>
#include <syslog.h>

#define APP_NAME "detection"

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

static void draw_obb(bbox_t* bbox, const obb_detection_t* detection) {
    bbox_coordinates_frame_normalized(bbox);
    for (int i = 0; i < 4; ++i) {
        obb_point_t a = detection->corners[i], b = detection->corners[(i + 1) % 4];
        float x = a.x / MODEL_INPUT_WIDTH, y = a.y / MODEL_INPUT_HEIGHT;
        float dx = (b.x - a.x) / MODEL_INPUT_WIDTH;
        float dy = (b.y - a.y) / MODEL_INPUT_HEIGHT;
        /* Clip each rotated edge to the viewport without distorting its angle. */
        float p[4] = {-dx, dx, -dy, dy};
        float q[4] = {x, 1 - x, y, 1 - y};
        float first = 0, last = 1;
        bool visible = true;
        for (int j = 0; j < 4; ++j) {
            if (fabsf(p[j]) < 1e-8f) {
                if (q[j] < 0) visible = false;
            } else if (p[j] < 0) {
                first = fmaxf(first, q[j] / p[j]);
            } else {
                last = fminf(last, q[j] / p[j]);
            }
        }
        if (!visible || first > last) continue;
        bbox_move_to(bbox, CLAMP(x + first * dx, 0, 1), CLAMP(y + first * dy, 0, 1));
        bbox_line_to(bbox, CLAMP(x + last * dx, 0, 1), CLAMP(y + last * dy, 0, 1));
        bbox_draw_path(bbox);
    }
}

int main(int argc, char** argv) {
    g_autoptr(GError) vdo_error           = NULL;
    img_provider_t* image_provider        = NULL;
    model_provider_t* model_provider      = NULL;
    model_tensor_output_t* tensor_outputs = NULL;
    bbox_t* bbox                          = NULL;
    char** labels = NULL;
    char* label_file_data = NULL;
    size_t num_labels = 0;
    int exit_status = EXIT_FAILURE;

    // Stop main loop at signal
    signal(SIGTERM, shutdown);
    signal(SIGINT, shutdown);

    args_t args;
    parse_args(argc, argv, &args);
    if (!events_start()) panic("Failed to start detection result store/FastCGI server");

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
    syslog(LOG_INFO, "Model input layout: %s", MODEL_INPUT_LAYOUT_NCHW ? "NCHW" : "NHWC");
    syslog(LOG_INFO, "Quantization scale: %f", model_params->quantization_scale);
    syslog(LOG_INFO, "Quantization zero point: %f", model_params->quantization_zero_point);
    syslog(LOG_INFO, "Number of classes: %d", model_params->num_classes);
    syslog(LOG_INFO, "Number of detections: %d", model_params->num_detections);

    obb_config_t obb_config = {
        .candidates = NUM_DETECTIONS, .classes = NUM_CLASSES,
        .input_width = MODEL_INPUT_WIDTH, .input_height = MODEL_INPUT_HEIGHT,
        .dtype = MODEL_OUTPUT_DTYPE, .scale = QUANTIZATION_SCALE,
        .zero_point = QUANTIZATION_ZERO_POINT, .normalized = MODEL_COORDINATES_NORMALIZED};
    obb_detection_t detections[OBB_MAX_DETECTIONS];

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
    VdoFormat model_format = MODEL_INPUT_LAYOUT_NCHW ? VDO_FORMAT_PLANAR_RGB : VDO_FORMAT_RGB;
    model_provider               = create_model_provider(model_params->input_width,
                                           model_params->input_height,
                                           image_provider->width,
                                           image_provider->height,
                                           image_provider->pitch,
                                           image_provider->format,
                                           model_format,
                                           args.model_file,
                                           args.device_name,
                                           false,
                                           &number_output_tensors);
    if (!model_provider) {
        panic("%s: Could not create model provider", __func__);
    }
    larodError* tensor_error = NULL;
    if (number_output_tensors != 1) panic("YOLOv8 OBB requires one output tensor");
    const larodTensorDims* dims = larodGetTensorDims(model_provider->output_tensors[0], &tensor_error);
    if (!dims || dims->len != 3 || dims->dims[0] != 1 || dims->dims[1] != 9 ||
        dims->dims[2] != NUM_DETECTIONS) panic("Expected YOLOv8 OBB output [1,9,18900]");
    if (larodGetTensorDataType(model_provider->input_tensors[0], &tensor_error) !=
        LAROD_TENSOR_DATA_TYPE_UINT8) panic("The VDO pipeline requires a uint8 RGB model input");
    tensor_outputs = calloc(number_output_tensors, sizeof(model_tensor_output_t));
    if (!tensor_outputs) {
        panic("%s: Could not allocate tensor outputs", __func__);
    }

    parse_labels(&labels, &label_file_data, args.labels_file, &num_labels);
    if (num_labels != NUM_CLASSES) panic("Expected exactly four class labels");
    for (size_t i = 0; i < num_labels; ++i) {
        if (!g_utf8_validate(labels[i], -1, NULL) || strlen(labels[i]) > 256)
            panic("Class labels must be UTF-8 and at most 256 bytes");
    }

    syslog(LOG_INFO, "Start fetching video frames from VDO");
    if (!img_provider_start(image_provider)) {
        panic("%s: Could not start image provider", __func__);
    }

    bbox = setup_bbox();

    while (running) {
        if (!events_server_alive()) panic("FastCGI worker exited unexpectedly");
        struct timeval start_ts, end_ts;
        unsigned int preprocessing_ms = 0;
        unsigned int inference_ms     = 0;
        unsigned int total_elapsed_ms = 0;

        g_autoptr(VdoBuffer) vdo_buf = img_provider_get_frame(image_provider);
        gint64 capture_time_us = g_get_real_time();
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

        gettimeofday(&start_ts, NULL);
        int valid_detection_count = obb_decode(tensor_outputs[0].data, tensor_outputs[0].size,
                                               &obb_config, conf_threshold, iou_threshold, detections);
        if (valid_detection_count < 0) panic("Invalid OBB output buffer or decoder configuration");
        gettimeofday(&end_ts, NULL);
        syslog(LOG_INFO, "Ran parsing for %u ms", elapsed_ms(&start_ts, &end_ts));

        bbox_clear(bbox);
        for (int i = 0; i < valid_detection_count; ++i) draw_obb(bbox, &detections[i]);
        if (!events_publish(detections, valid_detection_count, labels,
                            MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT, capture_time_us))
            syslog(LOG_WARNING, "Detection result snapshot not updated; will retry next frame");

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
    exit_status = EXIT_SUCCESS;

end:
    // Cleanup
    events_stop();
    free(model_params);
    if (image_provider) {
        destroy_img_provider(image_provider);
    }
    if (model_provider) {
        destroy_model_provider(model_provider);
    }
    free(tensor_outputs);
    free(labels);
    free(label_file_data);
    if (bbox) bbox_destroy(bbox);

    syslog(LOG_INFO, "Exit %s", argv[0]);

    return exit_status;
}
