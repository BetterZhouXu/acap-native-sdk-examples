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
 * - object_detection_bbox_yolov5 -
 *
 * This application loads a Larod YOLOv8 OBB model. Its float32 channel-first output is decoded into
 * classes, confidence scores, angles, and oriented bounding-box corners.
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
#include "detection_fastcgi.h"
#include "detection_result.h"
#include "imgprovider.h"
#include "labelparse.h"
#include "model.h"
#include "model_params.h"  //Generated at build time
#include "panic.h"
#include "yolov8_obb.h"
#include "vdo-error.h"
#include "vdo-frame.h"
#include "vdo-types.h"
#include <axsdk/axparameter.h>
#include <bbox.h>
#include <jansson.h>

#include <errno.h>
#include <math.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <syslog.h>

#define APP_NAME "detection"

volatile sig_atomic_t running = 1;

static void shutdown(int status) {
    (void)status;
    running = 0;
}

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


int main(int argc, char** argv) {
    g_autoptr(GError) vdo_error           = NULL;
    img_provider_t* image_provider        = NULL;
    model_provider_t* model_provider      = NULL;
    model_tensor_output_t* tensor_outputs = NULL;
    bbox_t* bbox                          = NULL;
    char** labels                         = NULL;
    char* label_file_data                 = NULL;
    size_t num_labels                     = 0;

    // Stop main loop at signal
    signal(SIGTERM, shutdown);
    signal(SIGINT, shutdown);
    signal(SIGPIPE, SIG_IGN);

    if (!detection_result_init()) {
        panic("Failed to initialize detection result storage: %s", strerror(errno));
    }
    if (!detection_fastcgi_start()) {
        panic("Failed to start detection result endpoint");
    }

    args_t args;
    parse_args(argc, argv, &args);

    yolov8_obb_params_t* model_params = malloc(sizeof(yolov8_obb_params_t));
    if (model_params == NULL) {
        panic("%s: Unable to allocate model parameters: %s", __func__, strerror(errno));
    }

    // Comes from model_params.h
    model_params->input_width    = MODEL_INPUT_WIDTH;
    model_params->input_height   = MODEL_INPUT_HEIGHT;
    model_params->num_classes    = NUM_CLASSES;
    model_params->num_detections = NUM_DETECTIONS;
    model_params->num_features   = NUM_FEATURES;

    syslog(LOG_INFO,
           "Model input size w/h: %d x %d",
           model_params->input_width,
           model_params->input_height);
    syslog(LOG_INFO, "Model input/output type: float32");
    syslog(LOG_INFO, "Number of classes: %d", model_params->num_classes);
    syslog(LOG_INFO, "Number of detections: %d", model_params->num_detections);
    syslog(LOG_INFO, "Features per detection: %d", model_params->num_features);

    obb_detection_t* parsed_detections =
        calloc(YOLOV8_OBB_MAX_RESULTS, sizeof(obb_detection_t));
    if (!parsed_detections) {
        panic("%s: Could not allocate parsed detections", __func__);
    }

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
                                           &number_output_tensors);
    if (!model_provider) {
        panic("%s: Could not create model provider", __func__);
    }
    if (number_output_tensors != 1) {
        panic("YOLOv8 OBB model must have exactly one output tensor, got %zu",
              number_output_tensors);
    }
    tensor_outputs = calloc(number_output_tensors, sizeof(model_tensor_output_t));
    if (!tensor_outputs) {
        panic("%s: Could not allocate tensor outputs", __func__);
    }

    parse_labels(&labels, &label_file_data, args.labels_file, &num_labels);
    if (num_labels != (size_t)model_params->num_classes) {
        panic("Model has %d classes but labels file has %zu labels",
              model_params->num_classes,
              num_labels);
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

        if (!model_get_tensor_output_info(model_provider, 0, &tensor_outputs[0])) {
            panic("Failed to get output tensor info");
        }

        size_t expected_output_size = (size_t)model_params->num_features *
                                      (size_t)model_params->num_detections * sizeof(float);
        if (tensor_outputs[0].datatype != LAROD_TENSOR_DATA_TYPE_FLOAT32 ||
            tensor_outputs[0].size != expected_output_size) {
            panic("Unexpected output tensor type or size (%zu, expected %zu)",
                  tensor_outputs[0].size,
                  expected_output_size);
        }

        size_t valid_detection_count = 0;
        gettimeofday(&start_ts, NULL);
        if (!yolov8_obb_decode(tensor_outputs[0].data,
                               model_params,
                               conf_threshold,
                               iou_threshold,
                               parsed_detections,
                               YOLOV8_OBB_MAX_RESULTS,
                               &valid_detection_count)) {
            panic("Failed to decode YOLOv8 OBB output");
        }
        gettimeofday(&end_ts, NULL);
        syslog(LOG_INFO, "Ran parsing for %u ms", elapsed_ms(&start_ts, &end_ts));

        bbox_clear(bbox);
        json_t* detections_json = json_array();
        if (!detections_json) {
            panic("Failed to create detection JSON array");
        }

        for (size_t i = 0; i < valid_detection_count; i++) {
            const obb_detection_t* detection = &parsed_detections[i];
            syslog(LOG_INFO,
                   "Object %zu: Label=%s, Confidence=%.2f, Angle=%.3f radians",
                   i + 1,
                   labels[detection->label_index],
                   detection->confidence,
                   detection->angle);

            // No need to compensate for rotation since bbox will handle this
            bbox_coordinates_frame_normalized(bbox);
            bbox_quad(bbox,
                      fminf(1.0F, fmaxf(0.0F, detection->corners[0].x)),
                      fminf(1.0F, fmaxf(0.0F, detection->corners[0].y)),
                      fminf(1.0F, fmaxf(0.0F, detection->corners[1].x)),
                      fminf(1.0F, fmaxf(0.0F, detection->corners[1].y)),
                      fminf(1.0F, fmaxf(0.0F, detection->corners[2].x)),
                      fminf(1.0F, fmaxf(0.0F, detection->corners[2].y)),
                      fminf(1.0F, fmaxf(0.0F, detection->corners[3].x)),
                      fminf(1.0F, fmaxf(0.0F, detection->corners[3].y)));

            json_t* corners = json_pack("[[f,f],[f,f],[f,f],[f,f]]",
                                        detection->corners[0].x,
                                        detection->corners[0].y,
                                        detection->corners[1].x,
                                        detection->corners[1].y,
                                        detection->corners[2].x,
                                        detection->corners[2].y,
                                        detection->corners[3].x,
                                        detection->corners[3].y);
            if (!corners) {
                json_decref(detections_json);
                panic("Failed to create detection corners JSON");
            }
            json_t* oriented_box = json_pack("{s:f,s:f,s:f,s:f,s:f,s:O}",
                                             "centerX",
                                             detection->center_x,
                                             "centerY",
                                             detection->center_y,
                                             "width",
                                             detection->width,
                                             "height",
                                             detection->height,
                                             "angleRadians",
                                             detection->angle,
                                             "corners",
                                             corners);
            json_decref(corners);
            if (!oriented_box) {
                json_decref(detections_json);
                panic("Failed to create oriented box JSON");
            }
            json_t* item = json_pack("{s:s,s:f,s:O}",
                                     "label",
                                     labels[detection->label_index],
                                     "confidence",
                                     detection->confidence,
                                     "orientedBox",
                                     oriented_box);
            json_decref(oriented_box);
            if (!item || json_array_append(detections_json, item) != 0) {
                json_decref(item);
                json_decref(detections_json);
                panic("Failed to create detection JSON");
            }
            json_decref(item);
        }

        if (valid_detection_count > 0) {
            struct timeval result_time;
            gettimeofday(&result_time, NULL);
            json_int_t timestamp_ms = (json_int_t)result_time.tv_sec * 1000 +
                                      (json_int_t)result_time.tv_usec / 1000;
            json_t* result = json_pack("{s:I,s:O}",
                                       "timestampUnixMs",
                                       timestamp_ms,
                                       "detections",
                                       detections_json);
            json_decref(detections_json);
            char* result_json = result ? json_dumps(result, JSON_COMPACT) : NULL;
            if (!result || !result_json) {
                json_decref(result);
                panic("Failed to serialize detection JSON");
            }
            if (!detection_result_publish(result_json, strlen(result_json))) {
                syslog(LOG_ERR, "Failed to save detection result: %s", strerror(errno));
            }
            free(result_json);
            json_decref(result);
        } else {
            json_decref(detections_json);
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
    free(parsed_detections);
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
