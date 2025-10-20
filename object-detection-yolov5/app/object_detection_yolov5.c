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
 * This application loads a larod YOLOv5 model which takes an image as input. The output is
 * YOLOv5-specifically parsed to retrieve values corresponding to the class, score and location of
 * detected objects in the image.
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
#include <sys/time.h>
#include <syslog.h>

#include <axsdk/axevent.h>
#include <errno.h>
#include <glib-object.h>
#include <glib.h>
#include <string.h>  // for strerror if not already included
#include <unistd.h>

#define APP_NAME "object_detection_yolov5"

// Event system structure (following send_event.c pattern)
typedef struct {
    AXEventHandler* event_handler;
    guint event_id;
    gboolean declaration_complete;
    guint detection_timer;
} EventSystem;

typedef struct model_params {
    int input_width;
    int input_height;
    float quantization_scale;
    float quantization_zero_point;
    int num_classes;
    int num_detections;
    int size_per_detection;
} model_params_t;

typedef struct {
    img_provider_t* image_provider;
    model_provider_t* model_provider;
    model_tensor_output_t* tensor_outputs;
    size_t number_output_tensors;
    bbox_t* bbox;
    char** labels;
    model_params_t* model_params;
    int* invalid_detections;
    float conf_threshold;
    float iou_threshold;
} detection_data_t;

static void shutdown(int status);
static char* compose_detection_result(const char* object_class,
                                      float confidence,
                                      float x1,
                                      float y1,
                                      float x2,
                                      float y2);
static void send_object_detection_event(const char* object_class,
                                        float confidence,
                                        float x1,
                                        float y1,
                                        float x2,
                                        float y2);
static void declaration_complete(guint declaration, gpointer user_data);
static guint setup_object_detection_declaration(AXEventHandler* event_handler, gpointer user_data);
static void initialize_event_system(gpointer user_data);
static void cleanup_event_system(void);

volatile sig_atomic_t running                  = 1;
static EventSystem* event_system               = NULL;
static GMainLoop* main_loop                    = NULL;
static detection_data_t* global_detection_data = NULL;

static gboolean detection_timer_callback(gpointer user_data);

static void shutdown(int status) {
    syslog(LOG_INFO, "Received signal %d, shutting down", status);
    running = 0;

    // Stop detection timer
    if (event_system && event_system->detection_timer > 0) {
        g_source_remove(event_system->detection_timer);
        event_system->detection_timer = 0;
    }

    // Quit main loop
    if (main_loop) {
        g_main_loop_quit(main_loop);
    }
}

static char* compose_detection_result(const char* object_class,
                                      float confidence,
                                      float x1,
                                      float y1,
                                      float x2,
                                      float y2) {
    char* json_string = malloc(512);
    if (!json_string)
        return NULL;

    snprintf(json_string,
             512,
             "{\"class\":\"%s\",\"confidence\":%.3f,\"bbox\":[%.3f,%.3f,%.3f,%.3f]}",
             object_class,
             confidence,
             x1,
             y1,
             x2,
             y2);

    return json_string;
}

/**
 * Send object detection event immediately when object is detected.
 * Following the exact pattern from send_event.c
 */
static void send_object_detection_event(const char* object_class,
                                        float confidence,
                                        float x1,
                                        float y1,
                                        float x2,
                                        float y2) {
    AXEventKeyValueSet* key_value_set = NULL;
    AXEvent* event                    = NULL;
    GError* error                     = NULL;

    // Only send if declaration is complete
    if (!event_system || !event_system->declaration_complete || !event_system->event_handler) {
        syslog(LOG_WARNING,
               "Event system not ready: system=%p, complete=%d, handler=%p",
               event_system,
               event_system ? event_system->declaration_complete : -1,
               event_system ? event_system->event_handler : NULL);
        return;
    }

    syslog(LOG_INFO, "📤 Sending event for %s", object_class);

    // Create key-value set (like send_event.c)
    key_value_set = ax_event_key_value_set_new();

    const char* detection_result =
        compose_detection_result(object_class, confidence, x1, y1, x2, y2);
    if (!detection_result) {
        syslog(LOG_ERR, "❌ Failed to compose detection result JSON");
        ax_event_key_value_set_free(key_value_set);
        return;
    }

    // Add detection result
    syslog(LOG_INFO, "Add Result: %s", detection_result);
    if (!ax_event_key_value_set_add_key_value(key_value_set,
                                              "Result",
                                              NULL,
                                              &detection_result,
                                              AX_VALUE_TYPE_STRING,
                                              NULL)) {
        syslog(LOG_ERR, "❌ Failed to add Result to event");
        free((void*)detection_result);
        return;
    }

    // Create the event (like send_event.c)
    // Use ax_event_new2 since ax_event_new is deprecated from 3.2
    event = ax_event_new2(key_value_set, NULL);
    if (!event) {
        syslog(LOG_ERR, "❌ Failed to create event");
        free((void*)detection_result);
        ax_event_key_value_set_free(key_value_set);
        return;
    }

    // The key/value set is no longer needed (following send_event.c pattern)
    ax_event_key_value_set_free(key_value_set);
    free((void*)detection_result);

    // Send the event (like send_event.c)
    if (!ax_event_handler_send_event(event_system->event_handler,
                                     event_system->event_id,
                                     event,
                                     &error)) {
        syslog(LOG_ERR, "❌ Failed to send event: %s", error ? error->message : "Unknown error");
        if (error)
            g_error_free(error);
    } else {
        syslog(LOG_INFO, "✅ Successfully sent event: %s", object_class);
        syslog(LOG_INFO, "event id is %d", event_system->event_id);
    }

    // Free the event (like send_event.c)
    ax_event_free(event);
}

/**
 * Callback function which is called when event declaration is completed.
 * Following the exact pattern from send_event.c
 */
static void declaration_complete(guint declaration, gpointer user_data) {
    syslog(LOG_INFO, "Declaration complete for: %d", declaration);

    event_system->declaration_complete = TRUE;
    syslog(LOG_INFO,
           "Event declaration marked as complete, status of declaration_complete=%d",
           event_system->declaration_complete);

    detection_data_t* detection_data = (detection_data_t*)user_data;

    // Start detection timer - run every 500ms (about 2 FPS)
    event_system->detection_timer = g_timeout_add(500, detection_timer_callback, detection_data);
    syslog(LOG_INFO, "Detection timer started (timer ID: %u)", event_system->detection_timer);
}

/**
 * Setup a declaration of an object detection event.
 * Following the exact pattern from send_event.c setup_declaration
 */
static guint setup_object_detection_declaration(AXEventHandler* event_handler, gpointer user_data) {
    AXEventKeyValueSet* key_value_set = NULL;
    guint declaration                 = 0;
    guint token                       = 0;
    GError* error                     = NULL;

    // Create keys, namespaces and nice names for the event (like send_event.c)
    key_value_set = ax_event_key_value_set_new();

    // Set up topic hierarchy (following send_event.c pattern)
    ax_event_key_value_set_add_key_value(key_value_set,
                                         "topic0",
                                         "tns1",
                                         "VideoAnalytics",
                                         AX_VALUE_TYPE_STRING,
                                         NULL);
    ax_event_key_value_set_add_key_value(key_value_set,
                                         "topic1",
                                         "tns1",
                                         "ObjectDetected",
                                         AX_VALUE_TYPE_STRING,
                                         NULL);

    // Add Token as source (like send_event.c)
    ax_event_key_value_set_add_key_value(key_value_set,
                                         "Token",
                                         NULL,
                                         &token,
                                         AX_VALUE_TYPE_INT,
                                         NULL);

    // Add data elements with empty initial values
    const char* empty_string = "";
    ax_event_key_value_set_add_key_value(key_value_set,
                                         "Result",
                                         NULL,
                                         &empty_string,
                                         AX_VALUE_TYPE_STRING,
                                         NULL);

    // Mark Token as source (like send_event.c)
    ax_event_key_value_set_mark_as_source(key_value_set, "Token", NULL, NULL);
    ax_event_key_value_set_mark_as_user_defined(key_value_set,
                                                "Token",
                                                NULL,
                                                "wstype:tt:ReferenceToken",
                                                NULL);

    // Mark data elements (like send_event.c marks "Value")
    ax_event_key_value_set_mark_as_data(key_value_set, "Result", NULL, NULL);
    ax_event_key_value_set_mark_as_user_defined(key_value_set,
                                                "Result",
                                                NULL,
                                                "wstype:xs:string",
                                                NULL);

    // Declare event (like send_event.c)
    syslog(LOG_INFO, "Declaring object detection event");
    if (!ax_event_handler_declare(
            event_handler,
            key_value_set,
            FALSE,  // TRUE for stateless events (unlike send_event.c which uses FALSE for stateful)
            &declaration,
            (AXDeclarationCompleteCallback)declaration_complete,
            user_data,
            &error)) {
        syslog(LOG_WARNING, "Could not declare object detection event: %s", error->message);
        g_error_free(error);
    }

    syslog(LOG_INFO, "Declared object detection event inside setup_object_detection_declaration");

    // The key/value set is no longer needed (like send_event.c)
    ax_event_key_value_set_free(key_value_set);
    return declaration;
}

/**
 * Initialize event system (following send_event.c main function pattern)
 */
static void initialize_event_system(gpointer user_data) {
    syslog(LOG_INFO, "Initializing object detection event system");

    // Allocate event system (like send_event.c allocates app_data)
    event_system                       = calloc(1, sizeof(EventSystem));
    event_system->event_handler        = ax_event_handler_new();
    event_system->declaration_complete = FALSE;

    // Setup declaration (like send_event.c)
    event_system->event_id =
        setup_object_detection_declaration(event_system->event_handler, user_data);
    syslog(LOG_INFO,
           "Initialized object detection event system: event_id=%u",
           event_system->event_id);
}

/**
 * Cleanup event system (following send_event.c cleanup pattern)
 */
static void cleanup_event_system(void) {
    if (event_system) {
        syslog(LOG_INFO, "Cleaning up event system");

        // Undeclare event (like send_event.c)
        ax_event_handler_undeclare(event_system->event_handler, event_system->event_id, NULL);
        ax_event_handler_free(event_system->event_handler);
        free(event_system);
        event_system = NULL;
    }
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

static float intersection_over_union(float x1,
                                     float y1,
                                     float w1,
                                     float h1,
                                     float x2,
                                     float y2,
                                     float w2,
                                     float h2) {
    float xx1 = fmax(x1 - (w1 / 2), x2 - (w2 / 2));
    float yy1 = fmax(y1 - (h1 / 2), y2 - (h2 / 2));
    float xx2 = fmin(x1 + (w1 / 2), x2 + (w2 / 2));
    float yy2 = fmin(y1 + (h1 / 2), y2 + (h2 / 2));

    float inter_area = fmax(0, xx2 - xx1) * fmax(0, yy2 - yy1);
    float union_area = w1 * h1 + w2 * h2 - inter_area;

    return inter_area / union_area;
}

static void non_maximum_suppression(uint8_t* tensor,
                                    float iou_threshold,
                                    model_params_t* model_params,
                                    int* invalid_detections) {
    int size_per_detection = model_params->size_per_detection;
    int num_detections     = model_params->num_detections;
    float qt_zero_point    = model_params->quantization_zero_point;
    float qt_scale         = model_params->quantization_scale;

    for (int i = 0; i < num_detections; i++) {
        if (invalid_detections[i])  // Skip comparison if detection is already invalid
            continue;

        float x1                 = (tensor[size_per_detection * i + 0] - qt_zero_point) * qt_scale;
        float y1                 = (tensor[size_per_detection * i + 1] - qt_zero_point) * qt_scale;
        float w1                 = (tensor[size_per_detection * i + 2] - qt_zero_point) * qt_scale;
        float h1                 = (tensor[size_per_detection * i + 3] - qt_zero_point) * qt_scale;
        float object1_likelihood = (tensor[size_per_detection * i + 4] - qt_zero_point) * qt_scale;

        for (int j = i + 1; j < num_detections; j++) {
            if (invalid_detections[j])  // Skip comparison if detection is already invalid
                continue;

            float x2 = (tensor[size_per_detection * j + 0] - qt_zero_point) * qt_scale;
            float y2 = (tensor[size_per_detection * j + 1] - qt_zero_point) * qt_scale;
            float w2 = (tensor[size_per_detection * j + 2] - qt_zero_point) * qt_scale;
            float h2 = (tensor[size_per_detection * j + 3] - qt_zero_point) * qt_scale;
            float object2_likelihood =
                (tensor[size_per_detection * j + 4] - qt_zero_point) * qt_scale;

            if (intersection_over_union(x1, y1, w1, h1, x2, y2, w2, h2) > iou_threshold) {
                // invalidates the detection with lowest object likelihood score
                if (object1_likelihood > object2_likelihood) {
                    invalid_detections[j] = 1;
                } else {
                    invalid_detections[i] = 1;
                    break;
                }
            }
        }
    }
}

static void filter_detections(uint8_t* tensor,
                              float conf_theshold,
                              float iou_threshold,
                              model_params_t* model_params,
                              int* invalid_detections) {
    // Filter boxes by confidence
    for (int i = 0; i < model_params->num_detections; i++) {
        float object_likelihood = (tensor[model_params->size_per_detection * i + 4] -
                                   model_params->quantization_zero_point) *
                                  model_params->quantization_scale;

        if (object_likelihood < conf_theshold) {
            invalid_detections[i] = 1;
        } else {
            invalid_detections[i] = 0;
        }
    }

    non_maximum_suppression(tensor, iou_threshold, model_params, invalid_detections);
}

static void determine_class_and_object_likelihood(uint8_t* tensor,
                                                  int detection_idx,
                                                  int size_per_detection,
                                                  float qt_zero_point,
                                                  float qt_scale,
                                                  float* highest_class_likelihood,
                                                  int* label_idx,
                                                  float* object_likelihood) {
    // Find what class this object is
    for (int j = 5; j < size_per_detection; j++) {
        float class_likelihood =
            (tensor[size_per_detection * detection_idx + j] - qt_zero_point) * qt_scale;
        if (class_likelihood > *highest_class_likelihood) {
            *highest_class_likelihood = class_likelihood;
            *label_idx                = j - 5;
        }
    }

    *object_likelihood =
        (tensor[size_per_detection * detection_idx + 4] - qt_zero_point) * qt_scale;
}

static void
find_corners(float x, float y, float w, float h, float* x1, float* y1, float* x2, float* y2) {
    *x1 = fmax(0.0, x - (w / 2));
    *y1 = fmax(0.0, y - (h / 2));
    *x2 = fmin(1.0, x + (w / 2));
    *y2 = fmin(1.0, y + (h / 2));
}

static void determine_bbox_coordinates(uint8_t* tensor,
                                       int detection_idx,
                                       int size_per_detection,
                                       float qt_zero_point,
                                       float qt_scale,
                                       float* x1,
                                       float* y1,
                                       float* x2,
                                       float* y2) {
    // Get coordinates for the object
    float x = (tensor[size_per_detection * detection_idx + 0] - qt_zero_point) * qt_scale;
    float y = (tensor[size_per_detection * detection_idx + 1] - qt_zero_point) * qt_scale;
    float w = (tensor[size_per_detection * detection_idx + 2] - qt_zero_point) * qt_scale;
    float h = (tensor[size_per_detection * detection_idx + 3] - qt_zero_point) * qt_scale;
    find_corners(x, y, w, h, x1, y1, x2, y2);
}

static gboolean detection_timer_callback(gpointer user_data) {
    // Move all your detection code here (from the while loop)
    // This will be called periodically by GLib

    if (!running) {
        syslog(LOG_INFO, "Stopping detection timer");
        return FALSE;  // Stop the timer
    }

    // Your existing detection code goes here (from line ~480 onwards)
    struct timeval start_ts, end_ts;
    unsigned int preprocessing_ms = 0;
    unsigned int inference_ms     = 0;
    unsigned int total_elapsed_ms = 0;

    // Get the necessary variables from user_data

    detection_data_t* data = (detection_data_t*)user_data;
    if (!data || !data->image_provider) {
        syslog(LOG_ERR, "Invalid detection data in timer callback");
        return FALSE;
    }

    g_autoptr(GError) vdo_error  = NULL;
    g_autoptr(VdoBuffer) vdo_buf = img_provider_get_frame(data->image_provider);

    if (!vdo_buf) {
        syslog(LOG_INFO,
               "No buffer because of changed global rotation. Application needs to be restarted");
        running = 0;
        return FALSE;
    }

    // All your existing detection processing code...
    gettimeofday(&start_ts, NULL);
    if (!model_run_preprocessing(data->model_provider, vdo_buf)) {
        if (!vdo_stream_buffer_unref(data->image_provider->vdo_stream, &vdo_buf, &vdo_error)) {
            if (!vdo_error_is_expected(&vdo_error)) {
                syslog(LOG_ERR, "Unexpected error: %s", vdo_error->message);
            }
            g_clear_error(&vdo_error);
        }
        img_provider_flush_all_frames(data->image_provider);
        return TRUE;  // Continue running
    }
    gettimeofday(&end_ts, NULL);

    preprocessing_ms = elapsed_ms(&start_ts, &end_ts);
    syslog(LOG_INFO, "Ran pre-processing for %u ms", preprocessing_ms);

    gettimeofday(&start_ts, NULL);
    if (!model_run_inference(data->model_provider, vdo_buf)) {
        if (!vdo_stream_buffer_unref(data->image_provider->vdo_stream, &vdo_buf, &vdo_error)) {
            if (!vdo_error_is_expected(&vdo_error)) {
                syslog(LOG_ERR, "Unexpected error: %s", vdo_error->message);
            }
            g_clear_error(&vdo_error);
        }
        img_provider_flush_all_frames(data->image_provider);
        return TRUE;
    }
    gettimeofday(&end_ts, NULL);

    inference_ms = elapsed_ms(&start_ts, &end_ts);
    syslog(LOG_INFO, "Ran inference for %u ms", inference_ms);

    total_elapsed_ms = inference_ms + preprocessing_ms;
    img_provider_update_framerate(data->image_provider, total_elapsed_ms);

    // Get tensor outputs
    for (size_t i = 0; i < data->number_output_tensors; i++) {
        if (!model_get_tensor_output_info(data->model_provider, i, &data->tensor_outputs[i])) {
            syslog(LOG_ERR, "Failed to get output tensor info for %zu", i);
            return TRUE;
        }
    }

    uint8_t* tensor_data = data->tensor_outputs[0].data;

    gettimeofday(&start_ts, NULL);
    filter_detections(tensor_data,
                      data->conf_threshold,
                      data->iou_threshold,
                      data->model_params,
                      data->invalid_detections);
    gettimeofday(&end_ts, NULL);
    syslog(LOG_INFO, "Ran parsing for %u ms", elapsed_ms(&start_ts, &end_ts));

    bbox_clear(data->bbox);

    int valid_detection_count = 0;
    int size_per_detection    = data->model_params->size_per_detection;
    float qt_zero_point       = data->model_params->quantization_zero_point;
    float qt_scale            = data->model_params->quantization_scale;

    // Process detections
    for (int i = 0; i < data->model_params->num_detections; i++) {
        if (data->invalid_detections[i] == 1) {
            continue;
        }

        valid_detection_count++;

        float highest_class_likelihood = 0.0;
        int label_idx                  = 0;
        float object_likelihood        = 0.0;

        determine_class_and_object_likelihood(tensor_data,
                                              i,
                                              size_per_detection,
                                              qt_zero_point,
                                              qt_scale,
                                              &highest_class_likelihood,
                                              &label_idx,
                                              &object_likelihood);

        syslog(LOG_INFO,
               "Object %d: Label=%s, Object Likelihood=%.2f, Class Likelihood=%.2f",
               valid_detection_count,
               data->labels[label_idx],
               object_likelihood,
               highest_class_likelihood);

        float x1, y1, x2, y2;
        determine_bbox_coordinates(tensor_data,
                                   i,
                                   size_per_detection,
                                   qt_zero_point,
                                   qt_scale,
                                   &x1,
                                   &y1,
                                   &x2,
                                   &y2);
        syslog(LOG_INFO, "Bounding Box: [%.2f, %.2f, %.2f, %.2f]", x1, y1, x2, y2);

        bbox_coordinates_frame_normalized(data->bbox);
        bbox_rectangle(data->bbox, x1, y1, x2, y2);

        syslog(LOG_INFO,
               "Detected %s with confidence %.2f at (%.2f, %.2f, %.2f, %.2f)",
               data->labels[label_idx],
               highest_class_likelihood,
               x1,
               y1,
               x2,
               y2);

        // Send object detection event
        send_object_detection_event(data->labels[label_idx],
                                    highest_class_likelihood,
                                    x1,
                                    y1,
                                    x2,
                                    y2);
    }

    if (!bbox_commit(data->bbox, 0u)) {
        syslog(LOG_ERR, "Failed to commit box drawer");
    }

    // Unref the buffer
    if (!vdo_stream_buffer_unref(data->image_provider->vdo_stream, &vdo_buf, &vdo_error)) {
        if (!vdo_error_is_expected(&vdo_error)) {
            syslog(LOG_ERR, "Unexpected error: %s", vdo_error->message);
        }
        g_clear_error(&vdo_error);
    }

    return TRUE;  // Continue running the timer
}

int main(int argc, char** argv) {
    // Signal handlers
    signal(SIGTERM, shutdown);
    signal(SIGINT, shutdown);

    args_t args;
    parse_args(argc, argv, &args);

    model_params_t* model_params          = NULL;
    int* invalid_detections               = NULL;
    img_provider_t* image_provider        = NULL;
    model_provider_t* model_provider      = NULL;
    model_tensor_output_t* tensor_outputs = NULL;
    char** labels                         = NULL;
    char* label_file_data                 = NULL;
    bbox_t* bbox                          = NULL;

    // All your initialization code...
    model_params = (model_params_t*)malloc(sizeof(model_params_t));
    if (model_params == NULL) {
        panic("%s: Unable to allocate model_params_t: %s", __func__, strerror(errno));
    }

    // Set up model parameters
    model_params->input_width             = MODEL_INPUT_WIDTH;
    model_params->input_height            = MODEL_INPUT_HEIGHT;
    model_params->quantization_scale      = QUANTIZATION_SCALE;
    model_params->quantization_zero_point = QUANTIZATION_ZERO_POINT;
    model_params->num_classes             = NUM_CLASSES;
    model_params->num_detections          = NUM_DETECTIONS;
    model_params->size_per_detection      = 5 + NUM_CLASSES;

    syslog(LOG_INFO,
           "Model input size w/h: %d x %d",
           model_params->input_width,
           model_params->input_height);
    syslog(LOG_INFO, "Quantization scale: %f", model_params->quantization_scale);
    syslog(LOG_INFO, "Quantization zero point: %f", model_params->quantization_zero_point);
    syslog(LOG_INFO, "Number of classes: %d", model_params->num_classes);
    syslog(LOG_INFO, "Number of detections: %d", model_params->num_detections);

    int* invalid_detections = calloc(model_params->num_detections, sizeof(int));

    // Get parameters
    GError* axparameter_error       = NULL;
    AXParameter* axparameter_handle = ax_parameter_new(APP_NAME, &axparameter_error);
    if (axparameter_handle == NULL) {
        panic("%s", axparameter_error->message);
    }

    float conf_threshold = ax_parameter_get_int(axparameter_handle, "ConfThresholdPercent") / 100.0;
    float iou_threshold  = ax_parameter_get_int(axparameter_handle, "IouThresholdPercent") / 100.0;
    ax_parameter_free(axparameter_handle);

    // Set up image and model providers (your existing code)
    VdoFormat vdo_format = VDO_FORMAT_YUV;
    double vdo_framerate = 30.0;

    if (!g_strcmp0(args.device_name, "a9-dlpu-tflite")) {
        vdo_format = VDO_FORMAT_RGB;
    }

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
        goto cleanup;
    }

    syslog(LOG_INFO,
           "Creating VDO image provider and creating stream %u x %u",
           stream_width,
           stream_height);

    img_provider_t* image_provider =
        create_img_provider(stream_width, stream_height, 2, vdo_format, vdo_framerate);
    if (!image_provider) {
        panic("%s: Could not create image provider", __func__);
    }

    size_t number_output_tensors     = 0;
    model_provider_t* model_provider = create_model_provider(model_params->input_width,
                                                             model_params->input_height,
                                                             image_provider->width,
                                                             image_provider->height,
                                                             image_provider->pitch,
                                                             image_provider->format,
                                                             VDO_FORMAT_RGB,
                                                             args.model_file,
                                                             args.device_name,
                                                             false,
                                                             &number_output_tensors);
    if (!model_provider) {
        panic("%s: Could not create model provider", __func__);
    }

    model_tensor_output_t* tensor_outputs =
        calloc(number_output_tensors, sizeof(model_tensor_output_t));
    if (!tensor_outputs) {
        panic("%s: Could not allocate tensor outputs", __func__);
    }

    char** labels = NULL;
    size_t num_labels;
    char* label_file_data = NULL;
    parse_labels(&labels, &label_file_data, args.labels_file, &num_labels);

    syslog(LOG_INFO, "Start fetching video frames from VDO");
    if (!img_provider_start(image_provider)) {
        panic("%s: Could not start image provider", __func__);
    }

    bbox_t* bbox = setup_bbox();

    detection_data_t detection_data = {.image_provider        = image_provider,
                                       .model_provider        = model_provider,
                                       .tensor_outputs        = tensor_outputs,
                                       .number_output_tensors = number_output_tensors,
                                       .bbox                  = bbox,
                                       .labels                = labels,
                                       .model_params          = model_params,
                                       .invalid_detections    = invalid_detections,
                                       .conf_threshold        = conf_threshold,
                                       .iou_threshold         = iou_threshold};

    // Set global reference for shutdown function
    global_detection_data = &detection_data;

    // Initialize event system AFTER creating detection_data
    initialize_event_system(&detection_data);

    // Send test events immediately (before starting main loop)
    syslog(LOG_INFO, "Sending test object detection events");
    send_object_detection_event("TestObject1", 0.99, 0.1, 0.2, 0.3, 0.4);
    send_object_detection_event("TestObject2", 0.88, 0.1, 0.2, 0.3, 0.4);
    send_object_detection_event("TestObject3", 0.77, 0.1, 0.2, 0.3, 0.4);

    // Create and run GLib main loop
    main_loop = g_main_loop_new(NULL, FALSE);
    syslog(LOG_INFO, "Starting GLib main loop - detection will begin after declaration completes");

    // This blocks until g_main_loop_quit() is called
    g_main_loop_run(main_loop);

cleanup:
    syslog(LOG_INFO, "Cleaning up resources...");

    cleanup_event_system();

    // Cleanup all resources
    if (model_params) {
        free(model_params);
    }
    if (invalid_detections) {
        free(invalid_detections);
    }
    if (image_provider) {
        destroy_img_provider(image_provider);
    }
    if (model_provider) {
        destroy_model_provider(model_provider);
    }
    if (tensor_outputs) {
        free(tensor_outputs);
    }
    if (labels) {
        free(labels);
    }
    if (label_file_data) {
        free(label_file_data);
    }
    if (bbox) {
        bbox_destroy(bbox);
    }

    if (main_loop) {
        g_main_loop_unref(main_loop);
    }

    syslog(LOG_INFO, "Exit %s", argv[0]);
    return 0;
}