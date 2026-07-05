/**
 * Copyright (C) 2021 Axis Communications AB, Lund, Sweden
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
 * - object_detection -
 *
 * This application loads a larod model which takes an image as input and
 * outputs values corresponding to the class, score and location of detected
 * objects in the image.
 *
 * The application expects at least one argument on the command line in the
 * following order: MODEL.
 *
 * If THRESHOLD and LABELSFILE is supplied postprocessing will be used.
 *
 * First argument, MODEL, is a string describing path to the model.
 *
 * Second argument, THRESHOLD is an integer ranging from 0 to 100 to select good detections.
 *
 * Third argument, LABELSFILE, is a string describing path to the label txt.
 *
 * FOURTH argument, DEVICE, is a string for which larod device to use.
 *
 */

#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <syslog.h>
#include <time.h>
#include <unistd.h>

#include <curl/curl.h>

#include "argparse.h"
#include "imgprovider.h"
#include "labelparse.h"
#include "model.h"
#include "panic.h"
#include "vdo-error.h"
#include "vdo-frame.h"
#include "vdo-types.h"
#include <bbox.h>

// Environment variable holding the HTTPS endpoint that detections are POSTed to.
// Example: export DETECTION_POST_URL="https://example.com/detections"
#define DETECTION_POST_URL_ENV "kepmpe01.s08299.us.wal-mart.com/detections"

volatile sig_atomic_t running = 1;

// define box struct
typedef struct {
    float y_min;
    float x_min;
    float y_max;
    float x_max;
    float score;
    int label;
} box;

static void shutdown(int status) {
    (void)status;
    running = 0;
}

/**
 * @brief Silently discard any response body from the HTTPS server.
 */
static size_t discard_response_cb(void* ptr, size_t size, size_t nmemb, void* userdata) {
    (void)ptr;
    (void)userdata;
    return size * nmemb;
}

/**
 * @brief POST a single detection result as JSON to the configured HTTPS URL.
 *
 * The URL is read from the DETECTION_POST_URL environment variable. If it is
 * not set, this function is a no-op. Any transport errors are logged but do
 * not stop the detection pipeline.
 *
 * @param url    Target HTTPS endpoint (must be non-NULL).
 * @param index  Index of the detection in the current frame.
 * @param label  Human-readable class label.
 * @param score  Detection confidence in [0.0, 1.0].
 * @param top    Normalized top    coordinate of the bounding box.
 * @param left   Normalized left   coordinate of the bounding box.
 * @param bottom Normalized bottom coordinate of the bounding box.
 * @param right  Normalized right  coordinate of the bounding box.
 * @return true on success, false on failure.
 */
static bool send_detection_https(const char* url,
                                 int index,
                                 const char* label,
                                 float score,
                                 float top,
                                 float left,
                                 float bottom,
                                 float right) {
    if (!url || !*url) {
        return false;
    }

    CURL* curl = curl_easy_init();
    if (!curl) {
        syslog(LOG_ERR, "curl_easy_init failed");
        return false;
    }

    // Build a compact JSON body describing the detection.
    char body[512];
    int n = snprintf(body,
                     sizeof(body),
                     "{\"index\":%d,\"label\":\"%s\",\"score\":%.4f,"
                     "\"bbox\":{\"top\":%.6f,\"left\":%.6f,"
                     "\"bottom\":%.6f,\"right\":%.6f},"
                     "\"timestamp\":%ld}",
                     index,
                     label ? label : "",
                     (double)score,
                     (double)top,
                     (double)left,
                     (double)bottom,
                     (double)right,
                     (long)time(NULL));
    if (n < 0 || (size_t)n >= sizeof(body)) {
        syslog(LOG_ERR, "Detection JSON truncated");
        curl_easy_cleanup(curl);
        return false;
    }

    struct curl_slist* headers = NULL;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    headers = curl_slist_append(headers, "Expect:");

    curl_easy_setopt(curl, CURLOPT_URL, url);
    curl_easy_setopt(curl, CURLOPT_POST, 1L);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, (long)n);
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, discard_response_cb);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 5L);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 3L);
    curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "acap-object-detection/1.0");
    // HTTPS peer/host verification is on by default; leave it enabled.

    CURLcode rc = curl_easy_perform(curl);
    bool ok    = (rc == CURLE_OK);
    if (!ok) {
        syslog(LOG_WARNING,
               "Failed to POST detection to %s: %s",
               url,
               curl_easy_strerror(rc));
    } else {
        long http_code = 0;
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
        if (http_code >= 400) {
            syslog(LOG_WARNING,
                   "Server %s responded with HTTP %ld for detection %d",
                   url,
                   http_code,
                   index);
            ok = false;
        }
    }

    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
    return ok;
}

static bbox_t* setup_bbox(uint32_t channel) {
    // Create box drawer for channel
    bbox_t* bbox = bbox_view_new(channel);
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

static bool parse_and_postprocess_output_tensors(bbox_t* bbox,
                                                 model_tensor_output_t* tensor_outputs,
                                                 float confidence_threshold,
                                                 char** labels,
                                                 unsigned int* post_processing_ms) {
    box* boxes = NULL;
    struct timeval start_ts, end_ts;
    const char* post_url = getenv(DETECTION_POST_URL_ENV);

    // From here this is different dependent on model
    float* locations = (float*)tensor_outputs[0].data;
    float* classes   = (float*)tensor_outputs[1].data;

    bbox_clear(bbox);

    gettimeofday(&start_ts, NULL);

    float* scores            = (float*)tensor_outputs[2].data;
    float* nbr_detections    = (float*)tensor_outputs[3].data;
    int number_of_detections = (int)nbr_detections[0];
    if (number_of_detections == 0) {
        syslog(LOG_INFO, "No object is detected");
        return true;
    }
    boxes = (box*)malloc(sizeof(box) * number_of_detections);
    for (int i = 0; i < number_of_detections; i++) {
        boxes[i].y_min = locations[4 * i];
        boxes[i].x_min = locations[4 * i + 1];
        boxes[i].y_max = locations[4 * i + 2];
        boxes[i].x_max = locations[4 * i + 3];
        boxes[i].score = scores[i];
        boxes[i].label = classes[i];
    }
    gettimeofday(&end_ts, NULL);

    *post_processing_ms = (unsigned int)(((end_ts.tv_sec - start_ts.tv_sec) * 1000) +
                                         ((end_ts.tv_usec - start_ts.tv_usec) / 1000));
    if (*post_processing_ms != 0) {
        syslog(LOG_INFO, "Postprocessing in %u ms", *post_processing_ms);
    }

    for (int i = 0; i < number_of_detections; i++) {
        if (boxes[i].score >= confidence_threshold) {
            float top    = boxes[i].y_min;
            float left   = boxes[i].x_min;
            float bottom = boxes[i].y_max;
            float right  = boxes[i].x_max;

            syslog(LOG_INFO,
                   "Object %d: Classes: %s - Scores: %f - Locations: [%f,%f,%f,%f]",
                   i,
                   labels[boxes[i].label],
                   boxes[i].score,
                   top,
                   left,
                   bottom,
                   right);
            bbox_coordinates_frame_normalized(bbox);
            bbox_rectangle(bbox, left, top, right, bottom);

            // Forward this detection to the configured HTTPS endpoint (if any).
            send_detection_https(post_url,
                                 i,
                                 labels[boxes[i].label],
                                 boxes[i].score,
                                 top,
                                 left,
                                 bottom,
                                 right);
        }
    }

    if (!bbox_commit(bbox, 0u)) {
        panic("Failed to commit box drawer");
    }
    if (boxes) {
        free(boxes);
    }
    return true;
}

/**
 * @brief Main function that starts a stream with different options.
 */
int main(int argc, char** argv) {
    img_provider_t* image_provider        = NULL;
    model_provider_t* model_provider      = NULL;
    model_tensor_output_t* tensor_outputs = NULL;
    g_autoptr(GError) vdo_error           = NULL;
    bbox_t* bbox                          = NULL;
    img_info_t model_metadata             = {0};
    img_info_t image_metadata             = {0};

    // Stop main loop at signal
    signal(SIGTERM, shutdown);
    signal(SIGINT, shutdown);

    args_t args;
    parse_args(argc, argv, &args);

    // Global libcurl init for the HTTPS detection reporter.
    if (curl_global_init(CURL_GLOBAL_DEFAULT) != CURLE_OK) {
        syslog(LOG_WARNING, "curl_global_init failed; HTTPS reporting disabled");
    } else {
        const char* post_url = getenv(DETECTION_POST_URL_ENV);
        if (post_url && *post_url) {
            syslog(LOG_INFO, "Detections will be POSTed to %s", post_url);
        } else {
            syslog(LOG_INFO,
                   "%s not set; HTTPS reporting disabled",
                   DETECTION_POST_URL_ENV);
        }
    }

    # test the send_detection_https
    send_detection_https(getenv(DETECTION_POST_URL_ENV),)
                         0,
                         "test_label",
                         0.95f,
                         0.1f,
                         0.2f,
                         0.3f,
                         0.4f);

    char* device_name          = args.device_name;
    char* model_file           = args.model_file;
    const char* labels_file    = args.labels_file;
    const int threshold        = args.threshold;
    size_t number_of_classes   = 0;
    double vdo_framerate       = 30.0;
    uint32_t vdo_input_channel = 1;
    bool parse_tensors         = true;

    // Start by loading the model and get the model metadata
    size_t number_output_tensors = 0;
    model_provider =
        model_provider_new(model_file, device_name, labels_file, &number_output_tensors);
    if (!model_provider) {
        panic("%s: Could not create model provider", __func__);
    }

    tensor_outputs = calloc(number_output_tensors, sizeof(model_tensor_output_t));
    if (!tensor_outputs) {
        panic("%s: Could not allocate tensor outputs", __func__);
    }

    // Get the model format and model input dimension and pitches
    model_metadata = model_provider_get_model_metadata(model_provider);

    image_provider = img_provider_new(vdo_input_channel, &model_metadata, 2, vdo_framerate);
    if (!image_provider) {
        // It is considered an error if the img provider can not supply the
        // requested stream
        panic("%s: Could not create image provider", __func__);
    }

    image_metadata = img_provider_get_image_metadata(image_provider);
    model_provider_update_image_metadata(model_provider, &image_metadata);
    if (labels_file == NULL) {
        parse_tensors = false;
    }

    char** labels = NULL;          // This is the array of label strings. The label
                                   // entries points into the large label_file_data buffer.
    char* label_file_data = NULL;  // Buffer holding the complete collection of label strings.

    if (parse_tensors) {
        parse_labels(&labels, &label_file_data, labels_file, &number_of_classes);
        bbox = setup_bbox(vdo_input_channel);
    }

    // Get the fd here instead so it possible to select on them in main loop instead
    syslog(LOG_INFO, "Start fetching video frames from VDO for the inference");
    if (!img_provider_start(image_provider)) {
        panic("%s: Could not start image provider", __func__);
    }

    while (running) {
        struct timeval start_ts, end_ts;
        unsigned int inference_ms     = 0;
        unsigned int total_elapsed_ms = 0;

        g_autoptr(VdoBuffer) vdo_buf = img_provider_get_frame(image_provider);
        if (!vdo_buf) {
            // This can only happen if it is global rotation then
            // the stream has to be restarted because rotation has been changed.
            panic(
                "%s: No buffer because of changed global rotation. Application needs to be "
                "restarted",
                __func__);
        }
        gettimeofday(&start_ts, NULL);
        if (!model_run_inference(model_provider, vdo_buf)) {
            // No power
            if (!vdo_stream_buffer_unref(image_provider->vdo_stream, &vdo_buf, &vdo_error)) {
                if (!vdo_error_is_expected(&vdo_error)) {
                    panic("%s: Unexpected error: %s", __func__, vdo_error->message);
                }
                g_clear_error(&vdo_error);
            }
            // All buffers in vdo should be flushed since the call to run_inference may
            // have taken a lot of time so the buffers in vdo may be old
            img_provider_flush_all_frames(image_provider);
            continue;
        }
        gettimeofday(&end_ts, NULL);

        inference_ms = (unsigned int)(((end_ts.tv_sec - start_ts.tv_sec) * 1000) +
                                      ((end_ts.tv_usec - start_ts.tv_usec) / 1000));
        syslog(LOG_INFO, "Ran inference for %u ms", inference_ms);

        for (size_t i = 0; i < number_output_tensors; i++) {
            if (!model_get_tensor_output_info(model_provider, i, &tensor_outputs[i])) {
                panic("Failed to get output tensor info for %zu", i);
            }
        }
        total_elapsed_ms = inference_ms;

        if (parse_tensors) {
            unsigned int post_processing_ms = 0;
            float confidence_threshold      = (float)(threshold / 100.0);
            parse_and_postprocess_output_tensors(bbox,
                                                 tensor_outputs,
                                                 confidence_threshold,
                                                 labels,
                                                 &post_processing_ms);
            total_elapsed_ms += post_processing_ms;
        }

        // Check if the framerate from vdo should be changed
        img_provider_update_framerate(image_provider, total_elapsed_ms);

        // This will allow vdo to fill this buffer with data again
        if (!vdo_stream_buffer_unref(image_provider->vdo_stream, &vdo_buf, &vdo_error)) {
            if (!vdo_error_is_expected(&vdo_error)) {
                panic("%s: Unexpexted error: %s", __func__, vdo_error->message);
            }
            g_clear_error(&vdo_error);
        }
    }

    if (image_provider) {
        img_provider_destroy(image_provider);
    }
    if (model_provider) {
        model_provider_destroy(model_provider);
    }
    free(tensor_outputs);

    if (labels) {
        free(labels);
    }
    if (label_file_data) {
        free(label_file_data);
    }
    if (parse_tensors) {
        bbox_destroy(bbox);
    }

    curl_global_cleanup();

    syslog(LOG_INFO, "Exit %s", argv[0]);
    return 0;
}
