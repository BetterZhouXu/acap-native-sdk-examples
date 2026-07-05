/**
 * Copyright (C) 2021, Axis Communications AB, Lund, Sweden
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
 * - send_event.c -
 *
 * This example illustrates how to send a stateful ONVIF event, which is
 * changing the value every 10th second.
 *
 * Error handling has been omitted for the sake of brevity.
 */
#include <axsdk/axevent.h>
#include <curl/curl.h>
#include <glib-object.h>
#include <glib.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <syslog.h>
#include <time.h>

// Target HTTPS endpoint for the fake-detection test.
// Edit this to point at your server (must start with https:// or http://).
#define DETECTION_POST_URL "https://55.128.197.187:8080/testaxis"
// How often to fire a fake detection event (seconds).
#define FAKE_DETECTION_PERIOD_S 5

typedef struct {
    AXEventHandler* event_handler;
    guint event_id;
    guint timer;
    gdouble value;
} AppData;

// State for the fake-detection HTTPS test loop.
typedef struct {
    guint counter;  // monotonically increasing event id
} FakeSenderCtx;

static AppData* app_data = NULL;

/* -------------------------------------------------------------------------- */
/* Fake-detection HTTPS test sender                                           */
/* -------------------------------------------------------------------------- */

/**
 * brief Discard any HTTP response body from the target server.
 */
static size_t discard_response_cb(void* ptr, size_t size, size_t nmemb, void* userdata) {
    (void)ptr;
    (void)userdata;
    return size * nmemb;
}

/**
 * brief POST one fake detection JSON payload to the configured HTTPS URL.
 *
 * This does not use real inference results — it produces a deterministic,
 * synthetic payload so you can verify that the camera can reach a server,
 * that TLS trust is set up correctly, and that the server accepts the
 * request. Returns TRUE on HTTP 2xx, FALSE otherwise. Never fatal.
 */
static gboolean send_fake_detection_https(const char* url, guint index) {
    if (!url || !*url) {
        return FALSE;
    }

    CURL* curl = curl_easy_init();
    if (!curl) {
        syslog(LOG_ERR, "curl_easy_init failed");
        return FALSE;
    }

    // Fake but plausible detection payload.
    char body[512];
    int n = snprintf(body,
                     sizeof(body),
                     "{\"timestamp\":%ld,\"index\":%u,\"fake\":true,"
                     "\"detections\":[{\"label\":\"person\",\"score\":0.9123,"
                     "\"bbox\":{\"top\":0.10,\"left\":0.20,"
                     "\"bottom\":0.50,\"right\":0.60}},"
                     "{\"label\":\"car\",\"score\":0.7841,"
                     "\"bbox\":{\"top\":0.30,\"left\":0.55,"
                     "\"bottom\":0.70,\"right\":0.90}}]}",
                     (long)time(NULL),
                     index);
    if (n < 0 || (size_t)n >= sizeof(body)) {
        syslog(LOG_ERR, "Fake detection JSON truncated");
        curl_easy_cleanup(curl);
        return FALSE;
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
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "acap-send-event-test/1.0");
    // HTTPS peer/host verification is on by default; leave it enabled.

    CURLcode rc = curl_easy_perform(curl);
    gboolean ok = (rc == CURLE_OK);
    if (!ok) {
        syslog(LOG_WARNING,
               "Fake event POST to %s failed: %s",
               url,
               curl_easy_strerror(rc));
    } else {
        long http_code = 0;
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
        if (http_code >= 400) {
            syslog(LOG_WARNING,
                   "Fake event POST to %s got HTTP %ld",
                   url,
                   http_code);
            ok = FALSE;
        } else {
            syslog(LOG_INFO,
                   "Fake event #%u POSTed to %s (HTTP %ld)",
                   index,
                   url,
                   http_code);
        }
    }

    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
    return ok;
}

/**
 * brief GLib timer callback: fire one fake detection event each tick.
 */
static gboolean fake_detection_timer_cb(gpointer user_data) {
    FakeSenderCtx* ctx = (FakeSenderCtx*)user_data;
    send_fake_detection_https(DETECTION_POST_URL, ctx->counter++);
    return G_SOURCE_CONTINUE;
}

/* -------------------------------------------------------------------------- */

/**
 * brief Send event.
 *
 * Send the previously declared event.
 *
 * param send_data Application data containing e.g. the event declaration id.
 * return TRUE
 */
static gboolean send_event(AppData* send_data) {
    AXEventKeyValueSet* key_value_set = NULL;
    AXEvent* event                    = NULL;

    key_value_set = ax_event_key_value_set_new();

    // Add the variable elements of the event to the set
    syslog(LOG_INFO, "Add value: %lf", send_data->value);
    ax_event_key_value_set_add_key_value(key_value_set,
                                         "Value",
                                         NULL,
                                         &send_data->value,
                                         AX_VALUE_TYPE_DOUBLE,
                                         NULL);

    // Create the event
    // Use ax_event_new2 since ax_event_new is deprecated from 3.2
    event = ax_event_new2(key_value_set, NULL);

    // The key/value set is no longer needed
    ax_event_key_value_set_free(key_value_set);

    // Send the event
    ax_event_handler_send_event(send_data->event_handler, send_data->event_id, event, NULL);

    syslog(LOG_INFO, "Send stateful event with value: %lf", send_data->value);

    ax_event_free(event);

    // Toggle value
    send_data->value = send_data->value >= 100 ? 0 : send_data->value + 10;

    // Returning TRUE keeps the timer going
    return TRUE;
}

/**
 * brief Callback function which is called when event declaration is completed.
 *
 * This callback will be called when the declaration has been registered
 * with the event system. The event declaration can now be used to send
 * events.
 *
 * param declaration Event declaration id.
 * param value Start value of the event.
 */
static void declaration_complete(guint declaration, gdouble* value) {
    syslog(LOG_INFO, "Declaration complete for: %d", declaration);

    app_data->value = *value;

    // Set up a timer to be called every 10th second
    app_data->timer = g_timeout_add_seconds(10, (GSourceFunc)send_event, app_data);
}

/**
 * brief Setup a declaration of an event.
 *
 * Declare a stateful ONVIF event that looks like this,
 * which is using ONVIF namespace "tns1".
 *
 * Topic: tns1:Monitoring/ProcessorUsage
 * <tt:MessageDescription IsProperty="true">
 *  <tt:Source>
 *   <tt:SimpleItemDescription Name=”Token” Type=”tt:ReferenceToken”/>
 *  </tt:Source>
 *  <tt:Data>
 *   <tt:SimpleItemDescription Name="Value" Type="xs:float"/>
 *  </tt:Data>
 * </tt:MessageDescription>
 *
 * Value = 0 <-- The initial value will be set to 0.0
 *
 * param event_handler Event handler.
 * return declaration id as integer.
 */
static guint setup_declaration(AXEventHandler* event_handler, gdouble* start_value) {
    AXEventKeyValueSet* key_value_set = NULL;
    guint declaration                 = 0;
    guint token                       = 0;
    GError* error                     = NULL;

    // Create keys, namespaces and nice names for the event
    key_value_set = ax_event_key_value_set_new();
    ax_event_key_value_set_add_key_value(key_value_set,
                                         "topic0",
                                         "tns1",
                                         "Monitoring",
                                         AX_VALUE_TYPE_STRING,
                                         NULL);
    ax_event_key_value_set_add_key_value(key_value_set,
                                         "topic1",
                                         "tns1",
                                         "ProcessorUsage",
                                         AX_VALUE_TYPE_STRING,
                                         NULL);
    ax_event_key_value_set_add_key_value(key_value_set,
                                         "Token",
                                         NULL,
                                         &token,
                                         AX_VALUE_TYPE_INT,
                                         NULL);
    ax_event_key_value_set_add_key_value(key_value_set,
                                         "Value",
                                         NULL,
                                         &start_value,
                                         AX_VALUE_TYPE_DOUBLE,
                                         NULL);
    ax_event_key_value_set_mark_as_source(key_value_set, "Token", NULL, NULL);
    ax_event_key_value_set_mark_as_user_defined(key_value_set,
                                                "Token",
                                                NULL,
                                                "wstype:tt:ReferenceToken",
                                                NULL);
    ax_event_key_value_set_mark_as_data(key_value_set, "Value", NULL, NULL);
    ax_event_key_value_set_mark_as_user_defined(key_value_set,
                                                "Value",
                                                NULL,
                                                "wstype:xs:float",
                                                NULL);

    // Declare event
    if (!ax_event_handler_declare(event_handler,
                                  key_value_set,
                                  FALSE,  // Indicate a property state event
                                  &declaration,
                                  (AXDeclarationCompleteCallback)declaration_complete,
                                  start_value,
                                  &error)) {
        syslog(LOG_WARNING, "Could not declare: %s", error->message);
        g_error_free(error);
    }

    // The key/value set is no longer needed
    ax_event_key_value_set_free(key_value_set);
    return declaration;
}

/**
 * brief Main function which sends an event.
 */
gint main(void) {
    GMainLoop* main_loop   = NULL;
    gdouble start_value    = 0.0;
    FakeSenderCtx fake_ctx = {0};
    guint fake_timer_id    = 0;

    syslog(LOG_INFO, "Started logging from send event application");

    // Global libcurl init for the HTTPS test sender.
    if (curl_global_init(CURL_GLOBAL_DEFAULT) != CURLE_OK) {
        syslog(LOG_WARNING, "curl_global_init failed; HTTPS test disabled");
    } else {
        fake_timer_id = g_timeout_add_seconds(FAKE_DETECTION_PERIOD_S,
                                              fake_detection_timer_cb,
                                              &fake_ctx);
        syslog(LOG_INFO,
               "Fake detection events will be POSTed to %s every %d s",
               DETECTION_POST_URL,
               FAKE_DETECTION_PERIOD_S);
    }

    // Event handler
    app_data                = calloc(1, sizeof(AppData));
    app_data->event_handler = ax_event_handler_new();
    app_data->event_id      = setup_declaration(app_data->event_handler, &start_value);

    // Main loop
    main_loop = g_main_loop_new(NULL, FALSE);
    g_main_loop_run(main_loop);

    // Cleanup fake sender
    if (fake_timer_id) {
        g_source_remove(fake_timer_id);
    }
    curl_global_cleanup();

    // Cleanup event handler
    ax_event_handler_undeclare(app_data->event_handler, app_data->event_id, NULL);
    ax_event_handler_free(app_data->event_handler);
    free(app_data);

    // Free g_main_loop
    g_main_loop_unref(main_loop);
}
