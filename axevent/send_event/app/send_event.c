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
#include <errno.h>
#include <fcgi_stdio.h>
#include <fcntl.h>
#include <glib-object.h>
#include <glib.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/uio.h>
#include <syslog.h>
#include <time.h>
#include <unistd.h>

// How often to fabricate a fake detection event (seconds).
#define FAKE_DETECTION_PERIOD_S 5

// Per-app persistent data directory (survives upgrades and reboots).
// Events are appended as JSON Lines; the FastCGI handler atomically
// drains this file when a collector reads /local/send_event/events.cgi.
#define EVENT_LOG_DIR   "/usr/local/packages/send_event/localdata/events"
#define EVENT_LOG_PATH  EVENT_LOG_DIR "/events.jsonl"
#define EVENT_LOG_STAGE EVENT_LOG_DIR "/events.draining.jsonl"
// Soft cap so an unreachable collector can't fill flash. Writes past this
// point are dropped (with a rate-limited warning) until the file is drained.
#define EVENT_LOG_MAX  (5 * 1024 * 1024)   // 5 MiB

// Set by the ACAP runtime for apps configured with httpConfig type fastCgi.
#define FCGI_SOCKET_ENV "FCGI_SOCKET_NAME"

typedef struct {
    AXEventHandler* event_handler;
    guint event_id;
    guint timer;
    gdouble value;
} AppData;

// State for the fake-detection generator.
typedef struct {
    guint counter;  // monotonically increasing event id
} FakeSenderCtx;

static AppData* app_data = NULL;

// The GLib timer and FastCGI request handler run in the same process.
// Serialize queue file access with an in-process mutex.
static pthread_mutex_t queue_lock = PTHREAD_MUTEX_INITIALIZER;

/* -------------------------------------------------------------------------- */
/* Local JSON-lines event queue                                               */
/* -------------------------------------------------------------------------- */

/**
 * brief Ensure the persistent event log directory exists.
 * Called once at startup. Safe to call repeatedly.
 */
static void ensure_event_log_dir(void) {
    // mkdir returns -1/EEXIST if it's already there; ignore that.
    if (mkdir(EVENT_LOG_DIR, 0750) < 0 && errno != EEXIST) {
        syslog(LOG_WARNING, "mkdir %s: %m", EVENT_LOG_DIR);
    }
}

/**
 * brief Append one JSON line to the local event queue.
 *
 * Refuses to grow past EVENT_LOG_MAX so a dead collector can't fill flash.
 */
static gboolean append_event_line(const char* json, size_t len) {
    gboolean ok = FALSE;
    pthread_mutex_lock(&queue_lock);

    struct stat st;
    if (stat(EVENT_LOG_PATH, &st) == 0 && st.st_size >= EVENT_LOG_MAX) {
        static time_t last_warn = 0;
        time_t now = time(NULL);
        if (now - last_warn > 60) {
            syslog(LOG_WARNING,
                   "events.jsonl at cap (%lld bytes); dropping until drained",
                   (long long)st.st_size);
            last_warn = now;
        }
        goto out;
    }

    int fd = open(EVENT_LOG_PATH,
                  O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0640);
    if (fd < 0) {
        syslog(LOG_WARNING, "open %s: %m", EVENT_LOG_PATH);
        goto out;
    }

    struct iovec iov[2] = {
        { .iov_base = (void*)json, .iov_len = len },
        { .iov_base = (void*)"\n", .iov_len = 1  },
    };
    ssize_t n = writev(fd, iov, 2);
    close(fd);

    if (n < 0) {
        syslog(LOG_WARNING, "write %s: %m", EVENT_LOG_PATH);
        goto out;
    }
    ok = TRUE;

out:
    pthread_mutex_unlock(&queue_lock);
    return ok;
}

/**
 * brief Generate one fake detection event and append it to the local queue.
 *
 * No network I/O. The event is picked up later by a collector calling
 * /local/send_event/events.cgi.
 */
static void generate_fake_detection(guint index) {
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
        return;
    }

    if (append_event_line(body, (size_t)n)) {
        syslog(LOG_INFO, "Fake event #%u queued (%d bytes)", index, n);
    }
}

/**
 * brief GLib timer callback: fabricate one fake detection each tick.
 */
static gboolean fake_detection_timer_cb(gpointer user_data) {
    FakeSenderCtx* ctx = (FakeSenderCtx*)user_data;
    generate_fake_detection(ctx->counter++);
    return G_SOURCE_CONTINUE;
}

/* -------------------------------------------------------------------------- */
/* FastCGI drain endpoint                                                     */
/* -------------------------------------------------------------------------- */

static void fcgi_headers(FCGX_Request* req,
                         const char* status,
                         const char* content_type,
                         long content_length) {
    FCGX_FPrintF(req->out, "Status: %s\r\n", status);
    FCGX_FPrintF(req->out, "Content-Type: %s\r\n", content_type);
    FCGX_FPrintF(req->out, "Cache-Control: no-store\r\n");
    FCGX_FPrintF(req->out, "Connection: close\r\n");
    if (content_length >= 0) {
        FCGX_FPrintF(req->out, "Content-Length: %ld\r\n", content_length);
    }
    FCGX_FPrintF(req->out, "\r\n");
}

static void fcgi_no_content(FCGX_Request* req) {
    FCGX_FPrintF(req->out,
                 "Status: 204 No Content\r\n"
                 "Cache-Control: no-store\r\n"
                 "Connection: close\r\n\r\n");
}

static void fcgi_handle_request(FCGX_Request* req) {
    const char* method = FCGX_GetParam("REQUEST_METHOD", req->envp);
    const char* query  = FCGX_GetParam("QUERY_STRING", req->envp);

    if (query && strstr(query, "debug=1")) {
        struct stat st;
        long size = (stat(EVENT_LOG_PATH, &st) == 0) ? (long)st.st_size : -1;
        fcgi_headers(req, "200 OK", "text/plain", -1);
        FCGX_FPrintF(req->out,
                     "events fastCgi probe\n"
                     "pid=%d\nuid=%d\nqueue=%s\nqueue_size=%ld\nmethod=%s\n",
                     (int)getpid(),
                     (int)geteuid(),
                     EVENT_LOG_PATH,
                     size,
                     method ? method : "unset");
        return;
    }

    if (method && strcmp(method, "HEAD") == 0) {
        struct stat st;
        if (stat(EVENT_LOG_PATH, &st) == 0 && st.st_size > 0) {
            fcgi_headers(req, "200 OK", "application/x-ndjson", (long)st.st_size);
        } else {
            fcgi_no_content(req);
        }
        return;
    }

    if (method && strcmp(method, "GET") != 0 && strcmp(method, "POST") != 0) {
        FCGX_FPrintF(req->out,
                     "Status: 405 Method Not Allowed\r\n"
                     "Allow: GET, HEAD\r\n\r\n");
        return;
    }

    pthread_mutex_lock(&queue_lock);

    struct stat st;
    if (stat(EVENT_LOG_PATH, &st) != 0 || st.st_size == 0) {
        pthread_mutex_unlock(&queue_lock);
        fcgi_no_content(req);
        return;
    }

    unlink(EVENT_LOG_STAGE);
    if (rename(EVENT_LOG_PATH, EVENT_LOG_STAGE) != 0) {
        pthread_mutex_unlock(&queue_lock);
        syslog(LOG_WARNING, "rename %s -> %s: %m", EVENT_LOG_PATH, EVENT_LOG_STAGE);
        fcgi_headers(req, "500 Internal Server Error", "text/plain", -1);
        FCGX_FPrintF(req->out, "rename failed\n");
        return;
    }

    pthread_mutex_unlock(&queue_lock);

    long size = (stat(EVENT_LOG_STAGE, &st) == 0) ? (long)st.st_size : -1;
    fcgi_headers(req, "200 OK", "application/x-ndjson", size);

    FILE* fp = fopen(EVENT_LOG_STAGE, "rb");
    if (fp) {
        char buf[4096];
        size_t nread;
        while ((nread = fread(buf, 1, sizeof(buf), fp)) > 0) {
            FCGX_PutStr(buf, (int)nread, req->out);
        }
        fclose(fp);
    } else {
        syslog(LOG_WARNING, "fopen %s: %m", EVENT_LOG_STAGE);
    }
    unlink(EVENT_LOG_STAGE);
}

static void* fcgi_thread_main(void* arg) {
    const char* socket_path = (const char*)arg;

    if (FCGX_Init() != 0) {
        syslog(LOG_ERR, "FCGX_Init failed");
        return NULL;
    }

    int sock = FCGX_OpenSocket(socket_path, 5);
    if (sock < 0) {
        syslog(LOG_ERR, "FCGX_OpenSocket(%s) failed", socket_path);
        return NULL;
    }
    chmod(socket_path, S_IRWXU | S_IRWXG | S_IRWXO);

    FCGX_Request req;
    if (FCGX_InitRequest(&req, sock, 0) != 0) {
        syslog(LOG_ERR, "FCGX_InitRequest failed");
        return NULL;
    }

    syslog(LOG_INFO, "FastCGI drain endpoint listening on %s", socket_path);

    while (FCGX_Accept_r(&req) == 0) {
        fcgi_handle_request(&req);
        FCGX_Finish_r(&req);
    }

    syslog(LOG_INFO, "FastCGI thread exiting");
    return NULL;
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
    pthread_t fcgi_thread  = 0;

    syslog(LOG_INFO, "Started logging from send event application");

    // Prepare local event queue (persistent, per-app).
    ensure_event_log_dir();
    syslog(LOG_INFO, "Local event queue: %s (cap %d bytes)",
           EVENT_LOG_PATH, EVENT_LOG_MAX);

    // Start FastCGI endpoint if the ACAP runtime supplied the socket path.
    const char* fcgi_socket = getenv(FCGI_SOCKET_ENV);
    if (fcgi_socket && *fcgi_socket) {
        if (pthread_create(&fcgi_thread, NULL, fcgi_thread_main, (void*)fcgi_socket) == 0) {
            pthread_detach(fcgi_thread);
        } else {
            syslog(LOG_ERR, "pthread_create(fcgi): %m");
        }
    } else {
        syslog(LOG_WARNING, "%s unset; FastCGI drain endpoint disabled", FCGI_SOCKET_ENV);
    }

    // Start the fake-detection generator: every FAKE_DETECTION_PERIOD_S
    // it appends one JSON line to the local queue.
    fake_timer_id = g_timeout_add_seconds(FAKE_DETECTION_PERIOD_S,
                                          fake_detection_timer_cb,
                                          &fake_ctx);
    syslog(LOG_INFO,
           "Fake detections will be queued to %s every %d s",
           EVENT_LOG_PATH, FAKE_DETECTION_PERIOD_S);

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

    // Cleanup event handler
    ax_event_handler_undeclare(app_data->event_handler, app_data->event_id, NULL);
    ax_event_handler_free(app_data->event_handler);
    free(app_data);

    // Free g_main_loop
    g_main_loop_unref(main_loop);
}
