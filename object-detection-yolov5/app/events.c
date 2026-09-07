/* SPDX-License-Identifier: Apache-2.0 */
#include "events.h"

#include <errno.h>
#include <fcgiapp.h>
#include <fcntl.h>
#include <math.h>
#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <sys/prctl.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <syslog.h>
#include <unistd.h>

static pid_t server_pid = -1;
static char* socket_path;
static char* session_id;
/* Pending writes only: successfully saved records must not be republished. */
static GQueue records = G_QUEUE_INIT;
static size_t record_bytes;
static guint64 sequence, frame;
static bool dirty;

static bool save(const char* data, gssize size) {
    GError* error = NULL;
    /* GLib writes a temporary sibling and renames it over the destination. */
    if (!g_file_set_contents(EVENTS_PATH, data, size, &error)) {
        syslog(LOG_ERR, "Cannot save detection results: %s", error->message);
        g_clear_error(&error);
        return false;
    }
    return true;
}

static int lock_store(void) {
    /* Open separately in each process; an inherited flock fd would share ownership. */
    int fd = open(EVENTS_LOCK_PATH, O_CREAT | O_RDWR | O_CLOEXEC, 0600);
    if (fd < 0) {
        syslog(LOG_ERR, "Cannot open detection store lock: %s", strerror(errno));
        return -1;
    }
    while (flock(fd, LOCK_EX) < 0) {
        if (errno == EINTR) continue;
        syslog(LOG_ERR, "Cannot lock detection store: %s", strerror(errno));
        close(fd);
        return -1;
    }
    return fd;
}

/* Caller must hold the store lock. */
static bool load(char** contents, gsize* length) {
    GError* error = NULL;
    if (!g_file_get_contents(EVENTS_PATH, contents, length, &error)) {
        syslog(LOG_ERR, "Cannot read detection results: %s", error->message);
        g_clear_error(&error);
        return false;
    }
    if (*length > EVENTS_MAX_BYTES) {
        syslog(LOG_ERR, "Detection store exceeds its size limit");
        g_clear_pointer(contents, g_free);
        return false;
    }
    return true;
}

static bool clear_served(const char* served) {
    int lock_fd = lock_store();
    if (lock_fd < 0) return false;
    char* current = NULL;
    gsize length = 0;
    bool success = load(&current, &length);
    if (success) {
        /* Exact records include unique event IDs. Do not delete new records that
         * the inference process saved while this response was being transmitted. */
        char** served_lines = g_strsplit(served, "\n", -1);
        char** current_lines = g_strsplit(current, "\n", -1);
        GHashTable* consumed = g_hash_table_new(g_str_hash, g_str_equal);
        for (char** line = served_lines; *line; ++line)
            if (**line) g_hash_table_add(consumed, *line);
        GString* remaining = g_string_sized_new(length);
        for (char** line = current_lines; *line; ++line) {
            if (**line && !g_hash_table_contains(consumed, *line)) {
                g_string_append(remaining, *line);
                g_string_append_c(remaining, '\n');
            }
        }
        success = save(remaining->str, (gssize)remaining->len);
        g_string_free(remaining, TRUE);
        g_hash_table_destroy(consumed);
        g_strfreev(served_lines);
        g_strfreev(current_lines);
    }
    g_free(current);
    close(lock_fd);
    return success;
}

static void serve(FCGX_Request* request) {
    while (FCGX_Accept_r(request) == 0) {
        const char* method = FCGX_GetParam("REQUEST_METHOD", request->envp);
        const char* uri = FCGX_GetParam("DOCUMENT_URI", request->envp);
        if (!uri) uri = FCGX_GetParam("REQUEST_URI", request->envp);
        const char* expected = "/local/detection/events.cgi";
        size_t path_length = uri ? strcspn(uri, "?") : 0;
        if (!uri || path_length != strlen(expected) || strncmp(uri, expected, path_length)) {
            FCGX_FPrintF(request->out, "Status: 404 Not Found\r\nContent-Length: 0\r\n\r\n");
        } else if (!method || strcmp(method, "GET")) {
            FCGX_FPrintF(request->out,
                        "Status: 405 Method Not Allowed\r\nAllow: GET\r\nContent-Length: 0\r\n\r\n");
        } else {
            gchar* contents = NULL;
            gsize length = 0;
            int lock_fd = lock_store();
            bool loaded = lock_fd >= 0 && load(&contents, &length);
            if (lock_fd >= 0) close(lock_fd);
            if (!loaded) {
                FCGX_FPrintF(request->out,
                            "Status: 500 Internal Server Error\r\nContent-Length: 0\r\n\r\n");
            } else {
                int header_status = FCGX_FPrintF(request->out,
                            "Status: 200 OK\r\nContent-Type: application/x-ndjson; charset=utf-8\r\n"
                            "Cache-Control: no-store\r\nX-Content-Type-Options: nosniff\r\n"
                            "Content-Length: %zu\r\n\r\n", (size_t)length);
                bool sent = header_status >= 0 &&
                            FCGX_PutStr(contents, (int)length, request->out) == (int)length &&
                            FCGX_FFlush(request->out) == 0;
                if (sent && length && !clear_served(contents))
                    syslog(LOG_ERR, "Response sent but results could not be cleared; poll may repeat them");
            }
            g_free(contents);
        }
        FCGX_Finish_r(request);
    }
}

void events_stop(void) {
    if (server_pid > 0) {
        kill(server_pid, SIGTERM);
        while (waitpid(server_pid, NULL, 0) < 0 && errno == EINTR) {}
        server_pid = -1;
    }
    if (socket_path) unlink(socket_path);
    g_clear_pointer(&socket_path, g_free);
    g_clear_pointer(&session_id, g_free);
    while (!g_queue_is_empty(&records)) g_free(g_queue_pop_head(&records));
    record_bytes = 0;
}

bool events_server_alive(void) {
    if (server_pid <= 0) return false;
    pid_t result = waitpid(server_pid, NULL, WNOHANG);
    if (result == 0 || (result < 0 && errno == EINTR)) return true;
    server_pid = -1;
    return false;
}

bool events_start(void) {
    const char* path = getenv("FCGI_SOCKET_NAME");
    if (!path || !*path) {
        syslog(LOG_ERR, "FCGI_SOCKET_NAME is missing; install with the FastCGI manifest");
        return false;
    }
    if (g_mkdir_with_parents(EVENTS_DIRECTORY, 0700) < 0 ||
        !save("", 0)) return false;
    if (FCGX_Init() != 0) return false;
    int socket_fd = FCGX_OpenSocket(path, 16);
    if (socket_fd < 0) return false;
    socket_path = g_strdup(path);
    FCGX_Request request;
    /* Axis's HTTP server must be able to connect to this local Unix socket. */
    if (chmod(path, 0777) < 0 || FCGX_InitRequest(&request, socket_fd, 0) != 0) {
        close(socket_fd);
        events_stop();
        return false;
    }
    pid_t parent_pid = getpid();
    server_pid = fork();
    if (server_pid == 0) {
        signal(SIGTERM, SIG_DFL);
        signal(SIGINT, SIG_DFL);
        signal(SIGPIPE, SIG_IGN);
        /* Also stop the worker if inference crashes or is killed with SIGKILL. */
        if (prctl(PR_SET_PDEATHSIG, SIGTERM) < 0 || getppid() != parent_pid) _exit(1);
        serve(&request);
        close(socket_fd);
        _exit(1);
    }
    close(socket_fd);
    if (server_pid < 0) {
        events_stop();
        return false;
    }
    session_id = g_uuid_string_random();
    sequence = frame = 0;
    dirty = false;
    if (atexit(events_stop) != 0) {
        events_stop();
        return false;
    }
    return true;
}

static void append_string(GString* json, const char* text) {
    g_string_append_c(json, '"');
    for (const unsigned char* p = (const unsigned char*)text; *p; ++p) {
        if (*p == '"' || *p == '\\') {
            g_string_append_c(json, '\\');
            g_string_append_c(json, (char)*p);
        } else if (*p < 0x20) {
            g_string_append_printf(json, "\\u%04x", (unsigned int)*p);
        } else {
            g_string_append_c(json, (char)*p);
        }
    }
    g_string_append_c(json, '"');
}

static void append_number(GString* json, double number) {
    char buffer[G_ASCII_DTOSTR_BUF_SIZE];
    g_ascii_formatd(buffer, sizeof(buffer), "%.7g", number);
    g_string_append(json, buffer);
}

bool events_publish(const obb_detection_t* detections, int count, char* const* labels,
                    int input_width, int input_height, gint64 capture_time_us) {
    ++frame;
    for (int i = 0; i < count; ++i) {
        const obb_detection_t* d = &detections[i];
        GString* line = g_string_new("{\"schema_version\":2,\"event_id\":\"");
        g_string_append_printf(line, "%s:%" G_GUINT64_FORMAT "\",\"frame_id\":%" G_GUINT64_FORMAT
                               ",\"timestamp_us\":%" G_GINT64_FORMAT ",\"class_id\":%d,\"label\":",
                               session_id, ++sequence, frame, capture_time_us, d->class_id);
        append_string(line, labels[d->class_id]);
        g_string_append(line, ",\"confidence\":");
        append_number(line, d->confidence);
        float left = d->corners[0].x, right = left;
        float top = d->corners[0].y, bottom = top;
        for (int corner = 1; corner < 4; ++corner) {
            left = fminf(left, d->corners[corner].x);
            right = fmaxf(right, d->corners[corner].x);
            top = fminf(top, d->corners[corner].y);
            bottom = fmaxf(bottom, d->corners[corner].y);
        }
        g_string_append_printf(line, ",\"model_input\":[%d,%d],\"bbox\":{\"cx\":",
                               input_width, input_height);
        append_number(line, (left + right) * 0.5f);
        g_string_append(line, ",\"cy\":");
        append_number(line, (top + bottom) * 0.5f);
        g_string_append(line, ",\"width\":");
        append_number(line, right - left);
        g_string_append(line, ",\"height\":");
        append_number(line, bottom - top);
        g_string_append(line, "},\"bbox_normalized\":[");
        append_number(line, left / input_width);
        g_string_append_c(line, ',');
        append_number(line, top / input_height);
        g_string_append_c(line, ',');
        append_number(line, right / input_width);
        g_string_append_c(line, ',');
        append_number(line, bottom / input_height);
        g_string_append(line, "]}\n");
        record_bytes += line->len;
        g_queue_push_tail(&records, g_string_free(line, FALSE));
        while (records.length > EVENTS_MAX_RECORDS || record_bytes > EVENTS_MAX_BYTES) {
            char* oldest = g_queue_pop_head(&records);
            record_bytes -= strlen(oldest);
            g_free(oldest);
        }
        dirty = true;
    }
    if (!dirty) return true;
    int lock_fd = lock_store();
    if (lock_fd < 0) return false;
    char* current = NULL;
    gsize length = 0;
    if (!load(&current, &length)) {
        close(lock_fd);
        return false;
    }
    GString* snapshot = g_string_new_len(current, (gssize)length);
    g_free(current);
    for (GList* item = records.head; item; item = item->next)
        g_string_append(snapshot, item->data);
    size_t lines = 0, offset = 0;
    for (size_t i = 0; i < snapshot->len; ++i)
        if (snapshot->str[i] == '\n') ++lines;
    while (lines > EVENTS_MAX_RECORDS || snapshot->len - offset > EVENTS_MAX_BYTES) {
        char* newline = strchr(snapshot->str + offset, '\n');
        if (!newline) break;
        offset = (size_t)(newline - snapshot->str) + 1;
        --lines;
    }
    g_string_erase(snapshot, 0, (gssize)offset);
    bool success = save(snapshot->str, (gssize)snapshot->len);
    g_string_free(snapshot, TRUE);
    close(lock_fd);
    if (success) {
        while (!g_queue_is_empty(&records)) g_free(g_queue_pop_head(&records));
        record_bytes = 0;
        dirty = false;
    }
    return success;
}

