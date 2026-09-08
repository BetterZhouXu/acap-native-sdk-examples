/**
 * Copyright (C) 2026 Axis Communications AB, Lund, Sweden
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 */

#include "detection_fastcgi.h"
#include "detection_result.h"

#include <fcgiapp.h>
#include <limits.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <syslog.h>
#include <unistd.h>

#define FCGI_SOCKET_NAME "FCGI_SOCKET_NAME"

static int socket_fd = -1;

static void finish_empty_response(FCGX_Request* request, const char* status) {
    FCGX_FPrintF(request->out,
                 "Status: %s\r\n"
                 "Content-Type: application/json\r\n"
                 "Cache-Control: no-store\r\n\r\n",
                 status);
    FCGX_Finish_r(request);
}

static void handle_request(FCGX_Request* request) {
    const char* method = FCGX_GetParam("REQUEST_METHOD", request->envp);
    if (!method || strcmp(method, "GET") != 0) {
        FCGX_FPrintF(request->out,
                     "Status: 405 Method Not Allowed\r\n"
                     "Allow: GET\r\n"
                     "Content-Type: application/json\r\n"
                     "Cache-Control: no-store\r\n\r\n"
                     "{\"error\":\"method not allowed\"}\n");
        FCGX_Finish_r(request);
        return;
    }

    char* json   = NULL;
    size_t length = 0;
    int claimed  = detection_result_claim(&json, &length);
    if (claimed == 0) {
        finish_empty_response(request, "204 No Content");
        return;
    }
    if (claimed < 0 || length > INT_MAX) {
        free(json);
        syslog(LOG_ERR, "Failed to claim detection result");
        finish_empty_response(request, "500 Internal Server Error");
        return;
    }

    int header_status = FCGX_FPrintF(request->out,
                                     "Status: 200 OK\r\n"
                                     "Content-Type: application/json\r\n"
                                     "Content-Length: %zu\r\n"
                                     "Cache-Control: no-store\r\n\r\n",
                                     length);
    int body_status = header_status < 0 ? -1 : FCGX_PutStr(json, (int)length, request->out);
    int flush_status = body_status == (int)length ? FCGX_FFlush(request->out) : -1;

    if (flush_status == 0) {
        if (!detection_result_consume()) {
            syslog(LOG_ERR, "Failed to remove served detection result");
        }
    } else {
        syslog(LOG_WARNING, "Detection response failed; retaining result for retry");
        if (!detection_result_release()) {
            syslog(LOG_ERR, "Failed to release detection result");
        }
    }

    free(json);
    FCGX_Finish_r(request);
}

static void* request_loop(void* data) {
    (void)data;
    FCGX_Request request;

    if (FCGX_InitRequest(&request, socket_fd, 0) != 0) {
        syslog(LOG_ERR, "FCGX_InitRequest failed");
        return NULL;
    }

    while (FCGX_Accept_r(&request) == 0) {
        handle_request(&request);
    }

    syslog(LOG_WARNING, "FastCGI request loop stopped");
    return NULL;
}

bool detection_fastcgi_start(void) {
    const char* socket_path = getenv(FCGI_SOCKET_NAME);
    if (!socket_path) {
        syslog(LOG_ERR, "Failed to get environment variable %s", FCGI_SOCKET_NAME);
        return false;
    }

    if (FCGX_Init() != 0) {
        syslog(LOG_ERR, "FCGX_Init failed");
        return false;
    }

    socket_fd = FCGX_OpenSocket(socket_path, 5);
    if (socket_fd < 0) {
        syslog(LOG_ERR, "Failed to open FastCGI socket %s", socket_path);
        return false;
    }
    if (chmod(socket_path, S_IRWXU | S_IRWXG | S_IRWXO) != 0) {
        syslog(LOG_ERR, "Failed to set FastCGI socket permissions");
        close(socket_fd);
        socket_fd = -1;
        return false;
    }

    pthread_t thread;
    int status = pthread_create(&thread, NULL, request_loop, NULL);
    if (status != 0) {
        syslog(LOG_ERR, "Failed to create FastCGI thread: %s", strerror(status));
        close(socket_fd);
        socket_fd = -1;
        return false;
    }
    if (pthread_detach(thread) != 0) {
        syslog(LOG_WARNING, "Failed to detach FastCGI thread");
    }

    syslog(LOG_INFO, "Detection result endpoint started");
    return true;
}

