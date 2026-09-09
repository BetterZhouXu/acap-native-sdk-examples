/**
 * Copyright (C) 2026 Axis Communications AB, Lund, Sweden
 * Licensed under the Apache License, Version 2.0.
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

static void finish_empty(FCGX_Request* request, const char* status) {
    FCGX_FPrintF(request->out,
                 "Status: %s\r\nContent-Type: application/json\r\n"
                 "Cache-Control: no-store\r\n\r\n",
                 status);
    FCGX_Finish_r(request);
}

static void handle_request(FCGX_Request* request) {
    const char* method = FCGX_GetParam("REQUEST_METHOD", request->envp);
    if (!method || strcmp(method, "GET") != 0) {
        FCGX_FPrintF(request->out,
                     "Status: 405 Method Not Allowed\r\nAllow: GET\r\n"
                     "Content-Type: application/json\r\nCache-Control: no-store\r\n\r\n"
                     "{\"error\":\"method not allowed\"}\n");
        FCGX_Finish_r(request);
        return;
    }

    char* json    = NULL;
    size_t length = 0;
    int claimed   = detection_result_claim(&json, &length);
    if (claimed == 0) {
        finish_empty(request, "204 No Content");
        return;
    }
    if (claimed < 0 || length > INT_MAX) {
        free(json);
        syslog(LOG_ERR, "Failed to claim detection result");
        finish_empty(request, "500 Internal Server Error");
        return;
    }

    int header = FCGX_FPrintF(request->out,
                              "Status: 200 OK\r\nContent-Type: application/json\r\n"
                              "Content-Length: %zu\r\nCache-Control: no-store\r\n\r\n",
                              length);
    int body = header < 0 ? -1 : FCGX_PutStr(json, (int)length, request->out);
    int flush = body == (int)length ? FCGX_FFlush(request->out) : -1;
    if (flush == 0) {
        if (!detection_result_consume()) {
            syslog(LOG_ERR, "Failed to remove served detection result");
        }
    } else if (!detection_result_release()) {
        syslog(LOG_ERR, "Failed to retain detection result after response failure");
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
    return NULL;
}

bool detection_fastcgi_start(void) {
    const char* path = getenv(FCGI_SOCKET_NAME);
    if (!path || FCGX_Init() != 0) {
        syslog(LOG_ERR, "Failed to initialize FastCGI");
        return false;
    }

    socket_fd = FCGX_OpenSocket(path, 5);
    if (socket_fd < 0 || chmod(path, S_IRWXU | S_IRWXG | S_IRWXO) != 0) {
        syslog(LOG_ERR, "Failed to open FastCGI socket");
        if (socket_fd >= 0) {
            close(socket_fd);
        }
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
    return true;
}

