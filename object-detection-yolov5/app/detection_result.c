/**
 * Copyright (C) 2026 Axis Communications AB, Lund, Sweden
 * Licensed under the Apache License, Version 2.0.
 */

#include "detection_result.h"

#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#ifndef DETECTION_RESULT_PATH
#define DETECTION_RESULT_PATH "/usr/local/packages/detection/localdata/detection-result.json"
#endif

#define RESULT_PATH DETECTION_RESULT_PATH
#define CLAIMED_PATH RESULT_PATH ".claimed"
#define TEMP_PATH RESULT_PATH ".tmp"

static bool remove_if_present(const char* path) {
    return unlink(path) == 0 || errno == ENOENT;
}

static bool write_all(int fd, const char* data, size_t length) {
    size_t offset = 0;
    while (offset < length) {
        ssize_t written = write(fd, data + offset, length - offset);
        if (written < 0 && errno == EINTR) {
            continue;
        }
        if (written <= 0) {
            return false;
        }
        offset += (size_t)written;
    }
    return true;
}

bool detection_result_init(void) {
    return remove_if_present(TEMP_PATH) && remove_if_present(CLAIMED_PATH) &&
           remove_if_present(RESULT_PATH);
}

bool detection_result_publish(const char* json, size_t length) {
    if (!json || length == 0) {
        errno = EINVAL;
        return false;
    }
    if (access(RESULT_PATH, F_OK) == 0) {
        return true;
    }
    if (errno != ENOENT || !remove_if_present(TEMP_PATH)) {
        return false;
    }

    int fd = open(TEMP_PATH, O_WRONLY | O_CREAT | O_EXCL, S_IRUSR | S_IWUSR);
    if (fd < 0) {
        return false;
    }
    bool success = write_all(fd, json, length);
    if (success && fsync(fd) != 0) {
        success = false;
    }
    if (close(fd) != 0) {
        success = false;
    }
    if (success && link(TEMP_PATH, RESULT_PATH) != 0 && errno != EEXIST) {
        success = false;
    }
    if (!remove_if_present(TEMP_PATH)) {
        success = false;
    }
    return success;
}

static int read_claimed(char** json, size_t* length) {
    int fd = open(CLAIMED_PATH, O_RDONLY);
    if (fd < 0) {
        return -1;
    }

    struct stat info;
    if (fstat(fd, &info) != 0 || info.st_size < 0 || (uintmax_t)info.st_size > SIZE_MAX - 1) {
        close(fd);
        return -1;
    }

    size_t expected = (size_t)info.st_size;
    char* buffer    = malloc(expected + 1);
    if (!buffer) {
        close(fd);
        return -1;
    }

    size_t offset = 0;
    while (offset < expected) {
        ssize_t count = read(fd, buffer + offset, expected - offset);
        if (count < 0 && errno == EINTR) {
            continue;
        }
        if (count <= 0) {
            free(buffer);
            close(fd);
            return -1;
        }
        offset += (size_t)count;
    }
    if (close(fd) != 0) {
        free(buffer);
        return -1;
    }

    buffer[expected] = '\0';
    *json            = buffer;
    *length          = expected;
    return 1;
}

int detection_result_claim(char** json, size_t* length) {
    if (!json || !length) {
        errno = EINVAL;
        return -1;
    }
    *json   = NULL;
    *length = 0;

    if (access(CLAIMED_PATH, F_OK) != 0) {
        if (errno != ENOENT) {
            return -1;
        }
        if (rename(RESULT_PATH, CLAIMED_PATH) != 0) {
            return errno == ENOENT ? 0 : -1;
        }
    }
    return read_claimed(json, length);
}

bool detection_result_consume(void) {
    return remove_if_present(CLAIMED_PATH);
}

bool detection_result_release(void) {
    if (link(CLAIMED_PATH, RESULT_PATH) == 0) {
        return remove_if_present(CLAIMED_PATH);
    }
    return errno == ENOENT || errno == EEXIST;
}


