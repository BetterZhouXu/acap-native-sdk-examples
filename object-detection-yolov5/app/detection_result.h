/**
 * Copyright (C) 2026 Axis Communications AB, Lund, Sweden
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 */

#ifndef DETECTION_RESULT_H
#define DETECTION_RESULT_H

#include <stdbool.h>
#include <stddef.h>

/** Remove stale result files left by an earlier application instance. */
bool detection_result_init(void);

/**
 * Atomically publish JSON unless an unconsumed result is already pending.
 *
 * Returns true both when the JSON is published and when an older result is pending. Returns false
 * on an I/O error.
 */
bool detection_result_publish(const char* json, size_t length);

/**
 * Atomically claim and read the pending result.
 *
 * Returns 1 and assigns a heap-allocated JSON buffer on success, 0 when no result is pending, and
 * -1 on error. The caller must free a returned buffer.
 */
int detection_result_claim(char** json, size_t* length);

/** Delete the claimed result after it has been served successfully. */
bool detection_result_consume(void);

/** Put a claimed result back so a later request can retry it. */
bool detection_result_release(void);

#endif

