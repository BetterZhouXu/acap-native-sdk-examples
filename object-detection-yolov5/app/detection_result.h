/**
 * Copyright (C) 2026 Axis Communications AB, Lund, Sweden
 * Licensed under the Apache License, Version 2.0.
 */

#ifndef DETECTION_RESULT_H
#define DETECTION_RESULT_H

#include <stdbool.h>
#include <stddef.h>

bool detection_result_init(void);
bool detection_result_publish(const char* json, size_t length);
int detection_result_claim(char** json, size_t* length);
bool detection_result_consume(void);
bool detection_result_release(void);

#endif

