/**
 * Copyright (C) 2026 Axis Communications AB, Lund, Sweden
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 */

#ifndef DETECTION_FASTCGI_H
#define DETECTION_FASTCGI_H

#include <stdbool.h>

/** Start the detection-result FastCGI request loop on a background thread. */
bool detection_fastcgi_start(void);

#endif

