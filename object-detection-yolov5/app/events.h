/* SPDX-License-Identifier: Apache-2.0 */
#pragma once

#include "obb.h"
#include <glib.h>

#define EVENTS_DIRECTORY "/usr/local/packages/detection/localdata"
#define EVENTS_PATH EVENTS_DIRECTORY "/events.jsonl"
#define EVENTS_LOCK_PATH EVENTS_DIRECTORY "/events.lock"
#define EVENTS_MAX_RECORDS 1000
#define EVENTS_MAX_BYTES (1024 * 1024)

/* Call before starting VDO/larod or any threads. Clears previous-run history. */
bool events_start(void);
bool events_server_alive(void);
void events_stop(void);
bool events_publish(const obb_detection_t* detections, int count, char* const* labels,
                    int input_width, int input_height, gint64 capture_time_us);

