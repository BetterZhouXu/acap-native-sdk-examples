/* SPDX-License-Identifier: Apache-2.0 */
#pragma once

#include <stdbool.h>
#include <stddef.h>

#define OBB_MAX_DETECTIONS 100
#define OBB_MAX_CANDIDATES 1000

typedef enum { OBB_FLOAT32, OBB_UINT8, OBB_INT8 } obb_dtype_t;
typedef struct { float x, y; } obb_point_t;
typedef struct {
    int class_id;
    int candidate_id;
    float confidence;
    /* All geometry is in model-input pixels; angle is radians. */
    float cx, cy, width, height, angle;
    obb_point_t corners[4];
} obb_detection_t;

typedef struct {
    size_t candidates;
    int classes;
    int input_width, input_height;
    obb_dtype_t dtype;
    float scale;
    int zero_point;
    bool normalized;
} obb_config_t;

void obb_corners(obb_detection_t* detection);
float obb_iou(const obb_detection_t* a, const obb_detection_t* b);
/* Returns -1 for invalid metadata/buffer/allocation, otherwise the retained count. */
int obb_decode(const void* data, size_t bytes, const obb_config_t* config,
               float confidence_threshold, float iou_threshold,
               obb_detection_t output[OBB_MAX_DETECTIONS]);
