/**
 * Copyright (C) 2026 Axis Communications AB, Lund, Sweden
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 */

#ifndef YOLOV8_OBB_H
#define YOLOV8_OBB_H

#include <stdbool.h>
#include <stddef.h>

#define YOLOV8_OBB_MAX_RESULTS 300

typedef struct obb_point {
    float x;
    float y;
} obb_point_t;

typedef struct obb_detection {
    float center_x;
    float center_y;
    float width;
    float height;
    float angle;
    float confidence;
    int label_index;
    obb_point_t corners[4];
} obb_detection_t;

typedef struct yolov8_obb_params {
    int input_width;
    int input_height;
    int num_classes;
    int num_detections;
    int num_features;
} yolov8_obb_params_t;

/**
 * Decode a [1, 4 + classes + 1, predictions] Ultralytics YOLOv8 OBB float tensor.
 * Coordinates must be normalized and the final channel must contain angles in radians.
 */
bool yolov8_obb_decode(const float* tensor,
                       const yolov8_obb_params_t* params,
                       float confidence_threshold,
                       float iou_threshold,
                       obb_detection_t* output,
                       size_t output_capacity,
                       size_t* output_count);

#endif



