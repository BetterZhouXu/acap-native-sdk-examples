/**
 * Copyright (C) 2026 Axis Communications AB, Lund, Sweden
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 */

#include "yolov8_obb.h"

#include <math.h>
#include <stdlib.h>

#define MAX_POLYGON_POINTS 8
#define PARALLEL_EPSILON 1.0e-7F

static void calculate_corners(obb_detection_t* detection) {
    const float half_width  = detection->width / 2.0F;
    const float half_height = detection->height / 2.0F;
    const float cosine      = cosf(detection->angle);
    const float sine        = sinf(detection->angle);
    const float offsets[4][2] = {
        {-half_width, -half_height},
        {half_width, -half_height},
        {half_width, half_height},
        {-half_width, half_height},
    };

    for (size_t i = 0; i < 4; i++) {
        detection->corners[i].x = detection->center_x + offsets[i][0] * cosine -
                                  offsets[i][1] * sine;
        detection->corners[i].y = detection->center_y + offsets[i][0] * sine +
                                  offsets[i][1] * cosine;
    }
}

static float cross_product(obb_point_t first, obb_point_t second, obb_point_t third) {
    return (second.x - first.x) * (third.y - first.y) -
           (second.y - first.y) * (third.x - first.x);
}

static bool is_inside(obb_point_t point, obb_point_t edge_start, obb_point_t edge_end) {
    return cross_product(edge_start, edge_end, point) >= 0.0F;
}

static obb_point_t intersection(obb_point_t line_start,
                                obb_point_t line_end,
                                obb_point_t edge_start,
                                obb_point_t edge_end) {
    obb_point_t line = {line_end.x - line_start.x, line_end.y - line_start.y};
    obb_point_t edge = {edge_end.x - edge_start.x, edge_end.y - edge_start.y};
    float denominator = line.x * edge.y - line.y * edge.x;

    if (fabsf(denominator) < PARALLEL_EPSILON) {
        return line_end;
    }

    float offset_x = edge_start.x - line_start.x;
    float offset_y = edge_start.y - line_start.y;
    float distance = (offset_x * edge.y - offset_y * edge.x) / denominator;
    obb_point_t point = {line_start.x + distance * line.x, line_start.y + distance * line.y};
    return point;
}

static size_t clip_polygon(const obb_point_t* subject,
                           size_t subject_count,
                           obb_point_t edge_start,
                           obb_point_t edge_end,
                           obb_point_t* output) {
    if (subject_count == 0) {
        return 0;
    }

    size_t output_count = 0;
    obb_point_t previous = subject[subject_count - 1];
    bool previous_inside = is_inside(previous, edge_start, edge_end);

    for (size_t i = 0; i < subject_count; i++) {
        obb_point_t current = subject[i];
        bool current_inside = is_inside(current, edge_start, edge_end);

        if (current_inside != previous_inside) {
            output[output_count++] = intersection(previous, current, edge_start, edge_end);
        }
        if (current_inside) {
            output[output_count++] = current;
        }

        previous        = current;
        previous_inside = current_inside;
    }

    return output_count;
}

static float polygon_area(const obb_point_t* points, size_t count) {
    if (count < 3) {
        return 0.0F;
    }

    float twice_area = 0.0F;
    for (size_t i = 0; i < count; i++) {
        size_t next = (i + 1) % count;
        twice_area += points[i].x * points[next].y - points[next].x * points[i].y;
    }
    return fabsf(twice_area) / 2.0F;
}

static float rotated_iou(const obb_detection_t* first, const obb_detection_t* second) {
    obb_point_t buffers[2][MAX_POLYGON_POINTS];
    for (size_t i = 0; i < 4; i++) {
        buffers[0][i] = first->corners[i];
    }

    size_t count = 4;
    int source   = 0;
    for (size_t edge = 0; edge < 4 && count > 0; edge++) {
        int destination = 1 - source;
        buffers[destination][0] = (obb_point_t){0.0F, 0.0F};
        count = clip_polygon(buffers[source],
                             count,
                             second->corners[edge],
                             second->corners[(edge + 1) % 4],
                             buffers[destination]);
        source = destination;
    }

    float intersection_area = polygon_area(buffers[source], count);
    float union_area = first->width * first->height + second->width * second->height -
                       intersection_area;
    return union_area > 0.0F ? intersection_area / union_area : 0.0F;
}

static int compare_confidence(const void* left, const void* right) {
    const obb_detection_t* first  = left;
    const obb_detection_t* second = right;
    if (first->confidence < second->confidence) {
        return 1;
    }
    if (first->confidence > second->confidence) {
        return -1;
    }
    return 0;
}

static float tensor_value(const float* tensor, int channel, int detection, int num_detections) {
    return tensor[(size_t)channel * (size_t)num_detections + (size_t)detection];
}

bool yolov8_obb_decode(const float* tensor,
                       const yolov8_obb_params_t* params,
                       float confidence_threshold,
                       float iou_threshold,
                       obb_detection_t* output,
                       size_t output_capacity,
                       size_t* output_count) {
    if (!tensor || !params || !output || !output_count || params->input_width <= 0 ||
        params->input_height <= 0 || params->num_classes <= 0 || params->num_detections <= 0 ||
        params->num_features != params->num_classes + 5) {
        return false;
    }

    *output_count = 0;
    obb_detection_t* candidates =
        calloc((size_t)params->num_detections, sizeof(obb_detection_t));
    if (!candidates) {
        return false;
    }

    size_t candidate_count = 0;
    for (int i = 0; i < params->num_detections; i++) {
        float confidence = -INFINITY;
        int label_index  = -1;
        for (int label = 0; label < params->num_classes; label++) {
            float score = tensor_value(tensor, 4 + label, i, params->num_detections);
            if (isfinite(score) && score > confidence) {
                confidence = score;
                label_index = label;
            }
        }
        if (label_index < 0 || confidence < confidence_threshold) {
            continue;
        }

        obb_detection_t detection = {
            .center_x = tensor_value(tensor, 0, i, params->num_detections),
            .center_y = tensor_value(tensor, 1, i, params->num_detections),
            .width    = tensor_value(tensor, 2, i, params->num_detections),
            .height   = tensor_value(tensor, 3, i, params->num_detections),
            .angle = tensor_value(tensor,
                                  params->num_features - 1,
                                  i,
                                  params->num_detections),
            .confidence  = confidence,
            .label_index = label_index,
        };
        if (!isfinite(detection.center_x) || !isfinite(detection.center_y) ||
            !isfinite(detection.width) || !isfinite(detection.height) ||
            !isfinite(detection.angle) || detection.width <= 0.0F || detection.height <= 0.0F) {
            continue;
        }

        calculate_corners(&detection);
        candidates[candidate_count++] = detection;
    }

    qsort(candidates, candidate_count, sizeof(obb_detection_t), compare_confidence);
    bool* suppressed = calloc(candidate_count, sizeof(bool));
    if (!suppressed && candidate_count > 0) {
        free(candidates);
        return false;
    }

    for (size_t i = 0; i < candidate_count; i++) {
        if (suppressed[i]) {
            continue;
        }
        if (*output_count >= output_capacity) {
            break;
        }
        output[(*output_count)++] = candidates[i];

        for (size_t j = i + 1; j < candidate_count; j++) {
            if (!suppressed[j] && candidates[i].label_index == candidates[j].label_index &&
                rotated_iou(&candidates[i], &candidates[j]) > iou_threshold) {
                suppressed[j] = true;
            }
        }
    }

    free(suppressed);
    free(candidates);
    return true;
}




