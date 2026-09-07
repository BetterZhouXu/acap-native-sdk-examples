/* SPDX-License-Identifier: Apache-2.0 */
#include "obb.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

void obb_corners(obb_detection_t* d) {
    const float c = cosf(d->angle), s = sinf(d->angle);
    const float xs[4] = {-0.5f, 0.5f, 0.5f, -0.5f};
    const float ys[4] = {-0.5f, -0.5f, 0.5f, 0.5f};
    for (int i = 0; i < 4; ++i) {
        float x = xs[i] * d->width, y = ys[i] * d->height;
        d->corners[i] = (obb_point_t){d->cx + c * x - s * y, d->cy + s * x + c * y};
    }
}

static double side(obb_point_t a, obb_point_t b, obb_point_t p) {
    return ((double)b.x - a.x) * ((double)p.y - a.y) -
           ((double)b.y - a.y) * ((double)p.x - a.x);
}

float obb_iou(const obb_detection_t* a, const obb_detection_t* b) {
    /* Clip one convex rectangle against the four directed edges of the other. */
    obb_point_t polygon[16], clipped[16];
    memcpy(polygon, a->corners, sizeof(a->corners));
    int count = 4;
    for (int edge = 0; edge < 4 && count; ++edge) {
        obb_point_t start = b->corners[edge], end = b->corners[(edge + 1) % 4];
        int next_count = 0;
        obb_point_t previous = polygon[count - 1];
        double previous_side = side(start, end, previous);
        for (int i = 0; i < count; ++i) {
            obb_point_t current = polygon[i];
            double current_side = side(start, end, current);
            if ((current_side >= 0) != (previous_side >= 0)) {
                double t = previous_side / (previous_side - current_side);
                clipped[next_count++] = (obb_point_t){
                    previous.x + t * (current.x - previous.x),
                    previous.y + t * (current.y - previous.y)};
            }
            if (current_side >= 0) clipped[next_count++] = current;
            previous = current;
            previous_side = current_side;
        }
        count = next_count;
        memcpy(polygon, clipped, (size_t)count * sizeof(*polygon));
    }
    double area = 0;
    for (int i = 1; i + 1 < count; ++i)
        area += side(polygon[0], polygon[i], polygon[i + 1]);
    double intersection = fabs(area) * 0.5;
    double area_a = (double)a->width * a->height;
    double area_b = (double)b->width * b->height;
    intersection = fmin(intersection, fmin(area_a, area_b));
    double total = area_a + area_b - intersection;
    return total > 0 ? (float)(intersection / total) : 0;
}

static float value(const void* data, const obb_config_t* config, int channel, size_t i) {
    size_t index = (size_t)channel * config->candidates + i;
    if (config->dtype == OBB_FLOAT32) {
        float result;
        memcpy(&result, (const uint8_t*)data + index * sizeof(float), sizeof(result));
        return result;
    }
    int raw = config->dtype == OBB_INT8 ? ((const int8_t*)data)[index] :
                                        ((const uint8_t*)data)[index];
    return (raw - config->zero_point) * config->scale;
}

static int score_order(const void* lhs, const void* rhs) {
    const obb_detection_t* a = lhs;
    const obb_detection_t* b = rhs;
    if (a->confidence > b->confidence) return -1;
    if (a->confidence < b->confidence) return 1;
    return (a->candidate_id > b->candidate_id) - (a->candidate_id < b->candidate_id);
}

int obb_decode(const void* data, size_t bytes, const obb_config_t* c,
               float confidence_threshold, float iou_threshold,
               obb_detection_t output[OBB_MAX_DETECTIONS]) {
    if (!data || !c || !output || c->candidates != 18900 || c->classes != 4 ||
        c->input_width <= 0 || c->input_height <= 0 ||
        (unsigned int)c->dtype > OBB_INT8 ||
        !isfinite(confidence_threshold) || confidence_threshold < 0 || confidence_threshold > 1 ||
        !isfinite(iou_threshold) || iou_threshold < 0 || iou_threshold > 1 ||
        (c->dtype != OBB_FLOAT32 && (!isfinite(c->scale) || c->scale <= 0))) return -1;
    size_t element_size = c->dtype == OBB_FLOAT32 ? sizeof(float) : 1;
    if (bytes != c->candidates * (size_t)(c->classes + 5) * element_size) return -1;
    obb_detection_t* candidates = calloc(c->candidates, sizeof(*candidates));
    if (!candidates) return -1;
    size_t count = 0;
    for (size_t i = 0; i < c->candidates; ++i) {
        obb_detection_t d = {.candidate_id = (int)i, .class_id = -1};
        for (int cls = 0; cls < c->classes; ++cls) {
            float score = value(data, c, 4 + cls, i);
            if (isfinite(score) && score > d.confidence) {
                d.confidence = score;
                d.class_id = cls;
            }
        }
        if (d.class_id < 0 || d.confidence < confidence_threshold) continue;
        d.confidence = fminf(d.confidence, 1.0f);
        d.cx = value(data, c, 0, i);
        d.cy = value(data, c, 1, i);
        d.width = value(data, c, 2, i);
        d.height = value(data, c, 3, i);
        d.angle = value(data, c, 8, i);
        if (c->normalized) {
            d.cx *= c->input_width;
            d.width *= c->input_width;
            d.cy *= c->input_height;
            d.height *= c->input_height;
        }
        if (!isfinite(d.cx) || !isfinite(d.cy) || !isfinite(d.width) ||
            !isfinite(d.height) || !isfinite(d.angle) || d.width <= 0 || d.height <= 0 ||
            fabsf(d.cx) > 4.0f * c->input_width || fabsf(d.cy) > 4.0f * c->input_height ||
            d.width > 4.0f * c->input_width || d.height > 4.0f * c->input_height) continue;
        obb_corners(&d);
        candidates[count++] = d;
    }
    qsort(candidates, count, sizeof(*candidates), score_order);
    if (count > OBB_MAX_CANDIDATES) count = OBB_MAX_CANDIDATES;
    int kept = 0;
    for (size_t i = 0; i < count && kept < OBB_MAX_DETECTIONS; ++i) {
        bool suppressed = false;
        for (int j = 0; j < kept; ++j) {
            if (candidates[i].class_id == output[j].class_id &&
                obb_iou(&candidates[i], &output[j]) > iou_threshold) {
                suppressed = true;
                break;
            }
        }
        if (!suppressed) output[kept++] = candidates[i];
    }
    free(candidates);
    return kept;
}

