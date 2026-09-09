#include "yolov8_obb.h"

#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>

#define TEST_EPSILON 1.0e-5F
#define TEST_PI 3.14159265358979323846F

static bool nearly_equal(float left, float right) {
    return fabsf(left - right) < TEST_EPSILON;
}

static void test_decode_and_class_aware_nms(void) {
    const yolov8_obb_params_t params = {
        .input_width = 960,
        .input_height = 960,
        .num_classes = 2,
        .num_detections = 4,
        .num_features = 7,
    };
    // Channel-first: x, y, w, h, class0, class1, angle.
    const float tensor[] = {
        0.50F, 0.50F, 0.50F, 0.10F,
        0.50F, 0.50F, 0.50F, 0.10F,
        0.40F, 0.40F, 0.40F, 0.10F,
        0.20F, 0.20F, 0.20F, 0.10F,
        0.90F, 0.80F, 0.05F, 0.10F,
        0.05F, 0.10F, 0.85F, 0.20F,
        0.00F, 0.00F, 0.00F, 0.00F,
    };
    obb_detection_t output[4];
    size_t count = 0;

    assert(yolov8_obb_decode(tensor, &params, 0.25F, 0.50F, output, 4, &count));
    assert(count == 2);
    assert(output[0].label_index == 0);
    assert(nearly_equal(output[0].confidence, 0.90F));
    assert(nearly_equal(output[0].corners[0].x, 0.30F));
    assert(nearly_equal(output[0].corners[0].y, 0.40F));
    assert(nearly_equal(output[0].corners[2].x, 0.70F));
    assert(nearly_equal(output[0].corners[2].y, 0.60F));
    assert(output[1].label_index == 1);
    assert(nearly_equal(output[1].confidence, 0.85F));
}

static void test_rotation_and_capacity(void) {
    const yolov8_obb_params_t params = {
        .input_width = 960,
        .input_height = 960,
        .num_classes = 1,
        .num_detections = 2,
        .num_features = 6,
    };
    const float tensor[] = {
        0.50F, 0.20F,
        0.50F, 0.20F,
        0.40F, 0.10F,
        0.20F, 0.10F,
        0.90F, 0.80F,
        TEST_PI / 2.0F, 0.00F,
    };
    obb_detection_t output[1];
    size_t count = 0;

    assert(yolov8_obb_decode(tensor, &params, 0.25F, 0.50F, output, 1, &count));
    assert(count == 1);
    assert(nearly_equal(output[0].corners[0].x, 0.60F));
    assert(nearly_equal(output[0].corners[0].y, 0.30F));
    assert(nearly_equal(output[0].corners[2].x, 0.40F));
    assert(nearly_equal(output[0].corners[2].y, 0.70F));
}

static void test_invalid_parameters(void) {
    const yolov8_obb_params_t invalid = {
        .input_width = 960,
        .input_height = 960,
        .num_classes = 2,
        .num_detections = 1,
        .num_features = 8,
    };
    const float tensor[8] = {0};
    obb_detection_t output[1];
    size_t count = 0;

    assert(!yolov8_obb_decode(tensor, &invalid, 0.25F, 0.50F, output, 1, &count));
    assert(!yolov8_obb_decode(NULL, &invalid, 0.25F, 0.50F, output, 1, &count));
}

int main(void) {
    test_decode_and_class_aware_nms();
    test_rotation_and_capacity();
    test_invalid_parameters();
    puts("YOLOv8 OBB parser tests passed");
    return 0;
}

