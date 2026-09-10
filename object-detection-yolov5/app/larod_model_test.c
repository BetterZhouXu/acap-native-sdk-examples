/**
 * Copyright (C) 2026 Axis Communications AB, Lund, Sweden
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 */

#include <larod.h>

#include <fcntl.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

static void print_error(const char* operation, const larodError* error) {
    if (error) {
        fprintf(stderr, "%s failed: %s (code %d)\n", operation, error->msg, error->code);
    } else {
        fprintf(stderr, "%s failed without Larod error details\n", operation);
    }
}

static bool clear_input_tensor(larodTensor* tensor, int* input_fd, larodError** error) {
    larodTensorDataType datatype = larodGetTensorDataType(tensor, error);
    if (datatype != LAROD_TENSOR_DATA_TYPE_FLOAT32) {
        fprintf(stderr, "Expected float32 input, got Larod datatype %d\n", (int)datatype);
        return false;
    }

    *input_fd = larodGetTensorFd(tensor, error);
    size_t size = 0;
    if (*input_fd == LAROD_INVALID_FD || !larodGetTensorFdSize(tensor, &size, error)) {
        print_error("get input tensor buffer", *error);
        return false;
    }

    void* data = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, *input_fd, 0);
    if (data == MAP_FAILED) {
        perror("mmap input tensor");
        return false;
    }
    memset(data, 0, size);
    bool success = true;
    if (msync(data, size, MS_SYNC) != 0) {
        perror("msync input tensor");
        success = false;
    }
    if (munmap(data, size) != 0) {
        perror("munmap input tensor");
        success = false;
    }
    return success;
}

static bool print_tensor(const char* kind,
                         size_t index,
                         larodTensor* tensor,
                         larodError** error) {
    const larodTensorDims* dims = larodGetTensorDims(tensor, error);
    if (!dims) {
        print_error("larodGetTensorDims", *error);
        return false;
    }

    larodTensorDataType datatype = larodGetTensorDataType(tensor, error);
    size_t size = 0;
    if (datatype == LAROD_TENSOR_DATA_TYPE_INVALID ||
        !larodGetTensorFdSize(tensor, &size, error)) {
        print_error("read tensor metadata", *error);
        return false;
    }

    printf("%s[%zu]: datatype=%d shape=[", kind, index, (int)datatype);
    for (size_t i = 0; i < dims->len; i++) {
        printf("%s%zu", i == 0 ? "" : ", ", dims->dims[i]);
    }
    printf("] bytes=%zu\n", size);
    return true;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s MODEL_FILE LAROD_DEVICE\n", argv[0]);
        return EXIT_FAILURE;
    }

    larodConnection* connection = NULL;
    larodModel* model = NULL;
    larodTensor** inputs = NULL;
    larodTensor** outputs = NULL;
    larodJobRequest* request = NULL;
    larodError* error = NULL;
    size_t input_count = 0;
    size_t output_count = 0;
    int model_fd = -1;
    int input_fd = -1;
    int result = EXIT_FAILURE;

    if (!larodConnect(&connection, &error)) {
        print_error("larodConnect", error);
        goto cleanup;
    }

    const larodDevice* device = larodGetDevice(connection, argv[2], 0, &error);
    if (!device) {
        print_error("larodGetDevice", error);
        goto cleanup;
    }

    model_fd = open(argv[1], O_RDONLY);
    if (model_fd < 0) {
        perror("open model");
        goto cleanup;
    }

    printf("Loading %s on %s...\n", argv[1], argv[2]);
    model = larodLoadModel(connection,
                           model_fd,
                           device,
                           LAROD_ACCESS_PRIVATE,
                           "larod_model_test",
                           NULL,
                           &error);
    if (!model) {
        print_error("larodLoadModel", error);
        goto cleanup;
    }
    puts("Model load succeeded.");

    inputs = larodAllocModelInputs(connection, model, 0, &input_count, NULL, &error);
    outputs = larodAllocModelOutputs(connection, model, 0, &output_count, NULL, &error);
    if (!inputs || !outputs || input_count != 1 || output_count != 1) {
        print_error("allocate model tensors", error);
        goto cleanup;
    }
    if (!print_tensor("input", 0, inputs[0], &error) ||
        !print_tensor("output", 0, outputs[0], &error) ||
        !clear_input_tensor(inputs[0], &input_fd, &error)) {
        goto cleanup;
    }

    request = larodCreateJobRequest(model,
                                    inputs,
                                    input_count,
                                    outputs,
                                    output_count,
                                    NULL,
                                    &error);
    if (!request) {
        print_error("larodCreateJobRequest", error);
        goto cleanup;
    }

    puts("Running one zero-input inference...");
    if (!larodRunJob(connection, request, &error)) {
        print_error("larodRunJob", error);
        goto cleanup;
    }
    puts("Inference succeeded.");
    result = EXIT_SUCCESS;

cleanup:
    larodDestroyJobRequest(&request);
    if (input_fd >= 0) {
        close(input_fd);
    }
    if (inputs) {
        larodDestroyTensors(connection, &inputs, input_count, NULL);
    }
    if (outputs) {
        larodDestroyTensors(connection, &outputs, output_count, NULL);
    }
    larodDestroyModel(&model);
    if (connection) {
        larodDisconnect(&connection, NULL);
    }
    if (error) {
        larodClearError(&error);
    }
    if (model_fd >= 0) {
        close(model_fd);
    }
    return result;
}



