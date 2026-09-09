#include "detection_result.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>

static void assert_claim(const char* expected) {
    char* json    = NULL;
    size_t length = 0;
    assert(detection_result_claim(&json, &length) == 1);
    assert(length == strlen(expected));
    assert(strcmp(json, expected) == 0);
    free(json);
}

int main(void) {
    const char* first  = "{\"event\":1}";
    const char* second = "{\"event\":2}";
    char* json         = NULL;
    size_t length      = 0;

    assert(detection_result_init());
    assert(detection_result_claim(&json, &length) == 0);

    assert(detection_result_publish(first, strlen(first)));
    assert(detection_result_publish(second, strlen(second)));
    assert_claim(first);

    // A new result can become pending while the first result is claimed.
    assert(detection_result_publish(second, strlen(second)));
    assert(detection_result_release());

    // A failed response retries the claimed result before the newer pending result.
    assert_claim(first);
    assert(detection_result_consume());
    assert_claim(second);
    assert(detection_result_consume());
    assert(detection_result_claim(&json, &length) == 0);
    assert(detection_result_init());
    return 0;
}

