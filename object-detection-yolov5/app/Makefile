PROG1	= $(shell jq -r '.acapPackageConf.setup.appName' manifest.json)
OBJS1	= object_detection_yolov5.c argparse.c detection_fastcgi.c detection_result.c imgprovider.c labelparse.c model.c panic.c yolov8_obb.c
PROG2	= larod_model_test
OBJS2	= larod_model_test.c
PROGS	= $(PROG1) $(PROG2)
DEBUG_DIR = debug
HOST_CC ?= cc
HOST_TEST_FLAGS = -std=c11 -Wall -Wextra -Werror -I.
OBB_TEST = /tmp/test_yolov8_obb
RESULT_TEST = /tmp/test_detection_result
RESULT_PATH_FLAG = -DDETECTION_RESULT_PATH='"/tmp/detection-result-test.json"'

PKGS = axparameter bbox fcgi gio-2.0 gio-unix-2.0 jansson liblarod vdostream

CFLAGS += $(shell PKG_CONFIG_PATH=$(PKG_CONFIG_PATH) pkg-config --cflags $(PKGS))
LDLIBS += $(shell PKG_CONFIG_PATH=$(PKG_CONFIG_PATH) pkg-config --libs $(PKGS))
LDLIBS += -lm

CFLAGS += -DLAROD_API_VERSION_3
CFLAGS += -pthread
LDLIBS += -pthread

CFLAGS += -Wall \
          -Wextra \
          -Wformat=2 \
          -Wpointer-arith \
          -Wbad-function-cast \
          -Wstrict-prototypes \
          -Wmissing-prototypes \
          -Winline \
          -Wdisabled-optimization \
          -Wfloat-equal \
          -W \
          -Werror

all:	$(PROGS)

.PHONY: test-host test-yolov8-obb test-detection-result
test-host: test-yolov8-obb test-detection-result ;

test-yolov8-obb: ; $(HOST_CC) $(HOST_TEST_FLAGS) tests/test_yolov8_obb.c yolov8_obb.c -lm -o $(OBB_TEST) && $(OBB_TEST)

test-detection-result: ; $(HOST_CC) $(HOST_TEST_FLAGS) $(RESULT_PATH_FLAG) tests/test_detection_result.c detection_result.c -o $(RESULT_TEST) && $(RESULT_TEST)

$(PROG1): $(OBJS1)
	install -d $(DEBUG_DIR)
	$(CC) $^ $(CFLAGS) $(LIBS) $(LDFLAGS) -lm $(LDLIBS) -o $(DEBUG_DIR)/$@
	cp $(DEBUG_DIR)/$@ .
	$(STRIP) $@

$(PROG2): $(OBJS2) ; install -d $(DEBUG_DIR) && $(CC) $^ $(CFLAGS) $(LIBS) $(LDFLAGS) $(LDLIBS) -o $(DEBUG_DIR)/$@ && cp $(DEBUG_DIR)/$@ . && $(STRIP) $@

clean:
	rm -rf $(PROGS) *.o *.eap* *_LICENSE.txt package.conf* param.conf tmp* manifest.json $(DEBUG_DIR)
