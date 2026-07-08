PROG1	= $(shell jq -r '.acapPackageConf.setup.appName' manifest.json)
OBJS1	= $(PROG1).c

PROGS	= $(PROG1)
DEBUG_DIR = debug

PKGS = glib-2.0 axevent
CFLAGS += $(shell PKG_CONFIG_PATH=$(PKG_CONFIG_PATH) pkg-config --cflags $(PKGS))
LDLIBS += $(shell PKG_CONFIG_PATH=$(PKG_CONFIG_PATH) pkg-config --libs $(PKGS))

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

all:	$(PROGS) events.cgi

$(PROG1): $(OBJS1)
	install -d $(DEBUG_DIR)
	$(CC) $^ $(CFLAGS) $(LIBS) $(LDFLAGS) -lm $(LDLIBS) -o $(DEBUG_DIR)/$@
	cp $(DEBUG_DIR)/$@ .
	$(STRIP) $@

# Force the drain CGI to be POSIX-friendly and executable, and mark it as
# always-out-of-date so it runs on every build (chmod is idempotent).
# `.PHONY: events.cgi` prevents make from short-circuiting because a file
# with that name already exists.
.PHONY: events.cgi

# Ensure the drain CGI ships with the executable bit set and Unix line
# endings inside the .eap. Without this a Windows/macOS editor can insert
# CR chars and Apache will return HTTP 500 with "bad interpreter".
events.cgi:
	tr -d '\r' < events.cgi > events.cgi.tmp && mv events.cgi.tmp events.cgi
	chmod 0755 events.cgi

clean:
	rm -rf $(PROGS) *.o *.eap* *_LICENSE.txt package.conf* param.conf tmp* $(DEBUG_DIR)
