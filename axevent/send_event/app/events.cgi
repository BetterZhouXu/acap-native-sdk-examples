#!/bin/sh
# events.cgi — atomic drain-and-clear of the ACAP's local event queue.
#
# Behavior:
#   GET  -> stream all queued events (JSON Lines) and clear the queue.
#           Concurrent writers are safe: we use a portable directory-based
#           lock, then rename() the queue out of the way so the ACAP's next
#           open() picks up a fresh empty file. No `flock` binary required.
#   HEAD -> report Content-Length of the current queue without draining.
#
# Exposed by manifest.json as:
#   /local/send_event/events.cgi     (admin auth)
#
# Copy the file out with:
#   curl --anyauth -u root:pass \
#     http://<cam>/local/send_event/events.cgi -o events.jsonl

# NOTE: deliberately NO `set -e`. A CGI must always emit valid headers,
# even on partial failure, or Apache returns bare 500 to the client.

DIR="/usr/local/packages/send_event/localdata/events"
QUEUE="$DIR/events.jsonl"
LOCKDIR="$DIR/.drain.lock"
STAGE="$DIR/events.draining.jsonl.$$"

emit_status() {
    # Args: status_text [content_length]
    printf 'Status: %s\r\n' "$1"
    printf 'Content-Type: application/x-ndjson\r\n'
    printf 'Cache-Control: no-store\r\n'
    printf 'Connection: close\r\n'
    if [ -n "${2:-}" ]; then
        printf 'Content-Length: %s\r\n' "$2"
    fi
    printf '\r\n'
}

emit_empty() {
    printf 'Status: 204 No Content\r\n'
    printf 'Cache-Control: no-store\r\n'
    printf 'Connection: close\r\n'
    printf '\r\n'
}

emit_error() {
    # Args: status_text message
    printf 'Status: %s\r\n' "$1"
    printf 'Content-Type: text/plain\r\n'
    printf 'Cache-Control: no-store\r\n'
    printf 'Connection: close\r\n'
    printf '\r\n'
    printf '%s\n' "$2"
}

# Ensure the dir exists so the writer can create the file on first use.
mkdir -p "$DIR" 2>/dev/null

method="${REQUEST_METHOD:-GET}"

case "$method" in
    HEAD)
        if [ -s "$QUEUE" ]; then
            size=$(wc -c < "$QUEUE" 2>/dev/null | tr -d ' ')
            emit_status "200 OK" "$size"
        else
            emit_empty
        fi
        exit 0
        ;;
    GET|POST)
        : # fall through to drain
        ;;
    *)
        printf 'Status: 405 Method Not Allowed\r\nAllow: GET, HEAD\r\n\r\n'
        exit 0
        ;;
esac

# Nothing queued -> quick 204, don't touch the file at all.
if [ ! -s "$QUEUE" ]; then
    emit_empty
    exit 0
fi

# Portable lock: mkdir is atomic on every POSIX FS. No `flock` binary needed.
# Give a concurrent drainer up to ~5 s to finish.
i=0
while ! mkdir "$LOCKDIR" 2>/dev/null; do
    i=$((i + 1))
    if [ "$i" -ge 50 ]; then
        emit_error "503 Service Unavailable" "drain lock busy"
        exit 0
    fi
    sleep 0.1 2>/dev/null || sleep 1
done

# Release the lock and stage file even on unexpected exit.
trap 'rmdir "$LOCKDIR" 2>/dev/null; rm -f "$STAGE" 2>/dev/null' EXIT INT TERM

# Re-check under the lock (another drainer may have just cleared it).
if [ ! -s "$QUEUE" ]; then
    emit_empty
    exit 0
fi

# Atomic rename: the ACAP writer will O_CREAT a fresh empty file on its
# next append. Any line written *before* this instant is in $STAGE; any
# line written *after* goes to the new empty file.
if ! mv "$QUEUE" "$STAGE" 2>/dev/null; then
    emit_error "500 Internal Server Error" "rename failed"
    exit 0
fi

size=$(wc -c < "$STAGE" 2>/dev/null | tr -d ' ')
emit_status "200 OK" "$size"
cat "$STAGE"

