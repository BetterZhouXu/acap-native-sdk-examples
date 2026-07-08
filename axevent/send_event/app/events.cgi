#!/bin/sh
# events.cgi — atomic drain-and-clear of the ACAP's local event queue.
#
# Behavior:
#   GET  -> stream all queued events (JSON Lines) and clear the queue.
#           If the queue is empty, returns 204 No Content with no body.
#           Concurrent writers are safe: we hold LOCK_EX around a rename()
#           so the ACAP's next write goes to a fresh empty file, and no
#           in-flight line is ever truncated.
#   HEAD -> report Content-Length of the current queue without draining.
#
# Exposed by manifest.json as:
#   /local/send_event/events.cgi   (admin auth)
#
# Copy the file out with:
#   curl --anyauth -u root:pass \
#     http://<cam>/local/send_event/events.cgi -o events.jsonl

set -eu

DIR="/usr/local/packages/send_event/localdata/events"
QUEUE="$DIR/events.jsonl"
STAGE="$DIR/events.draining.jsonl.$$"

emit_headers() {
    # status line + headers, terminated by blank line
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

method="${REQUEST_METHOD:-GET}"

# Ensure the directory exists even if the ACAP hasn't written anything yet.
mkdir -p "$DIR" 2>/dev/null || true

case "$method" in
    HEAD)
        if [ -s "$QUEUE" ]; then
            size=$(wc -c <"$QUEUE" | tr -d ' ')
            emit_headers "200 OK" "$size"
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

# Nothing to hand out? Bail early without touching the file.
if [ ! -s "$QUEUE" ]; then
    emit_empty
    exit 0
fi

# Atomic drain: exclusive-lock on the current queue file so the writer
# in send_event can't append past the rename. flock -x waits up to 5 s.
# Then rename it out of the way and let the writer create a fresh file.
if ! flock -x -w 5 "$QUEUE" -c "mv '$QUEUE' '$STAGE'"; then
    printf 'Status: 503 Service Unavailable\r\n\r\n'
    exit 0
fi

size=$(wc -c <"$STAGE" | tr -d ' ')
emit_headers "200 OK" "$size"
cat "$STAGE"
rm -f "$STAGE"

