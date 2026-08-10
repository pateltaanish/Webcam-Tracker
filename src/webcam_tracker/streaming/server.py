"""Live MJPEG video + Server-Sent hazard events over plain HTTP, stdlib only.

Runs as a thread inside the process that owns the camera. It has to: only
one process can hold the Pi's CSI camera at a time (see
watchdog-vision.service), so this can't be a sibling process reading a
shared file the way Watchdog's Convex relay did -- that file-plus-poll hop
was most of Watchdog's latency (500ms write cadence + up to 200ms poll,
before the upload itself). Publishing straight into an in-memory slot from
the same loop that already draws the overlay removes both hops.

No new dependency: http.server + threading is enough for one MJPEG stream
and one SSE stream, and pulling in aiohttp/FastAPI for two GET routes would
be the over-engineered choice, not the safe one.
"""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_MJPEG_BOUNDARY = b"frame"

_DASHBOARD_HTML = b"""<!doctype html>
<title>webcam_tracker live</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  body{background:#111;color:#eee;font:14px/1.4 system-ui,sans-serif;margin:0;padding:1rem}
  h1{font-size:1rem;font-weight:600;margin:0 0 .75rem}
  img{max-width:100%;border-radius:8px;display:block;background:#000}
  #log{font-size:.8rem;opacity:.85;white-space:pre-wrap;margin-top:.75rem;
       max-height:10rem;overflow-y:auto}
</style>
<h1>webcam_tracker &mdash; live</h1>
<img src="/stream" alt="live camera feed">
<div id="log">(hazard events appear here)</div>
<script>
  const log = document.getElementById("log");
  new EventSource("/events").onmessage = (e) => {
    const h = JSON.parse(e.data);
    const ttc = h.ttc === null ? "static" : `ttc=${h.ttc.toFixed(2)}s`;
    const line = `${new Date().toLocaleTimeString()}  track ${h.track_id}  ` +
      `${h.zone}  ${ttc}${h.urgent ? "  URGENT" : ""}`;
    log.textContent = line + "\\n" + log.textContent;
  };
</script>
"""


class LatestBroadcast:
    """Single newest-value slot, fanned out to any number of readers.

    Deliberately not a queue. A queue lets a slow viewer accumulate backlog,
    and draining that backlog would eventually apply backpressure to the
    publisher -- which here is the tracking loop, and it must never block on
    how fast a browser tab happens to be. A slow reader just misses whatever
    was published while it wasn't looking and picks up the current value.

    wait_for_next blocks the CALLER's thread (one per HTTP connection) until
    something newer than what it already has is published, so delivery is
    push-based -- no polling loop guessing an interval, and no extra latency
    beyond however long publish() takes to run.
    """

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._value: bytes | None = None
        self._version = 0

    def publish(self, value: bytes) -> None:
        with self._condition:
            self._value = value
            self._version += 1
            self._condition.notify_all()

    def wait_for_next(self, last_version: int, timeout: float) -> tuple[int, bytes] | None:
        """Blocks for a version newer than last_version, or until timeout.

        A version of 0 (a caller that hasn't seen anything yet) returns
        immediately if a value has already been published -- the predicate
        is checked before blocking, so a value published before this call
        is not missed just because nobody was waiting yet.

        Returns None on timeout, used by callers as a heartbeat interval to
        notice a connection has gone stale (the client disconnected but the
        write that would reveal it hasn't happened yet) without polling.
        """
        with self._condition:
            got = self._condition.wait_for(
                lambda: self._version != last_version, timeout=timeout
            )
            if not got:
                return None
            assert self._value is not None  # version only advances alongside a value
            return self._version, self._value


def _make_handler(frames: LatestBroadcast, events: LatestBroadcast) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        # Default protocol_version ("HTTP/1.0"): /stream and /events never send
        # a Content-Length (the whole point is an unbounded live feed), and
        # HTTP/1.0's close-terminates-the-body semantics are what every MJPEG
        # server since mjpg-streamer relies on. HTTP/1.1 keep-alive would make
        # that framing ambiguous for no benefit here.

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            pass  # the tracking loop's own structured logger covers this run

        def do_GET(self) -> None:
            if self.path == "/stream":
                self._stream_mjpeg()
            elif self.path == "/events":
                self._stream_events()
            elif self.path in ("/", "/index.html"):
                self._serve_dashboard()
            else:
                self.send_error(404)

        def do_HEAD(self) -> None:
            # Headers only, no body, no entering the streaming loops -- a
            # health check hitting /stream or /events with HEAD (some
            # dashboard tools probe this way before adding a panel) would
            # otherwise get BaseHTTPRequestHandler's default 501 and be
            # unable to tell the endpoint exists at all.
            if self.path == "/stream":
                boundary = _MJPEG_BOUNDARY.decode()
                self.send_response(200)
                self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={boundary}")
                self.end_headers()
            elif self.path == "/events":
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
            elif self.path in ("/", "/index.html"):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(_DASHBOARD_HTML)))
                self.end_headers()
            else:
                self.send_error(404)

        def _stream_mjpeg(self) -> None:
            try:
                self.send_response(200)
                boundary = _MJPEG_BOUNDARY.decode()
                self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={boundary}")
                self.send_header("Cache-Control", "no-cache, private")
                self.end_headers()
                last = 0
                while True:
                    got = frames.wait_for_next(last, timeout=5.0)
                    if got is None:
                        continue  # camera stalled, not a disconnect -- keep waiting
                    last, jpeg = got
                    self.wfile.write(b"--" + _MJPEG_BOUNDARY + b"\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii"))
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass  # viewer closed the tab -- the end of this connection, not an error

        def _stream_events(self) -> None:
            try:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                last = 0
                while True:
                    got = events.wait_for_next(last, timeout=15.0)
                    if got is None:
                        # SSE comment line: ignored by EventSource, just keeps
                        # the connection alive through idle/proxy timeouts --
                        # hazards are rate-limited by HazardMonitor's own
                        # cooldowns and may be genuinely quiet for minutes.
                        self.wfile.write(b": keep-alive\n\n")
                        continue
                    last, payload = got
                    self.wfile.write(b"data: " + payload + b"\n\n")
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass

        def _serve_dashboard(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(_DASHBOARD_HTML)))
            self.end_headers()
            self.wfile.write(_DASHBOARD_HTML)

    return Handler


def start_server(
    host: str, port: int, frames: LatestBroadcast, events: LatestBroadcast
) -> ThreadingHTTPServer:
    """Starts the HTTP server on a daemon thread and returns it immediately.

    ThreadingHTTPServer hands each connection its own thread, so multiple
    /stream and /events viewers are independent -- one slow viewer's socket
    never blocks another's, and neither blocks the caller (the tracking
    loop, via LatestBroadcast.publish).

    Caller owns shutdown: call server.shutdown() then server.server_close()
    when done, or the port stays bound until the process exits.
    """
    server = ThreadingHTTPServer((host, port), _make_handler(frames, events))
    threading.Thread(target=server.serve_forever, daemon=True, name="stream-http").start()
    return server
