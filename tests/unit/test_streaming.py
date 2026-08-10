"""Tests for the MJPEG/SSE streaming server.

The server tests use real sockets against 127.0.0.1:0 (OS-assigned port)
rather than mocking http.server -- the thing worth verifying is that bytes
published on one thread actually arrive, correctly framed, on another
thread's socket, which a mock can't tell you.
"""

from __future__ import annotations

import http.client
import time

import pytest

from webcam_tracker.streaming import LatestBroadcast, start_server

# --- LatestBroadcast -------------------------------------------------------


def test_wait_for_next_returns_a_value_published_before_the_call() -> None:
    """A value published before anyone was waiting must not be missed --
    a new viewer connecting should get whatever is current, not block until
    the next frame after it happened to connect."""
    b = LatestBroadcast()
    b.publish(b"frame-one")
    got = b.wait_for_next(last_version=0, timeout=1.0)
    assert got is not None
    version, value = got
    assert value == b"frame-one"
    assert version != 0


def test_wait_for_next_times_out_with_nothing_published() -> None:
    b = LatestBroadcast()
    assert b.wait_for_next(last_version=0, timeout=0.05) is None


def test_a_second_call_with_the_returned_version_blocks_until_something_new() -> None:
    b = LatestBroadcast()
    b.publish(b"one")
    version, _ = b.wait_for_next(0, timeout=1.0)  # type: ignore[misc]
    assert b.wait_for_next(version, timeout=0.05) is None  # nothing newer yet
    b.publish(b"two")
    got = b.wait_for_next(version, timeout=1.0)
    assert got is not None and got[1] == b"two"


def test_rapid_publishes_collapse_to_the_newest_value() -> None:
    """No queue: a reader that wasn't looking during two quick publishes
    sees only the latest, never a backlog of both."""
    b = LatestBroadcast()
    b.publish(b"stale")
    b.publish(b"fresh")
    got = b.wait_for_next(last_version=0, timeout=1.0)
    assert got is not None
    assert got[1] == b"fresh"


def test_a_waiting_reader_is_woken_by_a_publish_from_another_thread() -> None:
    """The actual push-based-delivery guarantee: a reader blocked in
    wait_for_next before anything exists still gets it as soon as it's
    published, without polling."""
    import threading

    b = LatestBroadcast()
    result: list[tuple[int, bytes] | None] = [None]

    def reader() -> None:
        result[0] = b.wait_for_next(last_version=0, timeout=2.0)

    t = threading.Thread(target=reader)
    t.start()
    time.sleep(0.05)  # let the reader actually reach wait_for_next first
    b.publish(b"woke-you-up")
    t.join(timeout=2.0)
    assert result[0] is not None
    assert result[0][1] == b"woke-you-up"


# --- HTTP server -------------------------------------------------------


def _read_until(resp, needle: bytes, timeout: float = 2.0) -> bytes:  # type: ignore[no-untyped-def]
    """Accumulate read1() chunks until needle appears.

    The server's multipart write is several small wfile.write() calls, so a
    single recv on the client side may only capture the first one --
    read1() returns as soon as anything is available, not once needle-worth
    of bytes have arrived.
    """
    deadline = time.monotonic() + timeout
    buf = b""
    while needle not in buf and time.monotonic() < deadline:
        buf += resp.fp.read1(4096)
    return buf


@pytest.fixture
def server():  # type: ignore[no-untyped-def]
    frames = LatestBroadcast()
    events = LatestBroadcast()
    srv = start_server("127.0.0.1", 0, frames, events)
    port = srv.server_address[1]
    try:
        yield port, frames, events
    finally:
        srv.shutdown()
        srv.server_close()


def test_dashboard_route_serves_html_wiring_both_endpoints(server) -> None:  # type: ignore[no-untyped-def]
    port, _frames, _events = server
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
    conn.request("GET", "/")
    resp = conn.getresponse()
    body = resp.read()
    assert resp.status == 200
    assert resp.getheader("Content-Type", "").startswith("text/html")
    assert b'src="/stream"' in body
    assert b'"/events"' in body
    conn.close()


def test_head_on_every_route_gets_headers_and_no_body(server) -> None:  # type: ignore[no-untyped-def]
    """A dashboard tool health-checking with HEAD before adding a panel must
    see 200 with the right Content-Type, not BaseHTTPRequestHandler's default
    501 for a method nobody implemented."""
    port, _frames, _events = server
    routes = (("/", "text/html"), ("/stream", "multipart"), ("/events", "text/event-stream"))
    for path, content_type in routes:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
        conn.request("HEAD", path)
        resp = conn.getresponse()
        assert resp.status == 200, path
        assert content_type in resp.getheader("Content-Type", ""), path
        assert resp.read() == b""
        conn.close()


def test_unknown_path_is_404(server) -> None:  # type: ignore[no-untyped-def]
    port, _frames, _events = server
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
    conn.request("GET", "/nope")
    resp = conn.getresponse()
    assert resp.status == 404
    resp.read()
    conn.close()


def test_stream_delivers_a_published_frame_multipart_framed(server) -> None:  # type: ignore[no-untyped-def]
    port, frames, _events = server
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
    conn.request("GET", "/stream")
    resp = conn.getresponse()
    assert resp.status == 200
    assert "multipart/x-mixed-replace" in resp.getheader("Content-Type", "")

    jpeg = b"\xff\xd8fake-jpeg-bytes\xff\xd9"
    frames.publish(jpeg)

    # read exactly one multipart part: boundary + headers + body + trailer
    expected = (
        b"--frame\r\n"
        b"Content-Type: image/jpeg\r\n"
        + f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii")
        + jpeg
        + b"\r\n"
    )
    got = resp.read(len(expected))
    assert got == expected
    conn.close()


def test_a_viewer_connecting_mid_stream_gets_the_current_frame_immediately(server) -> None:  # type: ignore[no-untyped-def]
    """A new viewer sees whatever is currently live right away -- it must
    not sit on a blank image waiting for the NEXT capture just because the
    current one was published before it connected. This is why
    wait_for_next(0, ...) returns immediately for an already-published
    value instead of only ever waiting for something strictly new."""
    port, frames, _events = server
    frames.publish(b"already-live-frame")
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
    conn.request("GET", "/stream")
    resp = conn.getresponse()
    got = _read_until(resp, b"already-live-frame")
    assert b"already-live-frame" in got
    conn.close()


def test_events_delivers_a_published_payload_as_sse(server) -> None:  # type: ignore[no-untyped-def]
    port, _frames, events = server
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
    conn.request("GET", "/events")
    resp = conn.getresponse()
    assert resp.status == 200
    assert resp.getheader("Content-Type") == "text/event-stream"

    events.publish(b'{"track_id": 1, "zone": "left"}')
    got = resp.read(len(b'data: {"track_id": 1, "zone": "left"}\n\n'))
    assert got == b'data: {"track_id": 1, "zone": "left"}\n\n'
    conn.close()


def test_multiple_stream_viewers_each_get_the_same_frame(server) -> None:  # type: ignore[no-untyped-def]
    """Fan-out: two independent connections, one publish, both see it --
    proves LatestBroadcast isn't accidentally consumed by the first reader."""
    port, frames, _events = server
    conns = [http.client.HTTPConnection("127.0.0.1", port, timeout=2) for _ in range(2)]
    resps = []
    for conn in conns:
        conn.request("GET", "/stream")
        resps.append(conn.getresponse())

    jpeg = b"shared-frame"
    frames.publish(jpeg)
    for resp in resps:
        assert jpeg in _read_until(resp, jpeg)
    for conn in conns:
        conn.close()
