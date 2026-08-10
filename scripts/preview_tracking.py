"""Interactive live preview of person tracking (Stage 1.4).

Opens the configured video source and runs detection + tracking on every
frame, drawing a box + persistent track ID around each tracked person, live,
in a window. Unlike scripts\\preview_detection.py, box colors here are keyed
by track_id (webcam_tracker.visualization.color_for_track_id), so the SAME
person keeps the SAME color across frames -- watch what happens when two
people cross paths or one is briefly occluded, that's the real test of
whether tracking is working.

A new person takes a couple of frames to appear (tracks are only reported
once "confirmed" -- see webcam_tracker.tracking.TrackedPerson's docstring),
so don't be surprised by a brief delay when someone first enters frame.

Run from the repo root:
    .venv\\Scripts\\python.exe scripts\\preview_tracking.py

Expected output: a window titled "webcam_tracker tracking preview" showing
your live camera feed with a colored box + "ID <n> <confidence>" label per
tracked person, plus FPS. Press 'q' or Esc to close.

Set WEBCAM_TRACKER_VIDEO__RECORD=true (or video.record: true in
configs/default.yaml) to also save the annotated feed to
data\\recordings\\preview_tracking_<timestamp>.avi -- useful for reviewing a
trial run afterward on hardware with no attached display, e.g. a Pi in the
field. The saved path is logged when the recording starts/stops.

Set WEBCAM_TRACKER_STREAMING__ENABLED=true (or streaming.enabled: true) to
serve the same annotated feed live over HTTP instead of opening a local
window -- open http://<device-ip>:<port>/ (port from streaming.port,
default 8080) in a browser on another device. This is the headless path:
no DISPLAY/XAUTHORITY needed, and the only way to stop the script is
Ctrl+C, since there's no window to press 'q' in. GET /stream is the raw
MJPEG feed (openable directly in VLC or an <img> tag); GET /events is a
Server-Sent stream of hazard alerts as they fire.

Common errors: same as scripts\\preview_webcam.py for camera issues.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict

import cv2

from webcam_tracker.config import load_config
from webcam_tracker.detection import create_detector
from webcam_tracker.hazard import HazardMonitor
from webcam_tracker.logging_utils import configure_logging, get_logger
from webcam_tracker.perf_monitor import PerfMonitor
from webcam_tracker.streaming import LatestBroadcast, start_server
from webcam_tracker.tracking import create_tracker
from webcam_tracker.video_input import VideoSourceError, create_source
from webcam_tracker.visualization import FrameRecorder, draw_perf_overlay, draw_tracked_people

WINDOW_NAME = "webcam_tracker tracking preview"


def main() -> None:
    config = load_config()
    configure_logging(level=config.logging.level, json_format=False)
    logger = get_logger(__name__)

    detector = create_detector(config)
    detector.load()
    tracker = create_tracker(config)

    source = create_source(config)
    perf = PerfMonitor(fps_window_seconds=config.perf_monitor.fps_window_seconds)

    recorder = None
    if config.video.record:
        recordings_dir = config.resolve_path(config.paths.data_dir) / "recordings"
        output_path = recordings_dir / f"preview_tracking_{time.strftime('%Y%m%d_%H%M%S')}.avi"
        recorder = FrameRecorder(output_path, fps=config.video.requested_fps)

    frame_broadcast = None
    event_broadcast = None
    hazard_monitor = None
    http_server = None
    if config.streaming.enabled:
        frame_broadcast = LatestBroadcast()
        event_broadcast = LatestBroadcast()
        hazard_monitor = HazardMonitor(config.hazard)
        http_server = start_server(
            config.streaming.host, config.streaming.port, frame_broadcast, event_broadcast
        )
        logger.info(
            "Streaming enabled -- local preview window disabled, Ctrl+C to stop",
            extra={"url": f"http://{config.streaming.host}:{config.streaming.port}/"},
        )

    try:
        with source:  # __enter__ calls source.open() -- do not call open() separately
            for frame in source:
                perf.start_frame()
                with perf.measure("detection"):
                    detections = detector.detect(frame.image)
                with perf.measure("tracking"):
                    tracked_people = tracker.update(detections)
                perf.end_frame()
                perf.log_if_due(config.perf_monitor.log_interval_seconds)

                image = frame.image.copy()
                draw_tracked_people(image, tracked_people)
                draw_perf_overlay(
                    image, perf.snapshot(), extra_text=f"tracked: {len(tracked_people)}"
                )
                if recorder is not None:
                    recorder.write(image)

                if frame_broadcast is not None:
                    ok, buf = cv2.imencode(
                        ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, config.streaming.jpeg_quality]
                    )
                    if ok:
                        frame_broadcast.publish(buf.tobytes())
                if hazard_monitor is not None and event_broadcast is not None:
                    height, width = frame.image.shape[:2]
                    hazard = hazard_monitor.update(tracked_people, width, height)
                    if hazard is not None:
                        event_broadcast.publish(json.dumps(asdict(hazard)).encode("utf-8"))

                if config.streaming.enabled:
                    continue  # headless: no window, no key to wait on

                cv2.imshow(WINDOW_NAME, image)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):  # 'q' or Esc
                    break
    except VideoSourceError as exc:
        logger.error("Could not open video source", extra={"error": str(exc)})
        raise SystemExit(1) from exc
    except KeyboardInterrupt:
        pass
    finally:
        if recorder is not None:
            recorder.close()
        if http_server is not None:
            http_server.shutdown()
            http_server.server_close()
        if not config.streaming.enabled:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
