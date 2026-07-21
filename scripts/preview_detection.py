"""Interactive live preview of person detection (Stage 1.3).

Opens the configured video source and runs the real person detector on every
frame, drawing a box + confidence around each detected person, live, in a
window. This is detection only -- no persistent track IDs yet (that's
tracking, Stage 1.4). Every frame is detected independently, so a box may
flicker or its position may not obviously connect frame-to-frame.

Each detected person gets a different box color (see
webcam_tracker.visualization.assign_colors), assigned by left-to-right
screen position -- NOT a stable per-person identity, since that requires
tracking. Watch what happens to the colors when two people cross paths:
they'll swap, because position is all this coloring has to go on. That's the
exact problem persistent tracking exists to solve -- see
scripts\\preview_tracking.py for the tracked version, which colors by a
genuinely stable per-person track ID instead.

Run from the repo root:
    .venv\\Scripts\\python.exe scripts\\preview_detection.py

Expected output: a window titled "webcam_tracker detection preview" showing
your live camera feed with a colored box + confidence score around each
detected person, plus an FPS counter. Press 'q' or Esc to close.

Common errors: same as scripts\\preview_webcam.py (see that file's
docstring) for camera-related issues. If the window is very laggy, GPU
inference may be falling back to CPU -- check the "Loading detector model"
log line at startup for which device it picked.
"""

from __future__ import annotations

import cv2

from webcam_tracker.config import load_config
from webcam_tracker.detection import create_detector
from webcam_tracker.logging_utils import configure_logging, get_logger
from webcam_tracker.perf_monitor import PerfMonitor
from webcam_tracker.video_input import VideoSourceError, create_source
from webcam_tracker.visualization import draw_detections, draw_perf_overlay

WINDOW_NAME = "webcam_tracker detection preview"


def main() -> None:
    config = load_config()
    configure_logging(level=config.logging.level, json_format=False)
    logger = get_logger(__name__)

    detector = create_detector(config)
    detector.load()

    source = create_source(config)
    perf = PerfMonitor(fps_window_seconds=config.perf_monitor.fps_window_seconds)

    try:
        with source:  # __enter__ calls source.open() -- do not call open() separately
            for frame in source:
                perf.start_frame()
                with perf.measure("detection"):
                    detections = detector.detect(frame.image)
                perf.end_frame()
                perf.log_if_due(config.perf_monitor.log_interval_seconds)

                image = frame.image.copy()
                draw_detections(image, detections)
                draw_perf_overlay(image, perf.snapshot(), extra_text=f"people: {len(detections)}")
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
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
