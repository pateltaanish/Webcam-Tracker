"""Interactive live preview of person detection (Stage 1.3).

Opens the configured video source and runs the real person detector on every
frame, drawing a box + confidence around each detected person, live, in a
window. This is detection only -- no persistent track IDs yet (that's
tracking, Stage 1.4, not built). Every frame is detected independently, so a
box may flicker or its position may not obviously connect frame-to-frame.

Run from the repo root:
    .venv\\Scripts\\python.exe scripts\\preview_detection.py

Expected output: a window titled "webcam_tracker detection preview" showing
your live camera feed with a green box + confidence score around each
detected person, plus an FPS counter. Press 'q' or Esc to close.

Common errors: same as scripts\\preview_webcam.py (see that file's
docstring) for camera-related issues. If the window is very laggy, GPU
inference may be falling back to CPU -- check the "Loading detector model"
log line at startup for which device it picked.
"""

from __future__ import annotations

import time

import cv2

from webcam_tracker.config import load_config
from webcam_tracker.detection import create_detector
from webcam_tracker.logging_utils import configure_logging, get_logger
from webcam_tracker.video_input import VideoSourceError, create_source

WINDOW_NAME = "webcam_tracker detection preview"


def main() -> None:
    config = load_config()
    configure_logging(level=config.logging.level, json_format=False)
    logger = get_logger(__name__)

    detector = create_detector(config)
    detector.load()

    source = create_source(config)

    fps_window_start = time.monotonic()
    frames_in_window = 0
    display_fps = 0.0

    try:
        with source:  # __enter__ calls source.open() -- do not call open() separately
            for frame in source:
                detections = detector.detect(frame.image)

                frames_in_window += 1
                elapsed = time.monotonic() - fps_window_start
                if elapsed >= 0.5:
                    display_fps = frames_in_window / elapsed
                    frames_in_window = 0
                    fps_window_start = time.monotonic()

                image = frame.image.copy()
                for detection in detections:
                    x1, y1, x2, y2 = (
                        int(v) for v in (detection.x1, detection.y1, detection.x2, detection.y2)
                    )
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"person {detection.confidence:.2f}"
                    cv2.putText(
                        image,
                        label,
                        (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )

                cv2.putText(
                    image,
                    f"FPS: {display_fps:.1f}  people: {len(detections)}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )
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
