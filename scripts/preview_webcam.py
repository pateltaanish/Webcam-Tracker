"""Interactive visual smoke test for video_input.

Opens the video source configured in configs/default.yaml (or overridden via
WEBCAM_TRACKER_VIDEO__SOURCE) and displays it live in a window with a
frame-rate overlay. This is the "does capture actually work on my machine"
check to run *before* trusting anything built on top of video_input --
run it on both your desktop (USB webcam) and, later, your laptop (built-in
webcam) to confirm `source: "auto"` picks the right camera on each.

Run from the repo root:
    .venv\\Scripts\\python.exe scripts\\preview_webcam.py

Expected output: a window titled "webcam_tracker preview" showing your live
camera feed with an FPS counter in the top-left corner. Press 'q' or Esc (or
Ctrl+C in the terminal) to close it.

Common errors:
  - VideoSourceError "No working webcam found": no camera detected. Run
    `scripts\\list_cameras.py` for details.
  - The window opens but stays black/frozen: another application (a
    video-call app, etc.) may be holding the camera exclusively -- close it
    and retry.
  - No window appears at all: this script requires a graphical desktop
    session; it will not work over a headless/SSH-only connection.
"""

from __future__ import annotations

import cv2

from webcam_tracker.config import load_config
from webcam_tracker.logging_utils import configure_logging, get_logger
from webcam_tracker.perf_monitor import PerfMonitor
from webcam_tracker.video_input import VideoSourceError, create_source
from webcam_tracker.visualization import draw_perf_overlay

WINDOW_NAME = "webcam_tracker preview"


def main() -> None:
    config = load_config()
    configure_logging(level=config.logging.level, json_format=False)
    logger = get_logger(__name__)

    source = create_source(config)
    perf = PerfMonitor(fps_window_seconds=config.perf_monitor.fps_window_seconds)

    try:
        with source:  # __enter__ calls source.open() -- do not call open() separately
            for frame in source:
                perf.start_frame()
                perf.end_frame()
                perf.log_if_due(config.perf_monitor.log_interval_seconds)

                image = frame.image.copy()
                draw_perf_overlay(image, perf.snapshot())
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
