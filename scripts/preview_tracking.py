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

Common errors: same as scripts\\preview_webcam.py for camera issues.
"""

from __future__ import annotations

import time

import cv2

from webcam_tracker.config import load_config
from webcam_tracker.detection import create_detector
from webcam_tracker.logging_utils import configure_logging, get_logger
from webcam_tracker.perf_monitor import PerfMonitor
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
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
