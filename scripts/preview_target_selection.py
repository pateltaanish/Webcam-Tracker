"""Interactive live preview of manual target selection (Stage 1.6).

Opens the configured video source and runs detection + tracking on every
frame, same as scripts\\preview_tracking.py, but adds manual target
selection on top: **left-click** a tracked person to select them as the
target, **right-click** (or press 'c') to clear the selection.

Once a target is selected, this exercises the rest of what Stage 1.6 adds:
the target's box is highlighted in a fixed color (not the per-track color),
a crosshair marks the frame center and the target's center, a line shows
the pixel offset between them, and the pixel error + normalized error
(fraction of half-frame-width/height, roughly -1..1) are printed on screen.
This is the raw error signal Stage 1.7's simulated gimbal controller will
consume -- nothing here moves a gimbal yet, it just computes and displays
the error.

If your target's track_id disappears (occlusion, leaves frame, or tracking
loses them), you'll see a red "TARGET LOST" warning instead -- recovering
from that is Stage 1.8, not handled here.

Run from the repo root:
    .venv\\Scripts\\python.exe scripts\\preview_target_selection.py

Expected output: a window titled "webcam_tracker target selection" showing
your live camera feed with tracked-person boxes; clicking one highlights it
as the target with error readout at the bottom of the frame. Press 'q' or
Esc to close.

Common errors: same as scripts\\preview_webcam.py for camera issues.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2

from webcam_tracker.config import load_config
from webcam_tracker.detection import create_detector
from webcam_tracker.logging_utils import configure_logging, get_logger
from webcam_tracker.perf_monitor import PerfMonitor
from webcam_tracker.target_selection import TargetSelector
from webcam_tracker.tracking import TrackedPerson, create_tracker
from webcam_tracker.video_input import VideoSourceError, create_source
from webcam_tracker.visualization import draw_perf_overlay, draw_target_overlay, draw_tracked_people

WINDOW_NAME = "webcam_tracker target selection"


@dataclass
class _MouseState:
    """Bridges the mouse callback (fires asynchronously) to the main loop's
    latest tracked people, so a click always selects against a current frame."""

    selector: TargetSelector
    tracked_people: list[TrackedPerson] = field(default_factory=list)


def _on_mouse(event: int, x: int, y: int, _flags: int, param: object) -> None:
    state = param
    assert isinstance(state, _MouseState)
    if event == cv2.EVENT_LBUTTONDOWN:
        state.selector.select_at_point((float(x), float(y)), state.tracked_people)
    elif event == cv2.EVENT_RBUTTONDOWN:
        state.selector.clear()


def main() -> None:
    config = load_config()
    configure_logging(level=config.logging.level, json_format=False)
    logger = get_logger(__name__)

    detector = create_detector(config)
    detector.load()
    tracker = create_tracker(config)
    selector = TargetSelector()

    source = create_source(config)
    perf = PerfMonitor(fps_window_seconds=config.perf_monitor.fps_window_seconds)

    mouse_state = _MouseState(selector=selector)
    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, _on_mouse, mouse_state)

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

                mouse_state.tracked_people = tracked_people
                height, width = frame.image.shape[:2]
                target_status = selector.status(tracked_people, width, height)

                image = frame.image.copy()
                draw_tracked_people(image, tracked_people)
                draw_target_overlay(image, target_status)
                extra = "click a person to select target, right-click/'c' to clear"
                draw_perf_overlay(image, perf.snapshot(), extra_text=extra)
                cv2.imshow(WINDOW_NAME, image)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):  # 'q' or Esc
                    break
                if key == ord("c"):
                    selector.clear()
    except VideoSourceError as exc:
        logger.error("Could not open video source", extra={"error": str(exc)})
        raise SystemExit(1) from exc
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
