"""Interactive live preview of the simulated gimbal controller (Stage 1.7).

Same detect + track + click-to-select flow as
scripts\\preview_target_selection.py, with the simulated gimbal controller
added on top: the target's normalized error feeds a PID controller per axis
(pan/tilt), and the result is drawn as a small attitude-indicator widget in
the top-right corner -- a dot showing the gimbal's current simulated
position within its angle limits, green normally, red when an axis is
saturated (hit its velocity or angle limit), orange when emergency-stopped.

No physical gimbal exists yet -- nothing here drives real hardware. This is
the control *logic* (PID, deadband, rate limiting, clamping, e-stop) running
against real tracking data, so it can be designed and watched working before
any hardware exists. See webcam_tracker.gimbal_control's module docstring.

Controls:
  Left-click a tracked person  -> select as target
  Right-click, or 'c'          -> clear target selection
  'e'                          -> emergency stop (freezes the simulated gimbal)
  'r'                          -> resume after an emergency stop
  'q' / Esc                    -> quit

Run from the repo root:
    .venv\\Scripts\\python.exe scripts\\preview_gimbal_control.py

Expected output: a window titled "webcam_tracker gimbal control preview".
Select a target and move around -- the widget's dot should move smoothly
(not jump) and turn red when you go far enough off-center that either axis
hits its configured limit.

Common errors: same as scripts\\preview_webcam.py for camera issues.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2

from webcam_tracker.config import load_config
from webcam_tracker.detection import create_detector
from webcam_tracker.gimbal_control import create_gimbal_controller
from webcam_tracker.logging_utils import configure_logging, get_logger
from webcam_tracker.perf_monitor import PerfMonitor
from webcam_tracker.target_selection import TargetSelector
from webcam_tracker.tracking import TrackedPerson, create_tracker
from webcam_tracker.video_input import VideoSourceError, create_source
from webcam_tracker.visualization import (
    draw_gimbal_widget,
    draw_perf_overlay,
    draw_target_overlay,
    draw_tracked_people,
)

WINDOW_NAME = "webcam_tracker gimbal control preview"


@dataclass
class _MouseState:
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
    gimbal = create_gimbal_controller(config)

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

                if target_status is not None and target_status.visible:
                    assert target_status.normalized_error is not None
                    pan_error, tilt_error = target_status.normalized_error
                else:
                    pan_error, tilt_error = None, None
                command = gimbal.compute(pan_error, tilt_error)

                image = frame.image.copy()
                draw_tracked_people(image, tracked_people)
                draw_target_overlay(image, target_status)
                draw_gimbal_widget(
                    image,
                    command,
                    pan_limit_deg=config.gimbal.pan.angle_limit_deg,
                    tilt_limit_deg=config.gimbal.tilt.angle_limit_deg,
                )
                extra = "click=select  right-click/'c'=clear  'e'=e-stop  'r'=resume"
                draw_perf_overlay(image, perf.snapshot(), extra_text=extra)
                cv2.imshow(WINDOW_NAME, image)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):  # 'q' or Esc
                    break
                if key == ord("c"):
                    selector.clear()
                elif key == ord("e"):
                    gimbal.emergency_stop()
                elif key == ord("r"):
                    gimbal.resume()
    except VideoSourceError as exc:
        logger.error("Could not open video source", extra={"error": str(exc)})
        raise SystemExit(1) from exc
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
