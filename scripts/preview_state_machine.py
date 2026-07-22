"""Interactive live preview of the full tracking state machine (Stage 1.9).

This is the whole Stage 1 pipeline wired together by the state machine. Notice
how little per-frame logic is left here compared to preview_recovery.py: the
detector + tracker produce tracked people, and a SINGLE
`state_machine.update(...)` call coordinates selection, motion prediction,
recovery, and the gimbal into one state + one set of outputs. All the
"what should the system do right now" logic that used to be an if/elif pile in
the preview scripts now lives (and is tested) in one place.

Watch the state banner (top-left, under the FPS line) change as you move:
  IDLE (gray)              -- nothing selected
  TRACKING (green)         -- following your selected target
  TEMPORARILY_OCCLUDED     -- you briefly vanished; it HOLDS and waits for the
    (yellow)                  same track id to come back
  RECOVERY_SEARCH (orange) -- you've been gone too long; it searches (gimbal
                              sweeps toward the predicted direction)
  SAFE_HOVER_REQUESTED     -- it gave up searching
    (red)
  STOPPED (red)            -- emergency stop
A "DRONE-ASSIST" flag appears when the gimbal is maxed out and the drone body
would need to move -- that's the signal a future drone_control will consume.

Controls:
  Left-click a tracked person  -> select as target
  Right-click, or 'c'          -> clear target selection
  'e' / 'r'                    -> emergency stop / resume
  'q' / Esc                    -> quit

Run from the repo root:
    .venv\\Scripts\\python.exe scripts\\preview_state_machine.py
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2

from webcam_tracker.config import load_config
from webcam_tracker.detection import create_detector
from webcam_tracker.logging_utils import configure_logging, get_logger
from webcam_tracker.perf_monitor import PerfMonitor
from webcam_tracker.state_machine import create_state_machine
from webcam_tracker.target_selection import TargetSelector
from webcam_tracker.tracking import TrackedPerson, create_tracker
from webcam_tracker.video_input import VideoSourceError, create_source
from webcam_tracker.visualization import (
    draw_gimbal_widget,
    draw_perf_overlay,
    draw_recovery_overlay,
    draw_state_banner,
    draw_target_overlay,
    draw_tracked_people,
)

WINDOW_NAME = "webcam_tracker state machine preview"


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
    state_machine = create_state_machine(config)

    source = create_source(config)
    perf = PerfMonitor(fps_window_seconds=config.perf_monitor.fps_window_seconds)

    # The state machine owns the selector; the mouse callback drives it.
    mouse_state = _MouseState(selector=state_machine.selector)
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

                status = state_machine.update(tracked_people, width, height)

                image = frame.image.copy()
                draw_tracked_people(image, tracked_people)
                draw_target_overlay(image, status.target_status)
                draw_recovery_overlay(
                    image,
                    status.recovery_status,
                    reacquire_radius_fraction=config.recovery.reacquire_radius_fraction,
                )
                draw_gimbal_widget(
                    image,
                    status.gimbal_command,
                    pan_limit_deg=config.gimbal.pan.angle_limit_deg,
                    tilt_limit_deg=config.gimbal.tilt.angle_limit_deg,
                )
                draw_state_banner(image, status)
                extra = "click=select  'c'=clear  'e'/'r'=estop/resume"
                draw_perf_overlay(image, perf.snapshot(), extra_text=extra)
                cv2.imshow(WINDOW_NAME, image)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):  # 'q' or Esc
                    break
                if key == ord("c"):
                    state_machine.selector.clear()
                elif key == ord("e"):
                    state_machine.emergency_stop()
                elif key == ord("r"):
                    state_machine.resume()
    except VideoSourceError as exc:
        logger.error("Could not open video source", extra={"error": str(exc)})
        raise SystemExit(1) from exc
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
