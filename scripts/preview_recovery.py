"""Interactive live preview of motion prediction + target-loss recovery (Stage 1.8).

Everything scripts\\preview_gimbal_control.py does (detect + track +
click-to-select + simulated gimbal), plus the Stage 1.8 loss/recovery layer:

  - motion_prediction estimates each track's velocity;
  - when your selected target leaves frame, recovery predicts where it went
    (a magenta 'predicted' marker + a re-lock radius circle), drives the
    gimbal in a search sweep toward that direction, and shows a RECOVERY
    banner;
  - when a NEW track appears inside the re-lock circle, recovery re-locks onto
    it automatically (this is the identity-FREE placeholder -- it grabs
    whoever walks in there, not necessarily you; Stage 2 adds real identity).

Try it: select yourself, walk out of frame, and watch it search; walk back in
and watch it re-lock (note your track ID changes -- that's the ID-switch
recovery is papering over until Stage 2's re-id does it properly).

Controls:
  Left-click a tracked person  -> select as target
  Right-click, or 'c'          -> clear target selection
  'e' / 'r'                    -> gimbal emergency stop / resume
  'q' / Esc                    -> quit

Run from the repo root:
    .venv\\Scripts\\python.exe scripts\\preview_recovery.py
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2

from webcam_tracker.config import load_config
from webcam_tracker.detection import create_detector
from webcam_tracker.gimbal_control import create_gimbal_controller
from webcam_tracker.logging_utils import configure_logging, get_logger
from webcam_tracker.motion_prediction import create_motion_predictor
from webcam_tracker.perf_monitor import PerfMonitor
from webcam_tracker.recovery import RecoveryState, create_recovery_controller
from webcam_tracker.target_selection import TargetSelector
from webcam_tracker.tracking import TrackedPerson, create_tracker
from webcam_tracker.video_input import VideoSourceError, create_source
from webcam_tracker.visualization import (
    draw_gimbal_widget,
    draw_perf_overlay,
    draw_recovery_overlay,
    draw_target_overlay,
    draw_tracked_people,
)

WINDOW_NAME = "webcam_tracker recovery preview"


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
    predictor = create_motion_predictor(config)
    selector = TargetSelector()
    recovery = create_recovery_controller(config)
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
                predictor.update(tracked_people)
                perf.end_frame()
                perf.log_if_due(config.perf_monitor.log_interval_seconds)

                mouse_state.tracked_people = tracked_people
                height, width = frame.image.shape[:2]

                target_id = selector.target_id
                target_status = selector.status(tracked_people, width, height)
                target_motion = predictor.motion(target_id) if target_id is not None else None
                recovery_status = recovery.update(
                    target_status, tracked_people, target_motion, width, height
                )

                # Act on a reacquisition recommendation: re-lock the selector.
                if recovery_status.reacquire_track_id is not None:
                    selector.select(recovery_status.reacquire_track_id)

                # Choose what error drives the gimbal this frame: the real
                # target error when visible, the search sweep while searching,
                # otherwise nothing (hold).
                if target_status is not None and target_status.visible:
                    assert target_status.normalized_error is not None
                    pan_error, tilt_error = target_status.normalized_error
                elif recovery_status.state is RecoveryState.SEARCHING:
                    assert recovery_status.search_error is not None
                    pan_error, tilt_error = recovery_status.search_error
                else:
                    pan_error, tilt_error = None, None
                command = gimbal.compute(pan_error, tilt_error)

                image = frame.image.copy()
                draw_tracked_people(image, tracked_people)
                draw_target_overlay(image, target_status)
                draw_recovery_overlay(
                    image,
                    recovery_status,
                    reacquire_radius_fraction=config.recovery.reacquire_radius_fraction,
                )
                draw_gimbal_widget(
                    image,
                    command,
                    pan_limit_deg=config.gimbal.pan.angle_limit_deg,
                    tilt_limit_deg=config.gimbal.tilt.angle_limit_deg,
                )
                extra = "click=select  'c'=clear  'e'/'r'=estop/resume"
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
