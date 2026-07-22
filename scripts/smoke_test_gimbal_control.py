"""Headless smoke test: runs the full detect -> track -> select -> gimbal
chain on real webcam frames, auto-selecting the first confirmed track, and
prints the simulated gimbal command each frame. No GUI window needed.

Run from the repo root:
    .venv\\Scripts\\python.exe scripts\\smoke_test_gimbal_control.py

Expected output: after auto-selecting a target, pan/tilt velocity and angle
should respond smoothly to your movement (small velocities/angle changes
near center, growing as you move toward frame edges), with no huge jumps
between consecutive frames (that's the rate limiter working) and no
"nan"/crazy values.
"""

from __future__ import annotations

from webcam_tracker.config import load_config
from webcam_tracker.detection import create_detector
from webcam_tracker.gimbal_control import create_gimbal_controller
from webcam_tracker.logging_utils import configure_logging
from webcam_tracker.target_selection import TargetSelector
from webcam_tracker.tracking import create_tracker
from webcam_tracker.video_input import create_source

WARMUP_FRAMES = 10
OBSERVATION_FRAMES = 60


def main() -> None:
    config = load_config()
    configure_logging(level=config.logging.level, json_format=False)

    detector = create_detector(config)
    detector.load()
    tracker = create_tracker(config)
    selector = TargetSelector()
    gimbal = create_gimbal_controller(config)

    source = create_source(config)

    with source:
        for _ in range(WARMUP_FRAMES):
            source.read()

        selected = False
        for frame_number in range(OBSERVATION_FRAMES):
            frame = source.read()
            assert frame is not None
            height, width = frame.image.shape[:2]

            detections = detector.detect(frame.image)
            tracked_people = tracker.update(detections)

            if not selected and tracked_people:
                target = tracked_people[0]
                selector.select_at_point(target.center, tracked_people)
                selected = True
                print(f"frame {frame_number}: auto-selected track_id={target.track_id}")

            status = selector.status(tracked_people, width, height)
            if status is not None and status.visible:
                assert status.normalized_error is not None
                pan_error, tilt_error = status.normalized_error
            else:
                pan_error, tilt_error = None, None

            command = gimbal.compute(pan_error, tilt_error)

            if status is None:
                print(f"frame {frame_number}: no target selected yet")
            else:
                sat = "SAT" if (command.pan_saturated or command.tilt_saturated) else ""
                print(
                    f"frame {frame_number}: visible={status.visible} "
                    f"pan(v={command.pan_velocity_deg_s:+6.1f} a={command.pan_angle_deg:+6.1f}) "
                    f"tilt(v={command.tilt_velocity_deg_s:+6.1f} a={command.tilt_angle_deg:+6.1f}) "
                    f"{sat}"
                )

    if not selected:
        print("No track was ever confirmed during the observation window -- nothing to select.")


if __name__ == "__main__":
    main()
