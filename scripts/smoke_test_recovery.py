"""Headless smoke test for motion_prediction + recovery (Stage 1.8).

Runs the full detect -> track -> predict -> select -> recovery chain on real
webcam frames. To exercise the loss/search path deterministically without
asking you to physically walk out of frame at an exact moment, it
*simulates* a loss: after a short tracking phase it stops showing the
selected target to the recovery controller (as if the person left), then
prints the recovery state each frame until it gives up.

What to look for:
  - a tracking phase where the predictor builds a velocity estimate;
  - the moment of (simulated) loss -> state SEARCHING, with a predicted
    position that extrapolates from the last-known velocity, and a search
    error (the sweep that would be fed to the gimbal);
  - eventually GAVE_UP after recovery.give_up_seconds with no reacquisition.

Reacquisition onto a brand-new track is best seen live in
scripts\\preview_recovery.py (walk out and back in); it's covered exactly by
the unit tests here in headless form.

Run from the repo root:
    .venv\\Scripts\\python.exe scripts\\smoke_test_recovery.py
"""

from __future__ import annotations

from webcam_tracker.config import load_config
from webcam_tracker.detection import create_detector
from webcam_tracker.logging_utils import configure_logging
from webcam_tracker.motion_prediction import create_motion_predictor
from webcam_tracker.recovery import RecoveryState, create_recovery_controller
from webcam_tracker.target_selection import TargetSelector
from webcam_tracker.tracking import create_tracker
from webcam_tracker.video_input import create_source

WARMUP_FRAMES = 10
TRACKING_FRAMES = 30
MAX_LOSS_FRAMES = 400  # safety cap so we never loop forever if give-up is slow


def main() -> None:
    config = load_config()
    configure_logging(level=config.logging.level, json_format=False)

    detector = create_detector(config)
    detector.load()
    tracker = create_tracker(config)
    predictor = create_motion_predictor(config)
    selector = TargetSelector()
    recovery = create_recovery_controller(config)

    source = create_source(config)

    with source:
        for _ in range(WARMUP_FRAMES):
            source.read()

        # Phase 1: normal tracking -- select the first confirmed track and let
        # the predictor build up a velocity estimate from real motion.
        selected = False
        for _ in range(TRACKING_FRAMES):
            frame = source.read()
            assert frame is not None
            height, width = frame.image.shape[:2]
            tracked = tracker.update(detector.detect(frame.image))
            predictor.update(tracked)

            if not selected and tracked:
                selector.select_at_point(tracked[0].center, tracked)
                selected = True
                print(f"auto-selected track_id={tracked[0].track_id}")

            target_id = selector.target_id
            status = selector.status(tracked, width, height)
            motion = predictor.motion(target_id) if target_id is not None else None
            recovery.update(status, tracked, motion, width, height)
            if motion is not None:
                print(
                    f"tracking: vel=({motion.velocity[0]:+.0f},{motion.velocity[1]:+.0f})px/s "
                    f"speed={motion.speed:.0f}"
                )

        if not selected:
            print("No track was ever confirmed -- nothing to recover. Sit in frame and rerun.")
            return

        target_id = selector.target_id
        print(f"\n--- simulating loss of track_id={target_id} (hiding it from recovery) ---")

        # Phase 2: simulated loss -- hide the target from what recovery sees, so
        # its track appears to have vanished, and watch the search play out.
        for i in range(MAX_LOSS_FRAMES):
            frame = source.read()
            assert frame is not None
            height, width = frame.image.shape[:2]
            tracked = tracker.update(detector.detect(frame.image))
            predictor.update(tracked)

            visible = [p for p in tracked if p.track_id != target_id]
            status = selector.status(visible, width, height)
            motion = predictor.motion(target_id) if target_id is not None else None
            result = recovery.update(status, visible, motion, width, height)

            predicted = result.predicted_position
            predicted_text = (
                f"predicted=({predicted[0]:.0f},{predicted[1]:.0f})" if predicted else "predicted=-"
            )
            search_text = (
                f"search_err=({result.search_error[0]:+.2f},{result.search_error[1]:+.2f})"
                if result.search_error
                else "search_err=-"
            )
            reacq = f" REACQUIRE->{result.reacquire_track_id}" if result.reacquire_track_id else ""
            print(f"loss frame {i}: {result.state.value:9} {predicted_text} {search_text}{reacq}")

            if result.state is RecoveryState.GAVE_UP:
                print("recovery gave up (expected after give_up_seconds) -- done.")
                break
        else:
            print("hit MAX_LOSS_FRAMES without giving up -- give_up_seconds may be long.")


if __name__ == "__main__":
    main()
