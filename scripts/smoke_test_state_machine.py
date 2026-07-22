"""Headless smoke test for the tracking state machine (Stage 1.9).

Runs the full pipeline through the state machine on real webcam frames and
prints the authoritative system state each frame, so the whole
IDLE -> TRACKING -> TEMPORARILY_OCCLUDED -> RECOVERY_SEARCH ->
SAFE_HOVER_REQUESTED progression can be verified end to end.

Like smoke_test_recovery.py, it *simulates* the loss (hides the selected
target from the state machine after a tracking phase) rather than asking you
to physically leave frame at a scripted moment.

Run from the repo root:
    .venv\\Scripts\\python.exe scripts\\smoke_test_state_machine.py
"""

from __future__ import annotations

from webcam_tracker.config import load_config
from webcam_tracker.detection import create_detector
from webcam_tracker.logging_utils import configure_logging
from webcam_tracker.state_machine import TrackingState, create_state_machine
from webcam_tracker.tracking import create_tracker
from webcam_tracker.video_input import create_source

WARMUP_FRAMES = 10
TRACKING_FRAMES = 20
MAX_LOSS_FRAMES = 400


def main() -> None:
    config = load_config()
    configure_logging(level=config.logging.level, json_format=False)

    detector = create_detector(config)
    detector.load()
    tracker = create_tracker(config)
    state_machine = create_state_machine(config)

    source = create_source(config)

    with source:
        for _ in range(WARMUP_FRAMES):
            source.read()

        # Phase 1: select the first confirmed track and follow it.
        selected = False
        for _ in range(TRACKING_FRAMES):
            frame = source.read()
            assert frame is not None
            height, width = frame.image.shape[:2]
            tracked = tracker.update(detector.detect(frame.image))

            if not selected and tracked:
                state_machine.selector.select_at_point(tracked[0].center, tracked)
                selected = True
                print(f"auto-selected track_id={tracked[0].track_id}")

            status = state_machine.update(tracked, width, height)
            print(f"track phase: state={status.state.value}")

        if not selected:
            print("No track was ever confirmed -- sit in frame and rerun.")
            return

        target_id = state_machine.selector.target_id
        print(f"\n--- simulating loss of track_id={target_id} ---")

        # Phase 2: hide the target so the state machine sees it as gone.
        last_state = None
        for i in range(MAX_LOSS_FRAMES):
            frame = source.read()
            assert frame is not None
            height, width = frame.image.shape[:2]
            tracked = tracker.update(detector.detect(frame.image))
            visible = [p for p in tracked if p.track_id != target_id]

            status = state_machine.update(visible, width, height)
            if status.state != last_state:
                print(f"loss frame {i}: -> {status.state.value} (t={status.time_in_state_s:.2f}s)")
                last_state = status.state
            if status.reacquired_track_id is not None:
                print(f"  reacquired new track_id={status.reacquired_track_id}")

            if status.state is TrackingState.SAFE_HOVER_REQUESTED:
                print("reached SAFE_HOVER_REQUESTED (search gave up) -- done.")
                break
        else:
            print("hit MAX_LOSS_FRAMES without safe-hover -- timeouts may be long.")


if __name__ == "__main__":
    main()
