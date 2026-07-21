"""Headless smoke test: runs detection + tracking on real webcam frames,
auto-"clicks" the first confirmed track's own center (simulating what a
mouse click would do), then verifies the target status/pixel-error
computation over a few more frames. No GUI window needed.

Run from the repo root:
    .venv\\Scripts\\python.exe scripts\\smoke_test_target_selection.py

Expected output: a line reporting which track got auto-selected, then a few
frames of status showing visible=True and a pixel error close to 0 (since
we selected exactly at that person's own center) that tracks the person's
real movement in front of the camera.
"""

from __future__ import annotations

from webcam_tracker.config import load_config
from webcam_tracker.detection import create_detector
from webcam_tracker.logging_utils import configure_logging
from webcam_tracker.target_selection import TargetSelector
from webcam_tracker.tracking import create_tracker
from webcam_tracker.video_input import create_source

WARMUP_FRAMES = 10
OBSERVATION_FRAMES = 30


def main() -> None:
    config = load_config()
    configure_logging(level=config.logging.level, json_format=False)

    detector = create_detector(config)
    detector.load()
    tracker = create_tracker(config)
    selector = TargetSelector()

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
                print(f"  at center={target.center}")

            status = selector.status(tracked_people, width, height)
            if status is None:
                print(f"frame {frame_number}: no target selected yet")
            elif not status.visible:
                print(f"frame {frame_number}: TARGET LOST (id={status.target_id})")
            else:
                dx, dy = status.pixel_error  # type: ignore[misc]
                ndx, ndy = status.normalized_error  # type: ignore[misc]
                print(
                    f"frame {frame_number}: visible id={status.target_id} "
                    f"center={status.target_center} err=({dx:+.1f},{dy:+.1f})px "
                    f"norm=({ndx:+.2f},{ndy:+.2f})"
                )

    if not selected:
        print("No track was ever confirmed during the observation window -- nothing to select.")


if __name__ == "__main__":
    main()
