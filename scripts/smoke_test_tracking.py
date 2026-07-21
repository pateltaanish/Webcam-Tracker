"""Headless smoke test: runs detection + tracking on ~3 seconds of real
webcam frames and reports whether track IDs stayed stable, without needing a
GUI window. This is what actually verifies tracking is working (a single
frame can't show ID persistence -- that's the whole point of tracking).

Run from the repo root:
    .venv\\Scripts\\python.exe scripts\\smoke_test_tracking.py

Expected output: a per-frame line showing which track ID(s) were seen, then
a summary. For a single person standing in frame, you should see ONE track
ID dominate almost all frames after a short (~2 frame) startup delay --
that's a track staying stable, not switching identities.
"""

from __future__ import annotations

from collections import Counter

from webcam_tracker.config import load_config
from webcam_tracker.detection import create_detector
from webcam_tracker.logging_utils import configure_logging
from webcam_tracker.tracking import create_tracker
from webcam_tracker.video_input import create_source

WARMUP_FRAMES = 10
OBSERVATION_FRAMES = 90  # ~3 seconds at 30fps


def main() -> None:
    config = load_config()
    configure_logging(level=config.logging.level, json_format=False)

    detector = create_detector(config)
    detector.load()
    tracker = create_tracker(config)

    source = create_source(config)
    id_counts: Counter[int] = Counter()

    with source:
        for _ in range(WARMUP_FRAMES):
            source.read()

        for frame_number in range(OBSERVATION_FRAMES):
            frame = source.read()
            assert frame is not None
            detections = detector.detect(frame.image)
            tracked_people = tracker.update(detections)

            ids = [p.track_id for p in tracked_people]
            id_counts.update(ids)
            print(f"frame {frame_number:>3}: track_ids={ids}")

    print()
    print("Summary:")
    print(f"  Unique track IDs seen: {sorted(id_counts.keys())}")
    for track_id, count in id_counts.most_common():
        pct = 100 * count / OBSERVATION_FRAMES
        print(f"  ID {track_id}: present in {count}/{OBSERVATION_FRAMES} frames ({pct:.0f}%)")


if __name__ == "__main__":
    main()
