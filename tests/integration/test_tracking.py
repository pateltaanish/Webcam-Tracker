"""Integration test: real detector output fed through the real tracker.

Simulates a short "video" by running detection repeatedly on the same real
sample image -- confirms the detection -> tracking hand-off actually works
end to end (not just each module in isolation), i.e. that a person detected
in consecutive frames keeps the same track_id.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import ultralytics

from webcam_tracker.config import load_config
from webcam_tracker.detection import PersonDetector, create_detector
from webcam_tracker.tracking import PersonTracker, create_tracker

_ASSETS_DIR = Path(ultralytics.__file__).resolve().parent / "assets"


def test_same_person_keeps_track_id_across_repeated_frames() -> None:
    config = load_config()

    detector: PersonDetector = create_detector(config)
    detector.load()
    tracker: PersonTracker = create_tracker(config)

    image = cv2.imread(str(_ASSETS_DIR / "bus.jpg"))
    assert image is not None

    track_id_per_frame = []
    for _ in range(5):
        detections = detector.detect(image)
        tracked = tracker.update(detections)
        track_id_per_frame.append({t.track_id for t in tracked})

    # Give the tracker's confirmation delay (minimum_consecutive_frames) room
    # to settle, then require the same set of IDs to persist afterward.
    settled = track_id_per_frame[-1]
    assert settled, "expected at least one confirmed track on a real sample image"
    assert track_id_per_frame[-2] == settled
