"""Multi-object tracking via ByteTrack, using the `trackers` package (Apache-2.0).

Note on library choice: `supervision`'s built-in `sv.ByteTrack` is deprecated
(removed in supervision 0.30.0) in favor of `ByteTrackTracker` from the
dedicated `trackers` package -- that's what this wraps. `trackers` still
uses `supervision.Detections` as its data structure (that part isn't
deprecated), so both packages are dependencies.

Why wrap it at all instead of using it directly: everywhere else in this
codebase works with our own `Detection`/`TrackedPerson` types, not
supervision's. That's what lets `detection` and `tracking` be swapped
independently later (e.g. a different tracker library) without every
downstream module needing to know about it.
"""

from __future__ import annotations

import numpy as np
import supervision as sv
from trackers import ByteTrackTracker

from webcam_tracker.detection import Detection
from webcam_tracker.logging_utils import get_logger
from webcam_tracker.tracking.tracked_person import UNCONFIRMED_TRACK_ID, TrackedPerson

logger = get_logger(__name__)


def _to_sv_detections(detections: list[Detection]) -> sv.Detections:
    if not detections:
        return sv.Detections.empty()
    xyxy = np.array([[d.x1, d.y1, d.x2, d.y2] for d in detections], dtype=np.float32)
    confidence = np.array([d.confidence for d in detections], dtype=np.float32)
    # Single-class tracking (person only) -- class_id is required by the API
    # but not meaningful here, so it's a constant.
    class_id = np.zeros(len(detections), dtype=int)
    return sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)


def _from_sv_detections(result: sv.Detections) -> list[TrackedPerson]:
    assert result.tracker_id is not None  # always populated by ByteTrackTracker.update()
    assert result.confidence is not None

    tracked_people: list[TrackedPerson] = []
    for i in range(len(result)):
        track_id = int(result.tracker_id[i])
        if track_id == UNCONFIRMED_TRACK_ID:
            continue  # not yet confirmed -- see TrackedPerson's docstring
        x1, y1, x2, y2 = (float(v) for v in result.xyxy[i])
        tracked_people.append(
            TrackedPerson(
                track_id=track_id,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                confidence=float(result.confidence[i]),
            )
        )
    return tracked_people


class PersonTracker:
    """Assigns persistent track IDs to person detections across frames."""

    def __init__(
        self,
        lost_track_buffer: int,
        frame_rate: float,
        track_activation_threshold: float,
        minimum_consecutive_frames: int,
        minimum_iou_threshold: float,
        high_conf_det_threshold: float,
    ) -> None:
        self._tracker = ByteTrackTracker(
            lost_track_buffer=lost_track_buffer,
            frame_rate=frame_rate,
            track_activation_threshold=track_activation_threshold,
            minimum_consecutive_frames=minimum_consecutive_frames,
            minimum_iou_threshold=minimum_iou_threshold,
            high_conf_det_threshold=high_conf_det_threshold,
        )
        self._known_track_ids: set[int] = set()

    def update(self, detections: list[Detection]) -> list[TrackedPerson]:
        """Advance the tracker by one frame's worth of detections.

        Returns only *confirmed* tracks (see TrackedPerson docstring) -- a
        brand-new person takes `minimum_consecutive_frames` frames to first
        appear in this output, and a track that stops matching simply stops
        appearing (it is not returned with a stale/frozen box).
        """
        sv_detections = _to_sv_detections(detections)
        result = self._tracker.update(sv_detections)
        tracked_people = _from_sv_detections(result)
        self._log_new_tracks(tracked_people)
        return tracked_people

    def _log_new_tracks(self, tracked_people: list[TrackedPerson]) -> None:
        for person in tracked_people:
            if person.track_id not in self._known_track_ids:
                self._known_track_ids.add(person.track_id)
                logger.info(
                    "Track started",
                    extra={"track_id": person.track_id, "confidence": person.confidence},
                )

    def reset(self) -> None:
        """Clear all tracker state (e.g. between unrelated video files)."""
        self._tracker.reset()
        self._known_track_ids.clear()
