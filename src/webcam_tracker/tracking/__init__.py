"""Multi-object tracking.

Wraps ByteTrack (via the `trackers` package) to assign persistent track IDs
to person detections across frames (see docs/02_architecture.md sec 3.2).
"""

from webcam_tracker.tracking.factory import create_tracker
from webcam_tracker.tracking.tracked_person import UNCONFIRMED_TRACK_ID, TrackedPerson
from webcam_tracker.tracking.tracker import PersonTracker

__all__ = ["PersonTracker", "TrackedPerson", "UNCONFIRMED_TRACK_ID", "create_tracker"]
