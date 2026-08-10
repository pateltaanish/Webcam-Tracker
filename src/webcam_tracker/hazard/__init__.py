"""Looming-based hazard detection: which tracked person is closing on the camera.

Ported from the Watchdog wearable, which ran the identical math behind a
Hailo-8L NPU. The port drops the NPU entirely -- the accelerator only ever
supplied bounding boxes, and webcam_tracker's own CPU detector supplies those
now -- so nothing here imports hailo, gstreamer, or hailo_apps_infra. The
drone build has no Hailo hat.

Estimates time-to-collision from how fast a person's box grows (looming),
which needs no depth sensor and no camera calibration, then picks at most one
hazard per frame and rate-limits repeats.
"""

from webcam_tracker.hazard.monitor import Hazard, HazardMonitor
from webcam_tracker.hazard.ttc import TrackStore, compute_ttc, zone_from_cx

__all__ = ["Hazard", "HazardMonitor", "TrackStore", "compute_ttc", "zone_from_cx"]
