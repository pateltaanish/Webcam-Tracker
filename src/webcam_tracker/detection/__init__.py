"""Person detection.

Wraps a YOLO11n model (person class only -- see docs/02_architecture.md
sec 3.1) and returns bounding boxes + confidence per frame.
"""

from webcam_tracker.detection.detector import (
    Detection,
    DetectorError,
    PersonDetector,
    find_person_class_id,
    resolve_device,
)
from webcam_tracker.detection.factory import create_detector

__all__ = [
    "Detection",
    "DetectorError",
    "PersonDetector",
    "create_detector",
    "find_person_class_id",
    "resolve_device",
]
