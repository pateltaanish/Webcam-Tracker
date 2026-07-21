"""Integration tests for PersonDetector -- these load the real YOLO11n
weights (downloading them on first run) and run real inference, against
Ultralytics' own bundled sample images (shipped inside the installed
`ultralytics` package -- no network dependency or repo bloat needed)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import ultralytics

from webcam_tracker.config import load_config
from webcam_tracker.detection import PersonDetector, create_detector

_ASSETS_DIR = Path(ultralytics.__file__).resolve().parent / "assets"


@pytest.fixture(scope="module")
def loaded_detector() -> PersonDetector:
    config = load_config()
    detector = create_detector(config)
    detector.load()
    return detector


class TestPersonDetectorRealInference:
    def test_detects_people_in_bundled_sample_image(self, loaded_detector: PersonDetector) -> None:
        image = cv2.imread(str(_ASSETS_DIR / "zidane.jpg"))
        assert image is not None, "Ultralytics sample image failed to load"

        detections = loaded_detector.detect(image)

        assert len(detections) >= 1
        for detection in detections:
            assert 0.0 <= detection.confidence <= 1.0
            assert detection.x1 < detection.x2
            assert detection.y1 < detection.y2

    def test_no_detections_in_blank_image(self, loaded_detector: PersonDetector) -> None:
        blank = np.full((480, 640, 3), fill_value=128, dtype=np.uint8)
        assert loaded_detector.detect(blank) == []

    def test_higher_threshold_yields_no_more_detections_than_lower(self) -> None:
        image = cv2.imread(str(_ASSETS_DIR / "bus.jpg"))
        assert image is not None

        config = load_config()
        lenient = create_detector(config)
        lenient._confidence_threshold = 0.1  # noqa: SLF001 -- testing threshold behavior directly
        lenient.load()

        strict = create_detector(config)
        strict._confidence_threshold = 0.95  # noqa: SLF001
        strict.load()

        assert len(strict.detect(image)) <= len(lenient.detect(image))
