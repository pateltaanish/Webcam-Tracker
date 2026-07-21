"""Unit tests for webcam_tracker.detection -- pure logic only, no model load
(model loading + real inference is covered in tests/integration/test_detection.py,
since it's slow and needs the actual downloaded weights)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from webcam_tracker.detection import Detection, DetectorError, PersonDetector, find_person_class_id
from webcam_tracker.detection.detector import resolve_device


class TestResolveDevice:
    def test_auto_prefers_cuda_when_available(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("torch.cuda.is_available", lambda: True)
        assert resolve_device("auto") == "cuda:0"

    def test_auto_falls_back_to_cpu_when_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("torch.cuda.is_available", lambda: False)
        assert resolve_device("auto") == "cpu"

    def test_explicit_cpu_is_passed_through(self) -> None:
        assert resolve_device("cpu") == "cpu"

    def test_explicit_cuda_raises_when_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("torch.cuda.is_available", lambda: False)
        with pytest.raises(DetectorError):
            resolve_device("cuda:0")


class TestFindPersonClassId:
    def test_finds_person_among_other_classes(self) -> None:
        names = {0: "cat", 1: "person", 2: "dog"}
        assert find_person_class_id(names) == 1

    def test_raises_when_no_person_class(self) -> None:
        with pytest.raises(DetectorError):
            find_person_class_id({0: "cat", 1: "dog"})


class TestDetection:
    def test_center_width_height(self) -> None:
        detection = Detection(x1=10.0, y1=20.0, x2=30.0, y2=60.0, confidence=0.9)
        assert detection.center == (20.0, 40.0)
        assert detection.width == 20.0
        assert detection.height == 40.0


class TestPersonDetectorBeforeLoad:
    def test_detect_before_load_raises(self) -> None:
        detector = PersonDetector(model_path=Path("unused.pt"), confidence_threshold=0.5)
        image = np.zeros((48, 64, 3), dtype=np.uint8)
        with pytest.raises(DetectorError):
            detector.detect(image)

    def test_is_loaded_false_before_load(self) -> None:
        detector = PersonDetector(model_path=Path("unused.pt"), confidence_threshold=0.5)
        assert detector.is_loaded is False
