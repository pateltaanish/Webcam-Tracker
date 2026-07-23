"""Unit tests for webcam_tracker.registration.

Quality gates are pure image analysis -- tested with synthetic images and
thresholds crafted to isolate one failure at a time. The Registrar is tested
with a fake embedder (so no model is loaded) against a real temp store.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from webcam_tracker.database import KdfParams, ProfileStore
from webcam_tracker.face_recognition import DetectedFace
from webcam_tracker.registration import (
    QualityThresholds,
    Registrar,
    assess_face_quality,
)

_LENIENT = QualityThresholds(
    min_detection_score=0.5,
    min_blur_variance=10.0,
    min_brightness=40.0,
    max_brightness=220.0,
    min_face_fraction=0.01,
)


def _face(x1: float, y1: float, x2: float, y2: float, det: float = 0.9) -> DetectedFace:
    return DetectedFace(x1, y1, x2, y2, det, np.zeros(512, dtype=np.float32))


def _noise(h: int, w: int, mean: float) -> np.ndarray:
    rng = np.random.default_rng(0)
    return np.clip(rng.normal(mean, 40, (h, w, 3)), 0, 255).astype(np.uint8)


class TestQualityGates:
    def test_good_face_passes(self) -> None:
        image = _noise(200, 200, 128)
        report = assess_face_quality(image, _face(10, 10, 190, 190), _LENIENT)
        assert report.ok
        assert report.reasons == ()

    def test_face_box_outside_frame_rejected(self) -> None:
        image = _noise(200, 200, 128)
        report = assess_face_quality(image, _face(300, 300, 400, 400), _LENIENT)
        assert not report.ok
        assert any("outside" in r for r in report.reasons)

    def test_too_small_rejected(self) -> None:
        image = _noise(200, 200, 128)
        thresholds = QualityThresholds(0.5, 10.0, 40.0, 220.0, min_face_fraction=0.5)
        report = assess_face_quality(image, _face(10, 10, 40, 40), thresholds)
        assert not report.ok
        assert any("small" in r for r in report.reasons)

    def test_blurry_rejected(self) -> None:
        flat = np.full((200, 200, 3), 128, dtype=np.uint8)  # zero high-frequency detail
        report = assess_face_quality(flat, _face(10, 10, 190, 190), _LENIENT)
        assert not report.ok
        assert any("blurry" in r for r in report.reasons)

    def test_too_dark_rejected(self) -> None:
        image = _noise(200, 200, 20)  # sharp (noise) but dark
        report = assess_face_quality(image, _face(10, 10, 190, 190), _LENIENT)
        assert not report.ok
        assert any("dark" in r for r in report.reasons)

    def test_too_bright_rejected(self) -> None:
        image = _noise(200, 200, 245)
        report = assess_face_quality(image, _face(10, 10, 190, 190), _LENIENT)
        assert not report.ok
        assert any("bright" in r for r in report.reasons)

    def test_low_detection_score_rejected(self) -> None:
        image = _noise(200, 200, 128)
        report = assess_face_quality(image, _face(10, 10, 190, 190, det=0.2), _LENIENT)
        assert not report.ok
        assert any("confidence" in r for r in report.reasons)


class _FakeEmbedder:
    """Duck-typed stand-in for FaceEmbedder -- returns preset faces, no model."""

    model_id = "fake/model"

    def __init__(self, faces: list[DetectedFace]) -> None:
        self._faces = faces

    def detect(self, image: np.ndarray) -> list[DetectedFace]:
        return list(self._faces)


def _store(tmp_path: Path) -> ProfileStore:
    store = ProfileStore(
        tmp_path / "profiles.db",
        tmp_path / "keyvault.json",
        KdfParams(1, 8192, 1),
        min_passphrase_length=4,
    )
    store.initialize("test-pass")
    return store


def _good_image() -> np.ndarray:
    return _noise(200, 200, 128)


class TestRegistrarEvaluate:
    def test_no_face_rejected(self, tmp_path: Path) -> None:
        registrar = Registrar(_store(tmp_path), _FakeEmbedder([]), _LENIENT, "v1")  # type: ignore[arg-type]
        result = registrar.evaluate(_good_image())
        assert not result.accepted
        assert result.num_faces == 0

    def test_multiple_faces_rejected(self, tmp_path: Path) -> None:
        faces = [_face(10, 10, 90, 90), _face(100, 100, 190, 190)]
        registrar = Registrar(_store(tmp_path), _FakeEmbedder(faces), _LENIENT, "v1")  # type: ignore[arg-type]
        result = registrar.evaluate(_good_image())
        assert not result.accepted
        assert result.num_faces == 2
        assert any("one person" in r for r in result.reasons)

    def test_single_good_face_accepted(self, tmp_path: Path) -> None:
        face = _face(10, 10, 190, 190)
        registrar = Registrar(_store(tmp_path), _FakeEmbedder([face]), _LENIENT, "v1")  # type: ignore[arg-type]
        result = registrar.evaluate(_good_image())
        assert result.accepted
        assert result.face is face


class TestRegistrarEnroll:
    def test_enroll_stores_person_and_embeddings(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        face = DetectedFace(0, 0, 1, 1, 0.9, np.arange(512, dtype=np.float32))
        registrar = Registrar(store, _FakeEmbedder([face]), _LENIENT, "v1")  # type: ignore[arg-type]

        person = registrar.enroll("Alice", [face, face], consent_note="test")
        loaded = store.get_person(person.id)
        assert loaded is not None
        assert loaded.display_name == "Alice"

        embeddings = store.get_embeddings(person.id)
        assert len(embeddings) == 2
        assert embeddings[0].kind == "face"
        assert embeddings[0].model_id == "fake/model"
        np.testing.assert_array_equal(embeddings[0].vector, face.embedding)
        store.close()

    def test_enroll_with_no_samples_raises(self, tmp_path: Path) -> None:
        registrar = Registrar(_store(tmp_path), _FakeEmbedder([]), _LENIENT, "v1")  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            registrar.enroll("Alice", [])
