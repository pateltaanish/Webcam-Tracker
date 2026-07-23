"""Integration test: the real face embedder through registration into the
encrypted store (Stage 2.2).

Loads the real InsightFace model and runs the full registration backend --
detect + embed a real face, quality-gate it, enroll it, and read it back out
of the encrypted store -- confirming the whole chain works end to end (the
interactive webcam capture in scripts/register_person.py is the only part not
exercised here). Uses a bundled sample image cropped to a single face so the
single-person enrollment gate is satisfied.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import ultralytics

from webcam_tracker.config import load_config
from webcam_tracker.database import KdfParams, ProfileStore
from webcam_tracker.face_recognition import FaceEmbedder, create_face_embedder
from webcam_tracker.registration import create_registrar

_ASSETS_DIR = Path(ultralytics.__file__).resolve().parent / "assets"


@pytest.fixture(scope="module")
def embedder() -> FaceEmbedder:
    emb = create_face_embedder(load_config())
    emb.load()
    return emb


def _single_face_image() -> np.ndarray:
    """zidane.jpg cropped to the right side, which contains exactly one face."""
    image = cv2.imread(str(_ASSETS_DIR / "zidane.jpg"))
    assert image is not None
    return image[0:400, 760:1280].copy()


def _fast_store(tmp_path: Path) -> ProfileStore:
    return ProfileStore(
        tmp_path / "profiles.db",
        tmp_path / "keyvault.json",
        KdfParams(1, 8192, 1),
        min_passphrase_length=4,
    )


class TestFaceEmbedderRealInference:
    def test_detects_and_embeds_faces(self, embedder: FaceEmbedder) -> None:
        image = cv2.imread(str(_ASSETS_DIR / "zidane.jpg"))
        assert image is not None
        faces = embedder.detect(image)
        assert len(faces) == 2
        for face in faces:
            assert face.embedding.shape == (512,)
            assert face.embedding.dtype == np.float32
            assert np.linalg.norm(face.embedding) == pytest.approx(1.0, abs=1e-3)

    def test_blank_image_has_no_faces(self, embedder: FaceEmbedder) -> None:
        blank = np.full((480, 640, 3), 128, dtype=np.uint8)
        assert embedder.detect(blank) == []


class TestRegistrationRoundTrip:
    def test_evaluate_accepts_single_face(self, embedder: FaceEmbedder, tmp_path: Path) -> None:
        store = _fast_store(tmp_path)
        store.initialize("test-pass")
        registrar = create_registrar(load_config(), store, embedder)
        result = registrar.evaluate(_single_face_image())
        assert result.num_faces == 1
        assert result.accepted, f"expected a clean single face, got: {result.reasons}"
        store.close()

    def test_enroll_and_read_back(self, embedder: FaceEmbedder, tmp_path: Path) -> None:
        store = _fast_store(tmp_path)
        store.initialize("test-pass")
        registrar = create_registrar(load_config(), store, embedder)

        result = registrar.evaluate(_single_face_image())
        assert result.face is not None
        person = registrar.enroll("Zidane (test)", [result.face])
        store.close()

        # Reopen the encrypted store and confirm the embedding survived the
        # encrypt -> disk -> decrypt round trip exactly.
        reopened = _fast_store(tmp_path)
        reopened.unlock("test-pass")
        embeddings = reopened.get_embeddings(person.id)
        assert len(embeddings) == 1
        assert embeddings[0].model_id == "insightface/buffalo_l"
        np.testing.assert_array_equal(embeddings[0].vector, result.face.embedding)
        reopened.close()
