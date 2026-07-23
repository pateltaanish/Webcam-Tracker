"""Integration test: real face matching (Stage 2.3).

Enrolls one real person (a cropped face from a bundled image) into the
encrypted store, then confirms the matcher recognizes that same person from
the full image and returns UNKNOWN for a different person -- the whole
embed -> store -> match chain on the real model.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import ultralytics

from webcam_tracker.config import load_config
from webcam_tracker.database import KdfParams, ProfileStore
from webcam_tracker.face_recognition import FaceEmbedder, create_face_embedder, create_face_matcher
from webcam_tracker.registration import create_registrar

_ASSETS_DIR = Path(ultralytics.__file__).resolve().parent / "assets"


@pytest.fixture(scope="module")
def embedder() -> FaceEmbedder:
    emb = create_face_embedder(load_config())
    emb.load()
    return emb


def _full_image() -> np.ndarray:
    image = cv2.imread(str(_ASSETS_DIR / "zidane.jpg"))
    assert image is not None
    return image


def _single_face_image() -> np.ndarray:
    return _full_image()[0:400, 760:1280].copy()  # right side: one face


def _store(tmp_path: Path) -> ProfileStore:
    store = ProfileStore(
        tmp_path / "profiles.db",
        tmp_path / "keyvault.json",
        KdfParams(1, 8192, 1),
        min_passphrase_length=4,
    )
    store.initialize("test-pass")
    return store


def test_recognizes_enrolled_person_and_rejects_stranger(
    embedder: FaceEmbedder, tmp_path: Path
) -> None:
    config = load_config()
    store = _store(tmp_path)

    # Enroll the right-hand face.
    registrar = create_registrar(config, store, embedder)
    evaluation = registrar.evaluate(_single_face_image())
    assert evaluation.face is not None
    enrolled = registrar.enroll("Enrolled Person", [evaluation.face])

    matcher = create_face_matcher(config, store)
    assert matcher.num_templates == 1

    # In the full image, both people are present.
    faces = embedder.detect(_full_image())
    assert len(faces) == 2
    right = max(faces, key=lambda f: f.center[0])  # the enrolled person
    left = min(faces, key=lambda f: f.center[0])  # a different person

    enrolled_result = matcher.match(right.embedding)
    assert enrolled_result.is_match is True
    assert enrolled_result.person_id == enrolled.id
    assert enrolled_result.score >= config.face.match_threshold

    stranger_result = matcher.match(left.embedding)
    assert stranger_result.is_match is False  # different person -> UNKNOWN

    store.close()
