"""Builds a FaceEmbedder from config."""

from __future__ import annotations

from webcam_tracker.config import AppConfig
from webcam_tracker.database import ProfileStore
from webcam_tracker.face_recognition.embedder import FaceEmbedder
from webcam_tracker.face_recognition.matcher import FaceMatcher


def create_face_embedder(config: AppConfig) -> FaceEmbedder:
    face = config.face
    return FaceEmbedder(
        model_pack=face.model_pack,
        det_size=face.det_size,
        device=face.device,
    )


def create_face_matcher(config: AppConfig, store: ProfileStore) -> FaceMatcher:
    """Build a matcher and load the enrolled templates from an unlocked store."""
    matcher = FaceMatcher(match_threshold=config.face.match_threshold)
    matcher.load(store)
    return matcher
