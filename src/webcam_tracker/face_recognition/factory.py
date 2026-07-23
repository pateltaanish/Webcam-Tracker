"""Builds a FaceEmbedder from config."""

from __future__ import annotations

from webcam_tracker.config import AppConfig
from webcam_tracker.face_recognition.embedder import FaceEmbedder


def create_face_embedder(config: AppConfig) -> FaceEmbedder:
    face = config.face
    return FaceEmbedder(
        model_pack=face.model_pack,
        det_size=face.det_size,
        device=face.device,
    )
