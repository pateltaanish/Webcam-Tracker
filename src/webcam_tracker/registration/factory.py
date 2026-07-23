"""Builds a Registrar from config + a live store and embedder.

The store (unlocked) and embedder (loaded) are runtime objects the caller owns
-- the factory only pulls the quality thresholds + consent version from config.
"""

from __future__ import annotations

from webcam_tracker.config import AppConfig
from webcam_tracker.database import ProfileStore
from webcam_tracker.face_recognition import FaceEmbedder
from webcam_tracker.registration.quality import QualityThresholds
from webcam_tracker.registration.registrar import Registrar


def create_registrar(config: AppConfig, store: ProfileStore, embedder: FaceEmbedder) -> Registrar:
    reg = config.registration
    thresholds = QualityThresholds(
        min_detection_score=reg.min_detection_score,
        min_blur_variance=reg.min_blur_variance,
        min_brightness=reg.min_brightness,
        max_brightness=reg.max_brightness,
        min_face_fraction=reg.min_face_fraction,
    )
    return Registrar(
        store=store,
        embedder=embedder,
        thresholds=thresholds,
        consent_version=reg.consent_version,
    )
