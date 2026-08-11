"""Builds an IdentityTracker from config + a live embedder and matcher.

The embedder (loaded model) and matcher (loaded from an unlocked store) are
runtime objects the caller owns; the factory only pulls the fusion parameters
from config.
"""

from __future__ import annotations

from webcam_tracker.config import AppConfig
from webcam_tracker.face_recognition import FaceEmbedder, FaceMatcher
from webcam_tracker.identity.tracker import IdentityTracker


def create_identity_tracker(
    config: AppConfig, embedder: FaceEmbedder, matcher: FaceMatcher
) -> IdentityTracker:
    tracking = config.identity_tracking
    return IdentityTracker(
        embedder=embedder,
        matcher=matcher,
        update_every_n_frames=tracking.update_every_n_frames,
        history_window=tracking.history_window,
        min_confidence=tracking.min_confidence,
        reacquire_grace_frames=tracking.reacquire_grace_frames,
        appearance_match_threshold=tracking.appearance_match_threshold,
        appearance_memory_frames=tracking.appearance_memory_frames,
    )
