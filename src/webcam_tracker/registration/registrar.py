"""Registration orchestration (Stage 2.2).

Ties together the face embedder, the quality gates, and the encrypted store to
enroll a consenting person. The interactive capture loop + operator prompts
live in scripts/register_person.py; the testable logic -- evaluate one captured
frame, and commit a set of accepted samples as a new profile -- lives here.

Enrollment is deliberately single-person: a frame with zero or more than one
face is rejected, so we never accidentally mix two people into one profile.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from webcam_tracker.database import Person, ProfileStore
from webcam_tracker.face_recognition import DetectedFace, FaceEmbedder
from webcam_tracker.logging_utils import get_logger
from webcam_tracker.registration.quality import (
    QualityReport,
    QualityThresholds,
    assess_face_quality,
)

logger = get_logger(__name__)


@dataclass(frozen=True)
class SampleEvaluation:
    """Verdict on one captured frame."""

    accepted: bool
    reasons: tuple[str, ...]
    num_faces: int
    face: DetectedFace | None
    quality: QualityReport | None


class Registrar:
    def __init__(
        self,
        store: ProfileStore,
        embedder: FaceEmbedder,
        thresholds: QualityThresholds,
        consent_version: str,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._thresholds = thresholds
        self._consent_version = consent_version

    def evaluate(self, image: np.ndarray) -> SampleEvaluation:
        """Detect faces in a frame and, if there's exactly one, run the quality
        gates on it. Does NOT store anything -- the caller collects accepted
        samples and calls enroll() once it has enough."""
        faces = self._embedder.detect(image)
        if len(faces) == 0:
            return SampleEvaluation(False, ("no face detected",), 0, None, None)
        if len(faces) > 1:
            return SampleEvaluation(
                False,
                ("more than one face in frame -- register one person, alone",),
                len(faces),
                None,
                None,
            )
        face = faces[0]
        report = assess_face_quality(image, face, self._thresholds)
        return SampleEvaluation(report.ok, report.reasons, 1, face, report)

    def enroll(
        self,
        display_name: str,
        faces: Sequence[DetectedFace],
        consent_note: str = "",
    ) -> Person:
        """Create a profile from accepted face samples. The store records the
        'granted' consent event and audit entries; this adds one more audit
        line noting the registration and sample count."""
        if not faces:
            raise ValueError("cannot enroll with zero face samples")
        person = self._store.add_person(display_name, self._consent_version, consent_note)
        for face in faces:
            self._store.add_embedding(
                person.id,
                kind="face",
                model_id=self._embedder.model_id,
                vector=face.embedding,
                quality=face.det_score,
            )
        self._store.audit("person_registered", f"{person.id}:{len(faces)}_face_samples")
        logger.info(
            "Person registered",
            extra={"person_id": person.id, "samples": len(faces), "model": self._embedder.model_id},
        )
        return person
