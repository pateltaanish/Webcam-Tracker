"""Registration / enrollment (Stage 2.2).

Enrolls a consenting person into the encrypted identity store: capture face
samples, gate them on quality, embed them (ArcFace), and store them with a
consent record. Interactive capture lives in scripts/register_person.py; the
testable logic (quality gates + enrollment) lives here.
"""

from webcam_tracker.registration.factory import create_registrar
from webcam_tracker.registration.quality import (
    QualityReport,
    QualityThresholds,
    assess_face_quality,
)
from webcam_tracker.registration.registrar import Registrar, SampleEvaluation

__all__ = [
    "QualityReport",
    "QualityThresholds",
    "Registrar",
    "SampleEvaluation",
    "assess_face_quality",
    "create_registrar",
]
