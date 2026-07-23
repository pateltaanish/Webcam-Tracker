"""Face recognition (Stage 2.2+).

Face detection (SCRFD) + embedding extraction (ArcFace) via InsightFace,
producing 512-d L2-normalized embeddings for identity matching. Stage 2.2 uses
this for registration (enrolling a consenting person's face); Stage 2.3 adds
comparison against the stored profiles with an explicit "unknown" outcome.

License note: InsightFace code is MIT, buffalo_l models are non-commercial --
see docs/model_licenses.md.
"""

from webcam_tracker.face_recognition.embedder import FaceEmbedder, FaceEmbedderError
from webcam_tracker.face_recognition.factory import create_face_embedder
from webcam_tracker.face_recognition.models import DetectedFace

__all__ = [
    "DetectedFace",
    "FaceEmbedder",
    "FaceEmbedderError",
    "create_face_embedder",
]
