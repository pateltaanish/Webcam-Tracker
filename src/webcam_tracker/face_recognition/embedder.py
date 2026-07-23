"""Face detection + embedding via InsightFace (Stage 2.2).

Wraps InsightFace's `FaceAnalysis` (SCRFD detector + ArcFace embedder, the
`buffalo_l` pack) to turn a BGR frame into a list of DetectedFace, each with a
512-d L2-normalized embedding for identity matching.

License note: the InsightFace *code* is MIT, but the pretrained `buffalo_l`
*models* are non-commercial/research use only -- see docs/model_licenses.md.

We deliberately load only the detection + recognition sub-models
(`allowed_modules=['detection', 'recognition']`), skipping the pack's
gender/age model -- we neither need nor want to infer those.

InsightFace is imported lazily inside `load()` so importing this module (and
the package) doesn't require the heavy optional dependency unless face
recognition is actually used.
"""

from __future__ import annotations

import numpy as np

from webcam_tracker.face_recognition.models import DetectedFace
from webcam_tracker.logging_utils import get_logger

logger = get_logger(__name__)


class FaceEmbedderError(RuntimeError):
    """Raised when the face model can't be loaded or run."""


class FaceEmbedder:
    """Detects faces and extracts ArcFace embeddings. Call `load()` once before
    `detect()` (loading downloads/initializes the model, which is slow)."""

    def __init__(self, model_pack: str, det_size: int, device: str) -> None:
        self._model_pack = model_pack
        self._det_size = det_size
        self._device = device
        self._model_id = f"insightface/{model_pack}"
        self._app: object | None = None

    @property
    def model_id(self) -> str:
        """Identifier stored alongside each embedding, so a later match knows
        which model produced it (embeddings from different models aren't
        comparable)."""
        return self._model_id

    def load(self) -> None:
        try:
            from insightface.app import FaceAnalysis
        except ImportError as exc:  # pragma: no cover - exercised only without the optional dep
            raise FaceEmbedderError(
                "insightface is not installed; run `pip install -r requirements-identity.txt`"
            ) from exc

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if self._device == "cuda"
            else ["CPUExecutionProvider"]
        )
        ctx_id = 0 if self._device == "cuda" else -1
        app = FaceAnalysis(
            name=self._model_pack,
            allowed_modules=["detection", "recognition"],
            providers=providers,
        )
        app.prepare(ctx_id=ctx_id, det_size=(self._det_size, self._det_size))
        self._app = app
        logger.info("Face embedder loaded", extra={"model": self._model_id, "device": self._device})

    def detect(self, image: np.ndarray) -> list[DetectedFace]:
        """Detect and embed every face in a BGR image."""
        if self._app is None:
            raise FaceEmbedderError("call load() before detect()")
        faces = self._app.get(image)  # type: ignore[attr-defined]
        results: list[DetectedFace] = []
        for face in faces:
            x1, y1, x2, y2 = (float(v) for v in face.bbox)
            embedding = np.asarray(face.normed_embedding, dtype=np.float32)
            results.append(
                DetectedFace(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    det_score=float(face.det_score),
                    embedding=embedding,
                )
            )
        return results
