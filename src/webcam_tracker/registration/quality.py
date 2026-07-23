"""Capture quality gates for registration (Stage 2.2).

Pure image analysis -- no model, no camera -- so it's fully unit-testable with
synthetic images. A captured face must pass all gates before it's embedded and
stored, so we don't enroll blurry, badly-lit, or too-small faces that would
make a poor matching template.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from webcam_tracker.face_recognition import DetectedFace


@dataclass(frozen=True)
class QualityThresholds:
    min_detection_score: float
    min_blur_variance: float
    min_brightness: float
    max_brightness: float
    min_face_fraction: float


@dataclass(frozen=True)
class QualityReport:
    """Result of assessing one captured face. `ok` is True only if `reasons`
    is empty."""

    ok: bool
    reasons: tuple[str, ...]
    det_score: float
    blur_variance: float
    brightness: float
    face_fraction: float


def assess_face_quality(
    image_bgr: np.ndarray, face: DetectedFace, thresholds: QualityThresholds
) -> QualityReport:
    """Score a single detected face against the quality gates."""
    height, width = image_bgr.shape[:2]
    frame_area = float(height * width)
    face_fraction = face.area / frame_area if frame_area > 0 else 0.0

    x1 = max(0, int(face.x1))
    y1 = max(0, int(face.y1))
    x2 = min(width, int(face.x2))
    y2 = min(height, int(face.y2))
    crop = image_bgr[y1:y2, x1:x2]

    if crop.size == 0:
        return QualityReport(
            ok=False,
            reasons=("face box falls outside the frame",),
            det_score=face.det_score,
            blur_variance=0.0,
            brightness=0.0,
            face_fraction=face_fraction,
        )

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())

    reasons: list[str] = []
    if face.det_score < thresholds.min_detection_score:
        reasons.append(f"low detector confidence ({face.det_score:.2f})")
    if face_fraction < thresholds.min_face_fraction:
        reasons.append(f"face too small/far ({face_fraction:.1%} of frame)")
    if blur_variance < thresholds.min_blur_variance:
        reasons.append(f"too blurry (sharpness {blur_variance:.0f})")
    if brightness < thresholds.min_brightness:
        reasons.append(f"too dark (brightness {brightness:.0f})")
    elif brightness > thresholds.max_brightness:
        reasons.append(f"too bright/overexposed (brightness {brightness:.0f})")

    return QualityReport(
        ok=not reasons,
        reasons=tuple(reasons),
        det_score=face.det_score,
        blur_variance=blur_variance,
        brightness=brightness,
        face_fraction=face_fraction,
    )
