"""Lightweight body-appearance signature: an HSV color histogram over a
tracked box, compared by correlation.

This is the no-new-dependency first cut of Re-ID (docs/02_architecture.md
sec 3.4 selects OSNet/torchreid as the eventual, stronger embedder for the
same role -- swapping it in later only means replacing `embed`'s body, not
its callers, since both return a comparable fixed-length vector). Good
enough to bridge a short "track lost, then returns before the face is
visible again" gap using clothing color; not identity-grade on its own --
callers only use it as a supporting vote alongside face recognition, never
as a standalone identification.
"""

from __future__ import annotations

import cv2
import numpy as np

_HIST_BINS = [30, 32]  # hue, saturation
_HIST_RANGES = [0, 180, 0, 256]


def embed(image: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray | None:
    """Normalized HSV histogram of the box region, or None if the box has no
    pixels inside the image (fully out of frame)."""
    height, width = image.shape[:2]
    x1, y1 = max(0, int(box[0])), max(0, int(box[1]))
    x2, y2 = min(width, int(box[2])), min(height, int(box[3]))
    if x2 <= x1 or y2 <= y1:
        return None
    hsv = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, _HIST_BINS, _HIST_RANGES)
    cv2.normalize(hist, hist)
    return hist.flatten()


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Histogram correlation, -1..1 (1 = identical)."""
    return float(cv2.compareHist(a.astype(np.float32), b.astype(np.float32), cv2.HISTCMP_CORREL))
