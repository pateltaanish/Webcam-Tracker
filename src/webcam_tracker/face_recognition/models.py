"""Data model for a detected + embedded face (Stage 2.2)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DetectedFace:
    """One face found in a frame: its box, detector confidence, and the
    ArcFace embedding (512-d, L2-normalized) used for identity matching."""

    x1: float
    y1: float
    x2: float
    y2: float
    det_score: float
    embedding: np.ndarray  # float32, L2-normalized

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height
