"""Person detection via YOLO11n (Ultralytics), filtered to the 'person' class.

License note: Ultralytics YOLO models are AGPL-3.0 licensed (or available
under a paid Ultralytics Enterprise license for closed-source distribution)
-- see docs/model_licenses.md. Fine for personal/research development; matters
if this project is ever distributed.

Why filter to 'person' at all: the underlying model is trained on all 80 COCO
classes. We only ever want people, so we look up whichever class ID that
model calls "person" (rather than hardcoding COCO's usual id=0) and pass that
to Ultralytics' own class filter -- this fails loudly if we ever swap in a
model that isn't COCO-trained, instead of silently detecting the wrong thing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.engine.results import Results

from webcam_tracker.logging_utils import get_logger

logger = get_logger(__name__)


class DetectorError(RuntimeError):
    """Raised when the detector model can't be loaded or run."""


@dataclass(frozen=True)
class Detection:
    """One detected person in a single frame, in pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1


def resolve_device(requested: str) -> str:
    """Resolve 'auto' | 'cpu' | 'cuda' | 'cuda:<index>' to a concrete device string."""
    if requested == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise DetectorError(
            f"Requested device '{requested}' but CUDA is not available on this machine."
        )
    return requested


def find_person_class_id(names: dict[int, str]) -> int:
    """Find the class ID the model itself calls 'person', rather than assuming one."""
    for class_id, name in names.items():
        if name == "person":
            return class_id
    raise DetectorError(
        f"Model has no 'person' class among {list(names.values())}. "
        "PersonDetector requires a COCO-trained (or equivalent) model."
    )


class PersonDetector:
    """Wraps a YOLO model, returning only person detections above threshold."""

    def __init__(
        self,
        model_path: Path,
        confidence_threshold: float,
        image_size: int = 640,
        device: str = "auto",
    ) -> None:
        self._model_path = model_path
        self._confidence_threshold = confidence_threshold
        self._image_size = image_size
        self._requested_device = device
        self._model: YOLO | None = None
        self._person_class_id: int | None = None
        self._resolved_device: str | None = None

    def load(self) -> None:
        """Load (downloading if necessary) and prepare the model for inference."""
        self._resolved_device = resolve_device(self._requested_device)
        self._model_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Loading detector model",
            extra={"model_path": str(self._model_path), "device": self._resolved_device},
        )
        model = YOLO(str(self._model_path))
        model.to(self._resolved_device)

        self._person_class_id = find_person_class_id(model.names)
        self._model = model
        logger.info(
            "Detector model loaded",
            extra={"person_class_id": self._person_class_id, "device": self._resolved_device},
        )

    def detect(self, image: np.ndarray) -> list[Detection]:
        """Run inference on one BGR frame, returning only person detections."""
        if self._model is None or self._person_class_id is None:
            raise DetectorError("detect() called before load()")

        # predict() without stream=True always returns list[Results] -- the
        # broader union in Ultralytics' stubs only applies to stream=True,
        # which we don't use here.
        results = cast(
            "list[Results]",
            self._model.predict(
                image,
                conf=self._confidence_threshold,
                classes=[self._person_class_id],
                imgsz=self._image_size,
                device=self._resolved_device,
                verbose=False,
            ),
        )

        boxes = results[0].boxes
        assert boxes is not None  # always populated for a detection-task model

        detections: list[Detection] = []
        for (x1, y1, x2, y2), confidence in zip(
            boxes.xyxy.tolist(), boxes.conf.tolist(), strict=True
        ):
            detections.append(Detection(x1=x1, y1=y1, x2=x2, y2=y2, confidence=float(confidence)))
        return detections

    @property
    def is_loaded(self) -> bool:
        return self._model is not None
