"""Headless smoke test: captures a real frame from the configured video
source, runs the real person detector on it, draws the results, and saves
an annotated image to disk for visual review (no GUI window needed).

Run from the repo root:
    .venv\\Scripts\\python.exe scripts\\smoke_test_detection.py

Expected output: prints one line per detected person (bbox + confidence),
and writes data/debug_output/detection_smoke.jpg showing a colored box
around each detected person (color = left-to-right position, not a stable
per-person identity -- see webcam_tracker.visualization.assign_colors).
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2

from webcam_tracker.config import load_config
from webcam_tracker.detection import create_detector
from webcam_tracker.logging_utils import configure_logging
from webcam_tracker.video_input import create_source
from webcam_tracker.visualization import draw_detections

# Skip the first few frames so the webcam's auto-exposure/auto-focus has
# settled before we grab the frame we actually run detection on.
WARMUP_FRAMES = 10


def main() -> None:
    config = load_config()
    configure_logging(level=config.logging.level, json_format=False)

    detector = create_detector(config)
    detector.load()

    source = create_source(config)
    with source:
        for _ in range(WARMUP_FRAMES):
            frame = source.read()
        assert frame is not None

    start = time.monotonic()
    detections = detector.detect(frame.image)
    cold_elapsed_ms = (time.monotonic() - start) * 1000

    # First inference includes CUDA/cuDNN warmup; a few more calls on the
    # same frame show steady-state latency, which is what matters for FPS.
    warm_timings_ms = []
    for _ in range(5):
        warm_start = time.monotonic()
        detector.detect(frame.image)
        warm_timings_ms.append((time.monotonic() - warm_start) * 1000)
    avg_warm_ms = sum(warm_timings_ms) / len(warm_timings_ms)

    print(f"Ran detection on frame {frame.frame_index}, shape={frame.image.shape}")
    print(f"Cold inference time: {cold_elapsed_ms:.1f} ms")
    print(f"Warm inference time (avg of 5): {avg_warm_ms:.1f} ms (~{1000 / avg_warm_ms:.0f} FPS)")
    print(f"Detections: {len(detections)}")

    for detection in detections:
        x1, y1, x2, y2 = (int(v) for v in (detection.x1, detection.y1, detection.x2, detection.y2))
        print(f"  bbox=({x1},{y1},{x2},{y2}) confidence={detection.confidence:.3f}")

    annotated = draw_detections(frame.image.copy(), detections)

    output_dir = config.resolve_path("data/debug_output")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = Path(output_dir) / "detection_smoke.jpg"
    cv2.imwrite(str(output_path), annotated)
    print(f"Saved annotated frame to {output_path}")


if __name__ == "__main__":
    main()
