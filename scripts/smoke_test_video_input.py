"""Headless smoke test: opens the configured source and reads a handful of
real frames, with no GUI window (safe to run non-interactively). This is
what Stage 1.2 "test it before adding detection" actually checks, distinct
from scripts/preview_webcam.py which is for a human to look at.
"""

from __future__ import annotations

from webcam_tracker.config import load_config
from webcam_tracker.logging_utils import configure_logging
from webcam_tracker.video_input import create_source

N_FRAMES = 15


def main() -> None:
    config = load_config()
    configure_logging(level=config.logging.level, json_format=False)

    source = create_source(config)
    print(f"video.source config = {config.video.source!r} -> {type(source).__name__}")

    with source:
        print(f"Negotiated fps: {source.fps:.2f}")
        for _ in range(N_FRAMES):
            frame = source.read()
            if frame is None:
                print("Source ended early.")
                break
            print(
                f"frame={frame.frame_index:<3} shape={frame.image.shape} "
                f"dtype={frame.image.dtype} timestamp={frame.timestamp:.3f}"
            )
    print("OK: video_input smoke test passed.")


if __name__ == "__main__":
    main()
