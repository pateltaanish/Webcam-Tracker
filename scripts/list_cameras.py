"""Enumerate working camera indices on this machine.

Run this once on any new machine (desktop with a USB webcam, laptop with a
built-in webcam, ...) to see which device index(es) actually work and at
what resolution/fps -- useful when `video.source: "auto"` in
configs/default.yaml picks the wrong camera on a machine with more than one,
or just to sanity-check a camera is detected at all before running the rest
of the pipeline.

Run from the repo root:
    .venv\\Scripts\\python.exe scripts\\list_cameras.py

Successful output looks like a table with at least one "WORKING" row. If
every row says "not found", the camera isn't reachable by OpenCV at all --
check it isn't in use by another application (e.g. a video-call app) and
that Windows camera privacy settings allow desktop apps to use it
(Settings > Privacy & security > Camera).
"""

from __future__ import annotations

import argparse

from webcam_tracker.video_input import BACKEND_NAMES, probe_cameras


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-probe", type=int, default=10, help="Highest device index to probe (default: 10)"
    )
    args = parser.parse_args()

    print(f"Probing webcam indices 0..{args.max_probe - 1} ...\n")
    print(f"{'Index':<6} {'Status':<12} {'Backend':<14} {'Resolution':<12} {'FPS':<6}")
    print("-" * 54)

    results = probe_cameras(args.max_probe)
    for result in results:
        if not result.working:
            print(f"{result.index:<6} {'not found':<12}")
            continue
        assert result.backend is not None  # guaranteed by working=True
        backend_name = BACKEND_NAMES.get(result.backend, str(result.backend))
        resolution = f"{result.width}x{result.height}"
        fps_str = f"{result.fps:.1f}" if result.fps is not None else "n/a"
        print(f"{result.index:<6} {'WORKING':<12} {backend_name:<14} {resolution:<12} {fps_str}")

    print()
    if any(r.working for r in results):
        print(
            "To pin a specific camera instead of 'auto', set in a local .env "
            "(copy .env.example first):\n"
            "  WEBCAM_TRACKER_VIDEO__SOURCE=<index>"
        )
    else:
        print(
            "No working camera found. Check it's not in use by another app and "
            "that camera access is allowed for desktop apps in Windows privacy settings."
        )


if __name__ == "__main__":
    main()
