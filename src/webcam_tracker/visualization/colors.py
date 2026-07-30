"""Per-detection color assignment for on-screen visualization.

Until persistent track IDs exist (Stage 1.4), there's no real per-person
identity to color by -- this assigns colors by left-to-right screen position
instead. That's stable enough for people standing still, but will visibly
swap colors when two people cross paths, since position is all we have to go
on. That's expected, not a bug: it's a preview of exactly the identity-switch
problem persistent tracking (and later, identity verification) exists to
solve.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

from webcam_tracker.detection import Detection

# Bright, mutually distinguishable colors in OpenCV's BGR order. Reused
# cyclically if there are more detections than colors.
PALETTE: list[tuple[int, int, int]] = [
    (0, 255, 0),  # green
    (0, 165, 255),  # orange
    (255, 0, 0),  # blue
    (0, 0, 255),  # red
    (255, 255, 0),  # cyan
    (255, 0, 255),  # magenta
    (0, 255, 255),  # yellow
]


def assign_colors(detections: Sequence[Detection]) -> list[tuple[int, int, int]]:
    """Return one color per detection, same order as `detections`.

    Colors are assigned by rank in left-to-right horizontal position (not
    detection order, which isn't guaranteed stable frame-to-frame). Use this
    for raw detections, which have no persistent identity to color by --
    once you have TrackedPerson objects (post-tracking), use
    color_for_track_id instead for genuinely stable per-person colors.
    """
    order_by_position = sorted(range(len(detections)), key=lambda i: detections[i].center[0])
    colors: list[tuple[int, int, int]] = [(0, 0, 0)] * len(detections)
    for rank, original_index in enumerate(order_by_position):
        colors[original_index] = PALETTE[rank % len(PALETTE)]
    return colors


def color_for_track_id(track_id: int) -> tuple[int, int, int]:
    """Deterministic color for a track ID -- the same ID always gets the same
    color, across frames and across runs. Unlike assign_colors (position-based,
    for raw detections), this is a genuinely stable per-person color, since a
    track ID is a persistent identity, not a per-frame position.

    Note that a track ID itself is only stable while the same physical person
    stays continuously tracked -- if they leave the frame and come back after
    `tracking.lost_track_buffer` frames, ByteTrack assigns a brand-new track
    ID and this color changes with it. See `color_for_key` for a key that
    survives that (a registered person_id from `identity.IdentityTracker`)."""
    return PALETTE[track_id % len(PALETTE)]


def color_for_key(key: int | str) -> tuple[int, int, int]:
    """Deterministic color for any stable key -- a track_id (same as
    `color_for_track_id`), or, more durably across track-id changes, a
    registered person_id. Same key always gets the same color, across frames
    and across runs (string keys are hashed with a fixed algorithm, not
    Python's per-process-randomized `hash()`, so this holds across runs)."""
    if isinstance(key, int):
        index = key % len(PALETTE)
    else:
        digest = hashlib.sha256(key.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % len(PALETTE)
    return PALETTE[index]
