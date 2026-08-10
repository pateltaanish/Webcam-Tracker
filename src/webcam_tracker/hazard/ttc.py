"""Time-to-collision from looming, plus left/center/right zoning.

Ported from the Watchdog wearable's vision/tracking.py, which ran the same
math on a Hailo-8L NPU. Nothing here ever touched Hailo -- the accelerator
only produced the boxes; this is plain numpy over a box history, so it runs
unchanged on the Pi's CPU detector. The drone build has no Hailo hat.

TTC comes from LOOMING: an object approaching the camera grows in the frame,
and the rate of that growth relative to current size is a time. No depth
sensor, no stereo, no camera intrinsics required -- which is exactly why it
survives the move off the NPU.

COORDINATE CONTRACT: every width/position here is NORMALIZED to 0-1 of the
frame, not pixels. The thresholds callers pass in (min_width, the zone
bounds) are fractions of frame width, so feeding pixels would silently pass
every guard rather than raise -- see HazardMonitor, which normalizes at the
boundary because webcam_tracker.detection.Detection is in pixels.
"""

from __future__ import annotations

from collections import deque

import numpy as np


class TrackStore:
    """Per-track history of (timestamp, normalized bbox width).

    Only width and time are kept: they're all compute_ttc needs, and position
    is re-read from the current frame's box rather than remembered.
    """

    def __init__(self, sample_window: int) -> None:
        self._sample_window = sample_window
        self._tracks: dict[int, deque[tuple[float, float]]] = {}

    def update(self, track_id: int, timestamp: float, width: float) -> deque[tuple[float, float]]:
        """Record one sample for a track and return its full history."""
        samples = self._tracks.setdefault(track_id, deque(maxlen=self._sample_window))
        samples.append((timestamp, width))
        return samples

    def prune(self, active_track_ids: set[int]) -> None:
        """Drop tracks the tracker no longer reports.

        Trackers recycle IDs, so a stale history left behind would be
        attributed to whatever new person inherits that ID -- and a spliced
        history of two different people produces a meaningless growth rate.
        """
        for track_id in list(self._tracks):
            if track_id not in active_track_ids:
                del self._tracks[track_id]


def compute_ttc(
    samples: deque[tuple[float, float]] | list[tuple[float, float]],
    *,
    min_samples: int,
    min_width: float,
    ttc_min: float,
    ttc_max: float,
) -> float | None:
    """Seconds until collision, estimated from bbox width growth.

    Returns None when the estimate would be garbage rather than guessing:
    too few samples, a box that isn't growing (steady or receding -- not
    approaching), or a box too small for its width to be measured reliably.
    A None here is not "no hazard"; the caller decides what to do with an
    unmeasurable-but-present object (see HazardMonitor's static fallback).

    The fit is against wall-clock timestamps, NOT frame indices, so the
    estimate is frame-rate independent -- the same approach yields the same
    TTC at 9 FPS on the Pi's CPU as at 24 FPS on Watchdog's NPU. What a
    lower frame rate costs is latency to the first estimate (min_samples
    frames) and fit quality over a longer arc, not correctness.

    KNOWN BIAS, measured, and it errs in the unsafe direction. The ratio
    width/(dwidth/dt) is EXACT for an approach at constant speed -- the error
    is entirely in estimating dwidth/dt with a straight line across a window
    over which width grows curved (~1/distance). A straight line recovers the
    window's average slope, which is below the true slope at its end, so TTC
    comes out too LARGE: hazards read as further away than they are. The bias
    scales with closure speed, over an 8-sample window at 10 FPS:

        true TTC 2.50s -> 2.87s  (1.15x)
        true TTC 1.00s -> 1.40s  (1.40x)
        true TTC 0.40s -> 0.87s  (2.18x)
        true TTC 0.25s -> 0.80s  (3.20x)

    So the fastest closures -- the ones with the least time to react -- are
    the most under-reported. Shrinking sample_window reduces it (less curve
    per window) at the cost of a noisier slope. Fitting the slope of log
    (width) instead would remove it outright, exactly rather than
    approximately, but shifts every TTC downward and so invalidates the
    inherited ttc_alert/ttc_urgent thresholds -- worth doing together with
    re-deriving those from drone footage, not before.
    """
    if len(samples) < min_samples:
        return None

    t = np.array([s[0] for s in samples])
    w = np.array([s[1] for s in samples])
    width = float(w[-1])

    if width < min_width:
        return None

    dw_dt = float(np.polyfit(t, w, 1)[0])
    if dw_dt <= 0:
        return None

    return float(np.clip(width / dw_dt, ttc_min, ttc_max))


def zone_from_cx(
    cx: float, *, discard_margin: float, left_max: float, right_min: float
) -> str | None:
    """Coarse bearing ("left"/"center"/"right") from a normalized box center x.

    None means discard this object entirely: it's close enough to the frame
    edge that lens distortion makes its apparent position (and its width,
    hence its TTC) untrustworthy, and it's usually half out of frame anyway.
    """
    if cx < discard_margin or cx > 1.0 - discard_margin:
        return None
    if cx < left_max:
        return "left"
    if cx > right_min:
        return "right"
    return "center"
