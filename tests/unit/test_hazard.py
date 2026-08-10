"""Tests for looming-based hazard detection (ported from the Watchdog wearable).

The pixel-vs-normalized tests below are the important ones: feeding pixel
coordinates to logic whose thresholds are frame fractions fails silently
rather than raising, so only a test catches it.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict

import pytest

from webcam_tracker.config.settings import HazardConfig
from webcam_tracker.hazard import Hazard, HazardMonitor, TrackStore, compute_ttc, zone_from_cx
from webcam_tracker.tracking import TrackedPerson

FRAME_W, FRAME_H = 640, 480

TTC_KWARGS = {"min_samples": 5, "min_width": 0.03, "ttc_min": 0.2, "ttc_max": 30.0}
ZONE_KWARGS = {"discard_margin": 0.05, "left_max": 0.33, "right_min": 0.67}


def _config(**overrides) -> HazardConfig:  # type: ignore[no-untyped-def]
    base = {
        "min_samples": 5,
        "sample_window": 8,
        "min_width_fraction": 0.03,
        "ttc_min": 0.2,
        "ttc_max": 30.0,
        "zone_discard_margin": 0.05,
        "zone_left_max": 0.33,
        "zone_right_min": 0.67,
        "ttc_alert": 3.0,
        "ttc_urgent": 1.0,
        "track_cooldown": 4.0,
        "static_cooldown": 8.0,
        "global_gap": 1.2,
        "static_bottom_y": 0.75,
    }
    return HazardConfig(**{**base, **overrides})


def _person(track_id: int, *, cx_frac: float, width_frac: float, bottom_frac: float = 0.5):
    """A TrackedPerson in PIXELS, described in frame fractions."""
    half = width_frac * FRAME_W / 2.0
    cx = cx_frac * FRAME_W
    return TrackedPerson(
        track_id=track_id,
        x1=cx - half,
        y1=0.0,
        x2=cx + half,
        y2=bottom_frac * FRAME_H,
        confidence=0.9,
    )


# --- TTC math -------------------------------------------------------------


def test_ttc_falls_as_an_object_actually_approaches() -> None:
    """Perspective looming: width ~ k/distance, closing at constant speed.

    This is what a real approach looks like. A naive linear width increase
    would give a constant dw/dt and therefore a RISING ttc, so this shape
    is what distinguishes the physics from a plausible-looking stub.
    """
    store = TrackStore(sample_window=8)
    distance0, speed = 10.0, 1.0
    ttcs = []
    for i in range(10):
        t = i / 30
        distance = distance0 - speed * t
        store.update(1, t, 1.0 / distance)
        ttcs.append(compute_ttc(store._tracks[1], **TTC_KWARGS))
        real_ttc = distance / speed

    assert ttcs[:4] == [None] * 4  # min_samples=5 not reached yet
    assert all(v is not None for v in ttcs[4:])
    assert ttcs[-1] < ttcs[5]  # shrinking as it closes
    assert abs(ttcs[-1] - real_ttc) < 1.0  # estimate lands near the truth


def test_ttc_is_none_for_a_receding_object() -> None:
    store = TrackStore(sample_window=8)
    width = 0.2
    for i in range(10):
        store.update(2, i / 30, width)
        width -= 0.01
    assert compute_ttc(store._tracks[2], **TTC_KWARGS) is None


def test_ttc_is_none_for_a_box_too_small_to_measure() -> None:
    store = TrackStore(sample_window=8)
    width = 0.001
    for i in range(10):
        store.update(3, i / 30, width)
        width *= 1.05  # growing fast, but still far below min_width
    assert compute_ttc(store._tracks[3], **TTC_KWARGS) is None


def test_ttc_is_frame_rate_independent() -> None:
    """The same approach at 9 FPS and 24 FPS yields the same TTC.

    This is what makes the port off the NPU viable at all: the fit is against
    wall-clock timestamps, so the Pi's slower CPU detector changes latency,
    not the estimate.
    """
    estimates = []
    for fps in (9.0, 24.0):
        store = TrackStore(sample_window=8)
        ttc = None
        for i in range(8):
            t = i / fps
            store.update(1, t, 1.0 / (10.0 - 1.0 * t))
            ttc = compute_ttc(store._tracks[1], **TTC_KWARGS)
        estimates.append(ttc)
    assert abs(estimates[0] - estimates[1]) < 0.5


def test_track_store_prune_drops_stale_ids() -> None:
    """Trackers recycle IDs; a stale history would splice two people together."""
    store = TrackStore(sample_window=8)
    store.update(1, 0.0, 0.1)
    store.update(2, 0.0, 0.1)
    store.prune({1})
    assert set(store._tracks) == {1}


def test_zones_and_edge_discard() -> None:
    assert zone_from_cx(0.02, **ZONE_KWARGS) is None  # lens distortion, discard
    assert zone_from_cx(0.98, **ZONE_KWARGS) is None
    assert zone_from_cx(0.2, **ZONE_KWARGS) == "left"
    assert zone_from_cx(0.5, **ZONE_KWARGS) == "center"
    assert zone_from_cx(0.8, **ZONE_KWARGS) == "right"


# --- the pixel/normalized boundary ---------------------------------------


def test_monitor_normalizes_pixels_before_zoning() -> None:
    """A person centered in a 640px frame is 'center', not 'right'.

    Pixel cx (320) sails past zone_right_min (0.67) without raising, so this
    is the test that catches a missing normalization.
    """
    monitor = HazardMonitor(_config())
    hazard = monitor.update(
        [_person(1, cx_frac=0.5, width_frac=0.1, bottom_frac=0.9)], FRAME_W, FRAME_H, now=0.0
    )
    assert hazard is not None
    assert hazard.zone == "center"


def test_monitor_normalizes_pixels_before_min_width() -> None:
    """A 6px-wide box is below min_width_fraction and must not produce a TTC.

    In pixels it would be 6.0 >= 0.03 and pass -- and then loom convincingly.
    """
    monitor = HazardMonitor(_config())
    hazard = None
    for i in range(10):
        # tiny but growing fast: real looming, far too small to trust
        people = [_person(1, cx_frac=0.5, width_frac=0.002 * (1.1**i), bottom_frac=0.5)]
        hazard = monitor.update(people, FRAME_W, FRAME_H, now=i / 10.0)
    assert hazard is None


def test_monitor_rejects_a_zero_size_frame() -> None:
    monitor = HazardMonitor(_config())
    with pytest.raises(ValueError, match="frame size must be positive"):
        monitor.update([], 0, FRAME_H, now=0.0)


# --- arbitration ----------------------------------------------------------


GROWTH = 0.4  # width ~ e^(0.4t) -> a steady approach at a constant TTC of 1/0.4 = 2.5s


def _approach(
    monitor: HazardMonitor,
    seconds: float,
    *,
    cx_frac: float = 0.5,
    fps: float = 10.0,
    growth: float = GROWTH,
):
    """Feed ONE continuous approach and return every (time, hazard) reported.

    Continuous on purpose: the box history is a rolling window, so replaying
    disjoint bursts with time gaps between them would splice samples across
    the gap and flatten the measured growth rate -- an artifact of the test,
    not of the code. Real tracks are pruned when they leave frame.
    """
    fired = []
    for i in range(int(seconds * fps)):
        t = i / fps
        width = 0.04 * math.exp(growth * t)
        hazard = monitor.update(
            [_person(1, cx_frac=cx_frac, width_frac=width)], FRAME_W, FRAME_H, now=t
        )
        if hazard is not None:
            fired.append((t, hazard))
    return fired


def test_closing_person_is_reported_with_zone_and_ttc() -> None:
    monitor = HazardMonitor(_config())
    fired = _approach(monitor, seconds=2.0, cx_frac=0.2)
    assert fired
    _, hazard = fired[0]
    assert hazard.track_id == 1
    assert hazard.zone == "left"
    assert hazard.ttc is not None and hazard.ttc <= 3.0
    assert not hazard.is_static
    # constant-TTC approach: the estimate should land near 1/GROWTH
    assert abs(hazard.ttc - 1.0 / GROWTH) < 0.5


def test_static_fallback_fires_for_a_close_but_unmeasurable_person() -> None:
    """No looming (steady width) but filling the bottom of frame."""
    monitor = HazardMonitor(_config())
    person = _person(1, cx_frac=0.5, width_frac=0.3, bottom_frac=0.8)
    hazard = monitor.update([person], FRAME_W, FRAME_H, now=0.0)
    assert hazard is not None
    assert hazard.is_static
    assert hazard.ttc is None
    assert not hazard.urgent  # never urgent: distance is known, closing rate isn't


def test_static_person_high_in_frame_is_not_a_hazard() -> None:
    monitor = HazardMonitor(_config())
    person = _person(1, cx_frac=0.5, width_frac=0.3, bottom_frac=0.5)
    assert monitor.update([person], FRAME_W, FRAME_H, now=0.0) is None


def test_a_real_closing_hazard_outranks_a_static_one() -> None:
    """Even a static hazard filling the frame loses to a measured approach.

    We know one is closing and only that the other is near.
    """
    monitor = HazardMonitor(_config(global_gap=0.01, track_cooldown=0.01))
    static = _person(9, cx_frac=0.5, width_frac=0.3, bottom_frac=0.9)
    hazard = None
    for i in range(10):
        t = i / 10.0
        closing = _person(1, cx_frac=0.2, width_frac=0.04 * math.exp(GROWTH * t))
        got = monitor.update([static, closing], FRAME_W, FRAME_H, now=t)
        if got is not None and got.ttc is not None:
            hazard = got
    assert hazard is not None
    assert hazard.track_id == 1  # the closing one, not the close-but-static one


def test_soonest_collision_wins_among_several() -> None:
    monitor = HazardMonitor(_config())
    near = Hazard(1, "center", 0.5, urgent=True)
    far = Hazard(2, "left", 2.5, urgent=False)
    assert monitor._rank([far, near]) is near


def test_same_track_repeats_only_once_per_cooldown() -> None:
    """A person closing for 12s straight must not fire on all 120 frames."""
    monitor = HazardMonitor(_config())
    fired = _approach(monitor, seconds=12.0)
    assert len(fired) >= 2, "a 12s approach should re-report at least once"
    times = [t for t, _ in fired]
    gaps = [b - a for a, b in zip(times, times[1:], strict=False)]
    assert all(gap >= 4.0 for gap in gaps), f"track_cooldown violated: {gaps}"


def test_global_gap_suppresses_a_second_track_immediately_after() -> None:
    monitor = HazardMonitor(_config())
    fired = _approach(monitor, seconds=0.6, cx_frac=0.3)
    assert fired
    # a DIFFERENT track, with its own cooldown clear, still inside global_gap
    hazard = monitor.update(
        [_person(2, cx_frac=0.6, width_frac=0.3, bottom_frac=0.9)],
        FRAME_W,
        FRAME_H,
        now=fired[-1][0] + 0.1,
    )
    assert hazard is None


def test_monitor_derives_urgent_from_the_configured_threshold() -> None:
    """A fast closure is flagged urgent, an identical slow one is not.

    Both runs go through monitor.update, so this checks the comparison against
    ttc_urgent rather than that the dataclass stores the flag it was handed.
    """
    fast = HazardMonitor(_config())
    slow = HazardMonitor(_config())
    # Reported TTC, not true TTC: the linear fit reads high on fast closures
    # (see compute_ttc's KNOWN BIAS table). growth 2.5 reports ~0.87s (urgent),
    # growth 0.4 reports ~2.87s (not).
    fast_fired = _approach(fast, seconds=2.0, growth=2.5)
    slow_fired = _approach(slow, seconds=2.0, growth=0.4)

    assert fast_fired and slow_fired
    assert fast_fired[0][1].ttc < 1.0 and fast_fired[0][1].urgent
    assert slow_fired[0][1].ttc > 1.0 and not slow_fired[0][1].urgent


def test_static_report_does_not_swallow_a_later_urgent_one() -> None:
    """A person entering low in frame then looming must still raise urgent.

    The first frames have too few samples for a TTC, so the static fallback
    reports first. If that report's cooldown also gated the measured hazard,
    the urgent one would be suppressed for track_cooldown seconds -- most of
    the time available to react to it.
    """
    monitor = HazardMonitor(_config())
    reports = []
    for i in range(20):
        t = i / 10.0
        # low in frame from the start, and looming fast (reported ttc ~0.87s)
        person = _person(
            1, cx_frac=0.5, width_frac=0.04 * math.exp(2.5 * t), bottom_frac=0.9
        )
        hazard = monitor.update([person], FRAME_W, FRAME_H, now=t)
        if hazard is not None:
            reports.append((t, hazard))

    assert reports, "nothing reported at all"
    assert reports[0][1].is_static, "expected the static fallback to report first"
    urgent = [(t, h) for t, h in reports if h.urgent]
    assert urgent, f"urgent hazard never escaped the static cooldown: {reports}"
    # and it must arrive promptly, not after track_cooldown (4.0s) elapsed
    assert urgent[0][0] < 2.0


def test_a_measured_report_still_rate_limits_its_own_repeats() -> None:
    """Escalation is allowed once; repetition is not."""
    monitor = HazardMonitor(_config())
    fired = _approach(monitor, seconds=12.0, growth=2.5)
    moving_times = [t for t, h in fired if not h.is_static]
    gaps = [b - a for a, b in zip(moving_times, moving_times[1:], strict=False)]
    assert all(gap >= 4.0 for gap in gaps), f"track_cooldown violated: {gaps}"


def test_no_people_reports_nothing() -> None:
    monitor = HazardMonitor(_config())
    assert monitor.update([], FRAME_W, FRAME_H, now=0.0) is None


# --- config guards --------------------------------------------------------


def test_sample_window_below_min_samples_is_rejected() -> None:
    """Silently disables TTC forever otherwise -- the deque caps below the floor."""
    with pytest.raises(ValueError, match="sample_window"):
        _config(sample_window=3, min_samples=5)


def test_urgent_above_alert_is_rejected() -> None:
    with pytest.raises(ValueError, match="ttc_urgent"):
        _config(ttc_urgent=5.0, ttc_alert=3.0)


def test_inverted_zone_bounds_are_rejected() -> None:
    with pytest.raises(ValueError, match="zone_left_max"):
        _config(zone_left_max=0.8, zone_right_min=0.2)


# --- the JSON contract consumed by the streaming dashboard -----------------


def test_hazard_serializes_to_exactly_the_fields_the_dashboard_reads() -> None:
    """scripts/preview_tracking.py does json.dumps(asdict(hazard)) onto an SSE
    channel; the built-in dashboard's JS reads track_id/zone/ttc/urgent off
    the parsed object. Nothing exercises that chain in a live run unless a
    hazard actually fires, so a field rename here would break the dashboard
    silently -- this is the test that would catch it.

    is_static is a property, not a field, so asdict() omits it on purpose:
    the dashboard derives "static" from ttc being null rather than needing a
    fifth key.
    """
    hazard = Hazard(track_id=7, zone="left", ttc=2.5, urgent=False)
    round_tripped = json.loads(json.dumps(asdict(hazard)))
    assert round_tripped == {"track_id": 7, "zone": "left", "ttc": 2.5, "urgent": False}

    static = Hazard(track_id=8, zone="right", ttc=None, urgent=False)
    assert json.loads(json.dumps(asdict(static)))["ttc"] is None
