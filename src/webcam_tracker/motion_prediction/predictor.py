"""Per-track motion estimation (Stage 1.8).

Maintains one KalmanFilter2D per active track_id and exposes each track's
smoothed position + inferred velocity as a TrackMotion. The main consumer is
the recovery module: when the target's track is lost, recovery reads its
last-known velocity from here to guess which way it went.

Pure logic -- no OpenCV, no camera. Time is taken from an injectable clock
(default time.monotonic) so tests can advance it deterministically, matching
the pattern used by AxisController and PerfMonitor.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from webcam_tracker.motion_prediction.kalman import KalmanFilter2D
from webcam_tracker.tracking import TrackedPerson


@dataclass(frozen=True)
class TrackMotion:
    """One track's smoothed motion estimate for a single frame."""

    track_id: int
    position: tuple[float, float]
    velocity: tuple[float, float]

    @property
    def speed(self) -> float:
        vx, vy = self.velocity
        return (vx * vx + vy * vy) ** 0.5

    def predict_position(self, seconds_ahead: float) -> tuple[float, float]:
        """Extrapolate position `seconds_ahead` under constant velocity."""
        return (
            self.position[0] + self.velocity[0] * seconds_ahead,
            self.position[1] + self.velocity[1] * seconds_ahead,
        )


class MotionPredictor:
    """Owns a Kalman filter per track and advances them each frame.

    Call `update(tracked_people)` once per frame with that frame's confirmed
    tracks; then `motion(track_id)` returns the estimate for any track still
    within its coast window (including one that just disappeared, which is the
    whole point -- recovery needs the last-known velocity of a vanished
    target).
    """

    def __init__(
        self,
        process_noise: float,
        measurement_noise: float,
        max_coast_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._process_noise = process_noise
        self._measurement_noise = measurement_noise
        self._max_coast_seconds = max_coast_seconds
        self._clock = clock

        self._filters: dict[int, KalmanFilter2D] = {}
        self._last_seen: dict[int, float] = {}
        self._last_update_time: float | None = None

    def update(self, tracked_people: Sequence[TrackedPerson]) -> None:
        """Advance every present track's filter by one frame, create filters
        for new tracks, and drop filters whose track hasn't been seen within
        `max_coast_seconds`."""
        now = self._clock()
        dt = 0.0 if self._last_update_time is None else max(now - self._last_update_time, 0.0)
        self._last_update_time = now

        for person in tracked_people:
            existing = self._filters.get(person.track_id)
            if existing is None:
                # New track: seed a filter at its current position (no predict
                # step -- there's no prior state to roll forward yet).
                self._filters[person.track_id] = KalmanFilter2D(
                    person.center, self._process_noise, self._measurement_noise
                )
            else:
                existing.predict(dt)
                existing.correct(person.center)
            self._last_seen[person.track_id] = now

        stale = [
            track_id
            for track_id, seen_at in self._last_seen.items()
            if now - seen_at > self._max_coast_seconds
        ]
        for track_id in stale:
            del self._filters[track_id]
            del self._last_seen[track_id]

    def motion(self, track_id: int) -> TrackMotion | None:
        """The motion estimate for `track_id`, or None if we have no live
        filter for it (never seen, or coasted past max_coast_seconds)."""
        kalman = self._filters.get(track_id)
        if kalman is None:
            return None
        return TrackMotion(track_id=track_id, position=kalman.position, velocity=kalman.velocity)

    def known_track_ids(self) -> set[int]:
        """Track IDs we currently hold a filter for (present or still coasting)."""
        return set(self._filters)
