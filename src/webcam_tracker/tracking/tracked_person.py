"""The tracked-person data model."""

from __future__ import annotations

from dataclasses import dataclass

# The underlying tracker (ByteTrack, via the `trackers` package) reports this
# sentinel for a track that hasn't yet matched enough consecutive frames to be
# considered "confirmed" (see TrackingConfig.minimum_consecutive_frames).
# PersonTracker filters these out before they ever reach a TrackedPerson.
# Note: a track is NEVER confirmed on the very frame it's first detected,
# even with minimum_consecutive_frames=1 -- it always takes at least one
# more matching frame after that, so a new person takes a minimum of 2
# total frames to first appear as a TrackedPerson.
UNCONFIRMED_TRACK_ID = -1


@dataclass(frozen=True)
class TrackedPerson:
    """One tracked person in a single frame: a Detection plus a persistent ID.

    The same real person keeps the same track_id across frames (that's the
    entire point of tracking, vs. detection's independent-per-frame boxes) --
    for as long as the tracker can keep matching them. track_id is *not* a
    claim about who the person is; identity verification is a separate,
    later concern (Stage 2).
    """

    track_id: int
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
