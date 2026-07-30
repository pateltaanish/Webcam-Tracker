"""Per-track identity fusion (Stage 2.4).

Assigns registered-person identities to live tracks by combining face
recognition with temporal consistency:

  1. periodically (every N frames -- the face model is expensive) run the
     embedder on the frame to find + embed all faces;
  2. associate each face with the track whose box contains it;
  3. match the face against the enrolled profiles (FaceMatcher);
  4. keep a short rolling history of matches per track, and treat a track as
     "confirmed person X" only when a super-majority of that history agrees.

Step 4 is the point: a single mismatched frame can't flip a track's identity,
and -- critically -- a track is only *confirmed* as the target once its face
consistently says so. That confirmation is what makes reacquisition
identity-GATED (the state machine re-locks the target's returning track by
identity, not by "nearest to where we predicted"). When the face isn't visible
a track simply keeps its last identity (body Re-ID for face-away robustness is
a later slice).

Pure fusion logic: it depends on an embedder + matcher (injected) but has no
model/DB knowledge of its own, so it's testable with fakes.
"""

from __future__ import annotations

from collections import Counter, deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from webcam_tracker.face_recognition import FaceEmbedder, FaceMatcher
from webcam_tracker.logging_utils import get_logger
from webcam_tracker.tracking import TrackedPerson

logger = get_logger(__name__)


@dataclass(frozen=True)
class TrackIdentity:
    """A track's current fused identity."""

    track_id: int
    person_id: str | None  # top-voted person over the history window (may be unconfirmed)
    display_name: str | None
    confidence: float  # fraction of the window agreeing on person_id
    is_confirmed: bool  # confidence >= min_confidence and person_id is a real person


class IdentityTracker:
    def __init__(
        self,
        embedder: FaceEmbedder,
        matcher: FaceMatcher,
        update_every_n_frames: int,
        history_window: int,
        min_confidence: float,
    ) -> None:
        self._embedder = embedder
        self._matcher = matcher
        self._cadence = update_every_n_frames
        self._window = history_window
        self._min_confidence = min_confidence
        self._histories: dict[int, deque[str | None]] = {}
        self._names: dict[str, str] = {}
        self._frame = 0

    def update(self, image: np.ndarray, tracked_people: Sequence[TrackedPerson]) -> None:
        """Advance one frame. The heavy face model only actually runs every
        `update_every_n_frames`; other frames just prune vanished tracks."""
        self._frame += 1
        present = {t.track_id for t in tracked_people}

        if self._frame % self._cadence == 0:
            self._refresh(image, tracked_people)

        # Drop identity history for tracks that are no longer present (a
        # different physical person is a different track id, so stale history
        # must not linger and be re-locked onto).
        for track_id in list(self._histories):
            if track_id not in present:
                del self._histories[track_id]

    def _refresh(self, image: np.ndarray, tracked_people: Sequence[TrackedPerson]) -> None:
        faces = self._embedder.detect(image)
        for face in faces:
            track = self._containing_track(face.center, tracked_people)
            if track is None:
                continue
            result = self._matcher.match(face.embedding)
            person_id = result.person_id if result.is_match else None
            if person_id is not None and result.display_name is not None:
                self._names[person_id] = result.display_name
            self._histories.setdefault(track.track_id, deque(maxlen=self._window)).append(person_id)

    def identity_of(self, track_id: int) -> TrackIdentity | None:
        """The fused identity for a track, or None if we've never assessed it."""
        history = self._histories.get(track_id)
        if not history:
            return None
        person_id, votes = Counter(history).most_common(1)[0]
        confidence = votes / len(history)
        is_confirmed = person_id is not None and confidence >= self._min_confidence
        return TrackIdentity(
            track_id=track_id,
            person_id=person_id,
            display_name=self._names.get(person_id) if person_id is not None else None,
            confidence=confidence,
            is_confirmed=is_confirmed,
        )

    def stable_ids(self, track_ids: Iterable[int]) -> dict[int, str]:
        """track_id -> person_id for whichever of `track_ids` are CONFIRMED as
        a known person right now. Use this to key on-screen labels/colors by
        identity instead of the raw track_id: ByteTrack hands a returning
        person a brand-new track_id after any gap longer than
        `tracking.lost_track_buffer`, so anything keyed on track_id alone
        visibly changes at that exact moment even though the person is the
        same. Keying on the confirmed person_id instead means the displayed
        id/color stop changing the moment the face re-confirms who they are,
        which happens as soon as one identity refresh matches (see
        `identity_of`'s confidence == 1/1 case on the very first sample)."""
        stable: dict[int, str] = {}
        for track_id in track_ids:
            identity = self.identity_of(track_id)
            if identity is not None and identity.is_confirmed:
                assert identity.person_id is not None
                stable[track_id] = identity.person_id
        return stable

    def resolve_person(self, person_id: str) -> int | None:
        """The live track most confidently confirmed as `person_id`, or None.
        This is the identity-gated lock: only a track whose face consistently
        matches the person is returned."""
        best_track: int | None = None
        best_confidence = 0.0
        for track_id in self._histories:
            identity = self.identity_of(track_id)
            if (
                identity is not None
                and identity.is_confirmed
                and identity.person_id == person_id
                and identity.confidence > best_confidence
            ):
                best_track = track_id
                best_confidence = identity.confidence
        return best_track

    @staticmethod
    def _containing_track(
        point: tuple[float, float], tracked_people: Sequence[TrackedPerson]
    ) -> TrackedPerson | None:
        x, y = point
        candidates = [p for p in tracked_people if p.x1 <= x <= p.x2 and p.y1 <= y <= p.y2]
        if not candidates:
            return None
        return min(candidates, key=lambda p: p.width * p.height)  # smallest = frontmost
