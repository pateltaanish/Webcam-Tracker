"""Per-track identity fusion (Stage 2.4).

Assigns registered-person identities to live tracks by combining face
recognition with temporal consistency:

  1. run the embedder immediately when a new track appears, then periodically
     (every N frames -- the face model is expensive) to find + embed all faces;
  2. associate each face with the track whose box contains it;
  3. match the face against the enrolled profiles (FaceMatcher);
  4. keep a short rolling history of matches per track, and treat a track as
     "confirmed person X" only when a super-majority of that history agrees.

Step 4 is the point: a single mismatched frame can't flip a track's identity,
and -- critically -- a track is only *confirmed* as the target once its face
consistently says so. That confirmation is what makes reacquisition
identity-GATED (the state machine re-locks the target's returning track by
identity, not by "nearest to where we predicted"). When the face isn't visible
but the track never dropped, a track simply keeps its last identity (no new
evidence needed -- see `update`'s docstring). When the TRACK is lost and a new
one appears before the face is visible again (walked back in facing away),
step 3 falls back to a body-appearance vote (`webcam_tracker.reid`) against
whichever people this session has already face-confirmed recently -- it never
extends recognition to anyone who wasn't already face-matched first.

Pure fusion logic: it depends on an embedder + matcher (injected) but has no
model/DB knowledge of its own, so it's testable with fakes.
"""

from __future__ import annotations

from collections import Counter, deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from webcam_tracker import reid
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
        reacquire_grace_frames: int = 10,
        appearance_match_threshold: float = 0.75,
        appearance_memory_frames: int = 90,
    ) -> None:
        self._embedder = embedder
        self._matcher = matcher
        self._cadence = update_every_n_frames
        self._window = history_window
        self._min_confidence = min_confidence
        self._grace = reacquire_grace_frames
        self._appearance_threshold = appearance_match_threshold
        self._appearance_memory = appearance_memory_frames
        self._histories: dict[int, deque[str | None]] = {}
        self._names: dict[str, str] = {}
        self._present_track_ids: set[int] = set()
        self._frames_since_refresh = 0
        self._grace_remaining: dict[int, int] = {}
        # Last-seen body-appearance signature per person, kept fresh while
        # they're face-confirmed and visible, aged out `appearance_memory_frames`
        # after they're last seen -- a short-lived, session-only memory, not a
        # persisted template (see `update`/`_refresh` docstrings).
        self._appearance: dict[str, np.ndarray] = {}
        self._appearance_age: dict[str, int] = {}

    def update(self, image: np.ndarray, tracked_people: Sequence[TrackedPerson]) -> None:
        """Advance one frame.

        New tracks bypass the normal cadence so a returning person can be
        identified and re-locked as soon as ByteTrack reports them. That
        first-frame attempt can miss (motion blur, an off-angle face as
        someone walks back into frame), so a still-unconfirmed new track
        keeps forcing a refresh every frame -- not just once -- for up to
        `reacquire_grace_frames` frames, instead of falling back to the slow
        periodic cadence and leaving a stale/raw-track-id label on screen for
        up to `update_every_n_frames` frames. Once confirmed (or the grace
        window lapses), it drops back to the normal cadence. Existing,
        already-confirmed tracks always use the periodic cadence.
        """
        self._frames_since_refresh += 1
        present = {t.track_id for t in tracked_people}
        for track_id in present - self._present_track_ids:
            self._grace_remaining[track_id] = self._grace

        forcing = any(
            remaining > 0 and not self._is_confirmed(track_id)
            for track_id, remaining in self._grace_remaining.items()
        )

        if tracked_people and (forcing or self._frames_since_refresh >= self._cadence):
            self._refresh(image, tracked_people)
            self._frames_since_refresh = 0

        for track_id in present:
            if self._grace_remaining.get(track_id, 0) > 0:
                self._grace_remaining[track_id] -= 1

        # Drop identity history for tracks that are no longer present (a
        # different physical person is a different track id, so stale history
        # must not linger and be re-locked onto).
        for track_id in list(self._histories):
            if track_id not in present:
                del self._histories[track_id]
        for track_id in list(self._grace_remaining):
            if track_id not in present:
                del self._grace_remaining[track_id]
        self._present_track_ids = present

        # Age out appearance memory every frame (not just refresh frames) so
        # it decays in real time regardless of the cadence -- a person not
        # reconfirmed within `appearance_memory_frames` stops being an
        # appearance-fallback candidate.
        for person_id in list(self._appearance_age):
            self._appearance_age[person_id] += 1
            if self._appearance_age[person_id] > self._appearance_memory:
                del self._appearance[person_id]
                del self._appearance_age[person_id]

    def _is_confirmed(self, track_id: int) -> bool:
        identity = self.identity_of(track_id)
        return identity is not None and identity.is_confirmed

    def _refresh(self, image: np.ndarray, tracked_people: Sequence[TrackedPerson]) -> None:
        faces = self._embedder.detect(image)
        matched_track_ids: set[int] = set()
        for face in faces:
            track = self._containing_track(face.center, tracked_people)
            if track is None:
                continue
            matched_track_ids.add(track.track_id)
            result = self._matcher.match(face.embedding)
            person_id = result.person_id if result.is_match else None
            if person_id is not None and result.display_name is not None:
                self._names[person_id] = result.display_name
            self._vote(track.track_id, person_id)
            if person_id is not None:
                self._remember_appearance(image, track, person_id)

        # Appearance fallback for tracks with no face this refresh (turned
        # away, or the face detector simply missed): match its body-color
        # signature against a recently face-confirmed person's remembered
        # appearance. `_appearance` is only ever populated from a real face
        # match above, so this can only ever vote for someone this session
        # has already recognized -- never a person who's never been
        # face-matched, enrolled or not.
        if self._appearance:
            for track in tracked_people:
                if track.track_id in matched_track_ids:
                    continue
                person_id = self._match_appearance(image, track)
                if person_id is not None:
                    self._vote(track.track_id, person_id)

    def _vote(self, track_id: int, person_id: str | None) -> None:
        self._histories.setdefault(track_id, deque(maxlen=self._window)).append(person_id)

    def _remember_appearance(self, image: np.ndarray, track: TrackedPerson, person_id: str) -> None:
        signature = reid.embed(image, (track.x1, track.y1, track.x2, track.y2))
        if signature is not None:
            self._appearance[person_id] = signature
            self._appearance_age[person_id] = 0
            logger.info("Appearance signature stored", extra={"person_id": person_id})

    def _match_appearance(self, image: np.ndarray, track: TrackedPerson) -> str | None:
        signature = reid.embed(image, (track.x1, track.y1, track.x2, track.y2))
        if signature is None:
            return None
        best_person: str | None = None
        best_score = self._appearance_threshold
        best_seen_person: str | None = None
        best_seen_score = -1.0
        for person_id, remembered in self._appearance.items():
            score = reid.similarity(signature, remembered)
            if score > best_seen_score:
                best_seen_person, best_seen_score = person_id, score
            if score > best_score:
                best_person, best_score = person_id, score
        if best_seen_person is not None:
            # Diagnostic for tuning appearance_match_threshold/torso cropping
            # against real footage -- logged whether or not it matched, so a
            # near-miss (score just under threshold) is visible, not just hits.
            logger.info(
                "Appearance fallback attempted",
                extra={
                    "track_id": track.track_id,
                    "best_candidate": best_seen_person,
                    "best_score": round(best_seen_score, 3),
                    "threshold": self._appearance_threshold,
                    "candidate_age_frames": self._appearance_age.get(best_seen_person),
                    "matched": best_person is not None,
                },
            )
        return best_person

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
