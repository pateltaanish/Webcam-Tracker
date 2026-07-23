"""Unit tests for webcam_tracker.identity.tracker.

Uses a fake embedder (returns preset faces) and a fake matcher (maps an
embedding's first element to a person), so the fusion logic -- face->track
association, temporal consensus, confirmation, pruning, cadence -- is tested
with no model or store.
"""

from __future__ import annotations

import numpy as np

from webcam_tracker.face_recognition import DetectedFace, MatchResult
from webcam_tracker.identity import IdentityTracker
from webcam_tracker.tracking import TrackedPerson

_IMAGE = np.zeros((10, 10, 3), dtype=np.uint8)  # embedder is fake; contents ignored


def _emb(code: float) -> np.ndarray:
    vector = np.zeros(512, dtype=np.float32)
    vector[0] = code  # the fake matcher reads this to decide the person
    return vector


def _face(cx: float, cy: float, code: float) -> DetectedFace:
    return DetectedFace(cx - 10, cy - 10, cx + 10, cy + 10, 0.9, _emb(code))


def _track(track_id: int, cx: float, cy: float) -> TrackedPerson:
    return TrackedPerson(track_id, cx - 20, cy - 20, cx + 20, cy + 20, 0.9)


class _FakeEmbedder:
    def __init__(self) -> None:
        self.faces: list[DetectedFace] = []

    def detect(self, image: np.ndarray) -> list[DetectedFace]:
        return list(self.faces)


class _FakeMatcher:
    _PEOPLE = {1.0: ("alice", "Alice"), 2.0: ("bob", "Bob")}

    def match(self, embedding: np.ndarray) -> MatchResult:
        person = self._PEOPLE.get(float(embedding[0]))
        if person is None:
            return MatchResult(False, None, None, 0.1)
        return MatchResult(True, person[0], person[1], 0.9)


def _make(
    embedder: _FakeEmbedder,
    matcher: _FakeMatcher,
    cadence: int = 1,
    window: int = 3,
    min_confidence: float = 0.6,
) -> IdentityTracker:
    return IdentityTracker(embedder, matcher, cadence, window, min_confidence)  # type: ignore[arg-type]


class TestIdentityTracker:
    def test_confirms_person_after_match(self) -> None:
        embedder = _FakeEmbedder()
        embedder.faces = [_face(50, 50, 1.0)]
        tracker = _make(embedder, _FakeMatcher())
        tracker.update(_IMAGE, [_track(1, 50, 50)])

        identity = tracker.identity_of(1)
        assert identity is not None
        assert identity.is_confirmed
        assert identity.person_id == "alice"
        assert identity.display_name == "Alice"
        assert tracker.resolve_person("alice") == 1

    def test_majority_survives_single_mismatch(self) -> None:
        embedder = _FakeEmbedder()
        matcher = _FakeMatcher()
        tracker = _make(embedder, matcher, window=3)
        track = [_track(1, 50, 50)]
        embedder.faces = [_face(50, 50, 1.0)]
        tracker.update(_IMAGE, track)
        tracker.update(_IMAGE, track)
        embedder.faces = [_face(50, 50, 2.0)]  # one Bob frame among Alices
        tracker.update(_IMAGE, track)

        identity = tracker.identity_of(1)
        assert identity is not None
        assert identity.person_id == "alice"  # 2/3 still Alice

    def test_unknown_face_is_unconfirmed(self) -> None:
        embedder = _FakeEmbedder()
        embedder.faces = [_face(50, 50, 0.0)]  # matches nobody
        tracker = _make(embedder, _FakeMatcher())
        tracker.update(_IMAGE, [_track(1, 50, 50)])

        identity = tracker.identity_of(1)
        assert identity is not None
        assert identity.is_confirmed is False
        assert identity.person_id is None
        assert tracker.resolve_person("alice") is None

    def test_no_face_keeps_last_identity(self) -> None:
        embedder = _FakeEmbedder()
        embedder.faces = [_face(50, 50, 1.0)]
        tracker = _make(embedder, _FakeMatcher())
        track = [_track(1, 50, 50)]
        tracker.update(_IMAGE, track)
        assert tracker.resolve_person("alice") == 1

        embedder.faces = []  # face turned away -- no detection
        tracker.update(_IMAGE, track)
        tracker.update(_IMAGE, track)
        assert tracker.resolve_person("alice") == 1  # identity retained

    def test_associates_faces_to_correct_tracks(self) -> None:
        embedder = _FakeEmbedder()
        embedder.faces = [_face(50, 50, 1.0), _face(150, 150, 2.0)]
        tracker = _make(embedder, _FakeMatcher())
        tracker.update(_IMAGE, [_track(1, 50, 50), _track(2, 150, 150)])

        assert tracker.resolve_person("alice") == 1
        assert tracker.resolve_person("bob") == 2

    def test_cadence_skips_intermediate_frames(self) -> None:
        embedder = _FakeEmbedder()
        embedder.faces = [_face(50, 50, 1.0)]
        tracker = _make(embedder, _FakeMatcher(), cadence=3)
        track = [_track(1, 50, 50)]

        tracker.update(_IMAGE, track)  # frame 1 -- no refresh
        tracker.update(_IMAGE, track)  # frame 2 -- no refresh
        assert tracker.identity_of(1) is None
        tracker.update(_IMAGE, track)  # frame 3 -- refresh
        assert tracker.resolve_person("alice") == 1

    def test_vanished_track_is_pruned(self) -> None:
        embedder = _FakeEmbedder()
        embedder.faces = [_face(50, 50, 1.0)]
        tracker = _make(embedder, _FakeMatcher())
        tracker.update(_IMAGE, [_track(1, 50, 50)])
        assert tracker.resolve_person("alice") == 1

        embedder.faces = []
        tracker.update(_IMAGE, [])  # track 1 gone
        assert tracker.identity_of(1) is None
        assert tracker.resolve_person("alice") is None
