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
        self.detect_calls = 0

    def detect(self, image: np.ndarray) -> list[DetectedFace]:
        self.detect_calls += 1
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
    grace: int = 10,
    appearance_threshold: float = 0.75,
    appearance_memory: int = 90,
) -> IdentityTracker:
    return IdentityTracker(  # type: ignore[arg-type]
        embedder,
        matcher,
        cadence,
        window,
        min_confidence,
        grace,
        appearance_threshold,
        appearance_memory,
    )


def _colored_image(
    size: int, box: tuple[int, int, int, int], color: tuple[int, int, int]
) -> np.ndarray:
    """A solid-black image with one BGR-colored rectangle -- real pixel
    content for the appearance (color-histogram) fallback tests, which read
    actual crops rather than the fake embedder's preset face list."""
    image = np.zeros((size, size, 3), dtype=np.uint8)
    x1, y1, x2, y2 = box
    image[y1:y2, x1:x2] = color
    return image


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
        tracker = _make(embedder, _FakeMatcher(), cadence=3, window=1)
        track = [_track(1, 50, 50)]

        tracker.update(_IMAGE, track)  # new track -- immediate refresh
        assert tracker.resolve_person("alice") == 1
        embedder.faces = [_face(50, 50, 2.0)]
        tracker.update(_IMAGE, track)  # one frame since refresh
        tracker.update(_IMAGE, track)  # two frames since refresh
        assert tracker.resolve_person("alice") == 1
        tracker.update(_IMAGE, track)  # three frames -- periodic refresh
        assert tracker.resolve_person("bob") == 1

    def test_new_track_bypasses_cadence_for_immediate_reidentification(self) -> None:
        embedder = _FakeEmbedder()
        tracker = _make(embedder, _FakeMatcher(), cadence=10)

        # Put the periodic refresh far from its next scheduled run.
        tracker.update(_IMAGE, [])
        embedder.faces = [_face(50, 50, 1.0)]

        # A returning person's new ByteTrack ID must be identified on its first
        # visible frame, not up to nine frames later at the periodic cadence.
        tracker.update(_IMAGE, [_track(9, 50, 50)])

        assert tracker.resolve_person("alice") == 9

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

    def test_stable_ids_maps_confirmed_tracks_to_person_id(self) -> None:
        embedder = _FakeEmbedder()
        embedder.faces = [_face(50, 50, 1.0), _face(150, 150, 0.0)]  # alice + unknown
        tracker = _make(embedder, _FakeMatcher())
        tracker.update(_IMAGE, [_track(1, 50, 50), _track(2, 150, 150)])

        assert tracker.stable_ids([1, 2]) == {1: "alice"}  # unknown track omitted

    def test_stable_ids_survives_a_track_id_change_on_reentry(self) -> None:
        # This is the concrete "leaves and re-enters with a new track id" case:
        # once the face re-confirms alice under the NEW track id, stable_ids
        # reports the same person_id key it did for the old one.
        embedder = _FakeEmbedder()
        embedder.faces = [_face(50, 50, 1.0)]
        tracker = _make(embedder, _FakeMatcher())
        tracker.update(_IMAGE, [_track(1, 50, 50)])
        assert tracker.stable_ids([1]) == {1: "alice"}

        embedder.faces = []
        tracker.update(_IMAGE, [])  # alice leaves frame; track 1 dropped

        embedder.faces = [_face(50, 50, 1.0)]
        tracker.update(_IMAGE, [_track(9, 50, 50)])  # reappears as a NEW track id

        assert tracker.stable_ids([9]) == {9: "alice"}  # same stable key as before

    def test_reacquires_within_grace_window_after_missed_first_frame(self) -> None:
        # The new track's very first frame misses the face entirely (motion
        # blur / an off-angle turn as someone walks back in) -- confirmation
        # must not then wait for the periodic cadence (here, effectively
        # never, at cadence=100); it should keep retrying every frame while
        # inside the grace window and confirm once the face becomes visible.
        embedder = _FakeEmbedder()
        matcher = _FakeMatcher()
        tracker = _make(embedder, matcher, cadence=100, window=1, grace=5)

        embedder.faces = []  # frame 1: new track, but no face found yet
        tracker.update(_IMAGE, [_track(9, 50, 50)])
        assert tracker.resolve_person("alice") is None

        embedder.faces = [_face(50, 50, 1.0)]  # frame 2: face now visible
        tracker.update(_IMAGE, [_track(9, 50, 50)])

        assert tracker.resolve_person("alice") == 9

    def test_grace_window_lapses_and_falls_back_to_cadence(self) -> None:
        # An unconfirmable track (no matching template, or face never visible)
        # must not force a refresh forever -- once its grace window lapses, it
        # falls back to the periodic cadence like any other unconfirmed track,
        # so a stranger standing in frame doesn't pin the face model to every
        # single frame indefinitely.
        embedder = _FakeEmbedder()
        matcher = _FakeMatcher()
        tracker = _make(embedder, matcher, cadence=100, window=1, grace=2)
        embedder.faces = []  # never matches -- stranger with no face visible
        track = [_track(9, 50, 50)]

        tracker.update(_IMAGE, track)  # new track -- grace frame 1 of 2
        tracker.update(_IMAGE, track)  # grace frame 2 of 2
        assert embedder.detect_calls == 2  # both grace frames forced a refresh

        tracker.update(_IMAGE, track)  # grace lapsed, cadence=100 not due yet
        assert embedder.detect_calls == 2  # no forced refresh, and no periodic one either

    def test_reacquires_by_appearance_when_no_face_visible(self) -> None:
        # alice is confirmed by face (remembering her clothing color), leaves
        # frame, and returns as a new track id facing away -- no face this
        # frame -- but wearing the same-colored clothes. She should reacquire
        # by appearance alone.
        embedder = _FakeEmbedder()
        matcher = _FakeMatcher()
        tracker = _make(embedder, matcher, cadence=100, window=1, grace=5)
        red = (0, 0, 255)
        image = _colored_image(200, (30, 30, 70, 70), red)

        embedder.faces = [_face(50, 50, 1.0)]
        tracker.update(image, [_track(1, 50, 50)])
        assert tracker.resolve_person("alice") == 1

        embedder.faces = []
        tracker.update(image, [])  # alice leaves frame; track 1 pruned
        tracker.update(image, [_track(9, 50, 50)])  # same clothes, new track, no face

        assert tracker.resolve_person("alice") == 9

    def test_appearance_fallback_does_not_match_a_different_color(self) -> None:
        embedder = _FakeEmbedder()
        matcher = _FakeMatcher()
        tracker = _make(embedder, matcher, cadence=100, window=1, grace=5)
        red = (0, 0, 255)
        blue = (255, 0, 0)

        embedder.faces = [_face(50, 50, 1.0)]
        tracker.update(_colored_image(200, (30, 30, 70, 70), red), [_track(1, 50, 50)])
        assert tracker.resolve_person("alice") == 1

        embedder.faces = []
        tracker.update(_colored_image(200, (30, 30, 70, 70), red), [])  # track 1 gone

        # a differently-dressed stranger appears where alice was -- must not
        # be handed her identity just because a track appeared nearby.
        tracker.update(_colored_image(200, (30, 30, 70, 70), blue), [_track(9, 50, 50)])
        assert tracker.identity_of(9) is None

    def test_appearance_memory_expires_after_configured_frames(self) -> None:
        embedder = _FakeEmbedder()
        matcher = _FakeMatcher()
        tracker = _make(embedder, matcher, cadence=100, window=1, grace=1, appearance_memory=2)
        red = (0, 0, 255)
        image = _colored_image(200, (30, 30, 70, 70), red)

        embedder.faces = [_face(50, 50, 1.0)]
        tracker.update(image, [_track(1, 50, 50)])
        assert tracker.resolve_person("alice") == 1

        embedder.faces = []
        tracker.update(image, [])  # track 1 gone; appearance memory starts aging
        tracker.update(image, [])
        tracker.update(image, [])  # memory=2 frames elapsed -- entry expires

        tracker.update(image, [_track(9, 50, 50)])  # same clothes, but memory lapsed
        assert tracker.identity_of(9) is None

    def test_stable_ids_empty_when_nothing_confirmed(self) -> None:
        embedder = _FakeEmbedder()
        embedder.faces = [_face(50, 50, 0.0)]  # matches nobody
        tracker = _make(embedder, _FakeMatcher())
        tracker.update(_IMAGE, [_track(1, 50, 50)])

        assert tracker.stable_ids([1]) == {}
