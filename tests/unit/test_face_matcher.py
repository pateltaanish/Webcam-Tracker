"""Unit tests for webcam_tracker.face_recognition.matcher.

Uses synthetic L2-normalized embeddings (random high-dimensional unit vectors
are near-orthogonal, so they stand in for 'different people') against a real
temp store, so matching logic is tested without loading any model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from webcam_tracker.database import KdfParams, Person, ProfileStore
from webcam_tracker.face_recognition import FaceMatcher


def _unit(seed: int, dim: int = 512) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vector = rng.normal(size=dim).astype(np.float32)
    return vector / np.linalg.norm(vector)


def _store(tmp_path: Path) -> ProfileStore:
    store = ProfileStore(
        tmp_path / "profiles.db",
        tmp_path / "keyvault.json",
        KdfParams(1, 8192, 1),
        min_passphrase_length=4,
    )
    store.initialize("test-pass")
    return store


def _enroll(store: ProfileStore, name: str, vector: np.ndarray) -> Person:
    person = store.add_person(name, consent_version="v1")
    store.add_embedding(person.id, "face", "insightface/buffalo_l", vector, quality=0.9)
    return person


class TestFaceMatcher:
    def test_empty_store_is_unknown(self, tmp_path: Path) -> None:
        matcher = FaceMatcher(0.35)
        matcher.load(_store(tmp_path))
        assert matcher.num_templates == 0
        result = matcher.match(_unit(1))
        assert result.is_match is False
        assert result.person_id is None
        assert result.score == 0.0

    def test_matches_enrolled_person(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        alice = _enroll(store, "Alice", _unit(1))
        _enroll(store, "Bob", _unit(2))
        matcher = FaceMatcher(0.35)
        matcher.load(store)

        result = matcher.match(_unit(1))  # exactly Alice's template
        assert result.is_match is True
        assert result.person_id == alice.id
        assert result.display_name == "Alice"
        assert result.score > 0.99

    def test_near_template_still_matches(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        alice = _enroll(store, "Alice", _unit(1))
        matcher = FaceMatcher(0.35)
        matcher.load(store)

        base = _unit(1)
        noisy = base + 0.1 * _unit(7)
        noisy = noisy / np.linalg.norm(noisy)
        result = matcher.match(noisy)
        assert result.is_match is True
        assert result.person_id == alice.id

    def test_unknown_face_below_threshold(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        _enroll(store, "Alice", _unit(1))
        _enroll(store, "Bob", _unit(2))
        matcher = FaceMatcher(0.35)
        matcher.load(store)

        result = matcher.match(_unit(999))  # unrelated -> near-orthogonal to both
        assert result.is_match is False
        assert result.person_id is None
        assert result.score < 0.35

    def test_multiple_templates_per_person(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        alice = store.add_person("Alice", consent_version="v1")
        for seed in (1, 2, 3):
            store.add_embedding(alice.id, "face", "m", _unit(seed), quality=0.9)
        matcher = FaceMatcher(0.35)
        matcher.load(store)

        assert matcher.num_templates == 3
        # A query equal to the 2nd template matches Alice.
        assert matcher.match(_unit(2)).person_id == alice.id

    def test_revoked_person_not_matched(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        alice = _enroll(store, "Alice", _unit(1))
        store.revoke_person(alice.id)
        matcher = FaceMatcher(0.35)
        matcher.load(store)  # active-only -> Alice excluded

        assert matcher.num_templates == 0
        assert matcher.match(_unit(1)).is_match is False
