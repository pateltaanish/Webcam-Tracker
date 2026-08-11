"""Face matching: embedding -> "who is this?" (Stage 2.3).

Compares a query face embedding against the enrolled profiles' stored
embeddings by cosine similarity, returning the best-matching registered person
or an explicit UNKNOWN when nothing clears the threshold.

Design: `FaceMatcher` loads all active people's face embeddings from the
(unlocked) store ONCE into an in-memory matrix, so per-frame matching is a
single vectorized dot product -- the store is decrypted once, not every frame.
Call `load()` again (or rebuild via the factory) after new registrations.

ArcFace embeddings are L2-normalized, so cosine similarity is just their dot
product, in -1..1. A person may have several stored embeddings (different
angles); we match against the *best* of them.

Assumption: all stored face embeddings come from the same model as the query
(the buffalo_l ArcFace). Embeddings from different models aren't comparable;
mixing them would need a per-model index (not needed yet -- one model).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from webcam_tracker.database import ProfileStore
from webcam_tracker.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class MatchResult:
    """Outcome of matching one query embedding against the enrolled profiles.

    `is_match` False means UNKNOWN (no enrolled person cleared the threshold);
    `score` is still the best similarity found, for display/debugging.
    """

    is_match: bool
    person_id: str | None
    display_name: str | None
    score: float


class FaceMatcher:
    def __init__(self, match_threshold: float) -> None:
        self._threshold = match_threshold
        self._matrix = np.zeros((0, 512), dtype=np.float32)
        self._person_ids: list[str] = []
        self._display_names: list[str] = []

    def load(self, store: ProfileStore) -> None:
        """(Re)build the in-memory template index from the store's ACTIVE
        people. Revoked people (active=0) are excluded, so they stop matching."""
        vectors: list[np.ndarray] = []
        person_ids: list[str] = []
        display_names: list[str] = []
        for person in store.list_people():  # active only
            for embedding in store.get_embeddings(person.id, kind="face"):
                vectors.append(embedding.vector)
                person_ids.append(person.id)
                display_names.append(person.display_name)

        self._matrix = (
            np.ascontiguousarray(np.vstack(vectors), dtype=np.float32)
            if vectors
            else np.zeros((0, 512), dtype=np.float32)
        )
        self._person_ids = person_ids
        self._display_names = display_names
        logger.info(
            "Face matcher loaded",
            extra={"templates": len(person_ids), "people": len(set(person_ids))},
        )

    @property
    def num_templates(self) -> int:
        return len(self._person_ids)

    @property
    def enrolled_names(self) -> set[str]:
        return set(self._display_names)

    def _best(self, embedding: np.ndarray) -> tuple[str | None, str | None, float]:
        if self._matrix.shape[0] == 0:
            return None, None, 0.0
        query = np.asarray(embedding, dtype=np.float32)
        similarities = self._matrix @ query  # cosine (both L2-normalized)
        best = int(np.argmax(similarities))
        return self._person_ids[best], self._display_names[best], float(similarities[best])

    def match(self, embedding: np.ndarray) -> MatchResult:
        """Best registered person for this embedding, or UNKNOWN."""
        person_id, display_name, score = self._best(embedding)
        if person_id is not None and score >= self._threshold:
            return MatchResult(
                is_match=True, person_id=person_id, display_name=display_name, score=score
            )
        return MatchResult(is_match=False, person_id=None, display_name=None, score=score)

    def best_candidate(self, embedding: np.ndarray) -> MatchResult:
        """Best-scoring template regardless of `match_threshold` -- for offline
        threshold tuning (`scripts/accuracy_test.py`), which needs to re-decide
        accept/reject at many threshold values without reloading the store.
        `is_match` here just means "at least one template exists", not that it
        clears any threshold -- callers apply their own cutoff to `score`.
        """
        person_id, display_name, score = self._best(embedding)
        return MatchResult(
            is_match=person_id is not None,
            person_id=person_id,
            display_name=display_name,
            score=score,
        )
