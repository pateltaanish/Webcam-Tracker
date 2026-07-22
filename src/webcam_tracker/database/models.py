"""Data models for the identity store (Stage 2.1).

Plain dataclasses returned by ProfileStore. Sensitive fields (display_name,
embedding vectors) are decrypted by the store before they reach these objects
-- an in-memory model instance always holds plaintext; only the on-disk
representation is encrypted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Person:
    """A registered, consenting person."""

    id: str
    display_name: str
    created_at: str  # ISO-8601 UTC
    active: bool
    consent_version: str


@dataclass(frozen=True)
class Embedding:
    """One biometric embedding belonging to a person."""

    id: str
    person_id: str
    kind: str  # 'face' | 'reid'
    model_id: str  # which model produced it, e.g. 'insightface/buffalo_l'
    vector: np.ndarray  # float32, L2-normalized by convention
    quality: float
    created_at: str


@dataclass(frozen=True)
class ConsentEvent:
    """An enrollment / revocation / deletion consent record. Retained for the
    audit trail even after the person is deleted (holds no biometric data)."""

    id: str
    person_id: str
    event: str  # 'granted' | 'revoked' | 'deleted'
    timestamp: str
    note: str


@dataclass(frozen=True)
class AuditEntry:
    """One line of the audit log (store lifecycle + identity decisions)."""

    id: str
    timestamp: str
    action: str
    detail: str
