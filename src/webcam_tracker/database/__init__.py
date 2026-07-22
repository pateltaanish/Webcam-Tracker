"""Registered-person identity database (Stage 2.1).

Local, encrypted SQLite store for registered people + their face/Re-ID
embeddings + consent metadata. Sensitive columns are encrypted at the
application layer with AES-256-GCM under a key derived from the operator
passphrase (never stored). See docs/database_design.md for the full design
(consent model, threat model, crypto, schema).

Entry point: `create_profile_store(config)` builds a locked ProfileStore;
call `initialize(passphrase)` (first run) or `unlock(passphrase)` to use it.
"""

from webcam_tracker.database.crypto import InvalidPassphraseError, KdfParams
from webcam_tracker.database.errors import (
    DatabaseError,
    DatabaseLockedError,
    PersonNotFoundError,
    StoreAlreadyInitializedError,
    StoreNotInitializedError,
    WeakPassphraseError,
)
from webcam_tracker.database.factory import create_profile_store
from webcam_tracker.database.models import AuditEntry, ConsentEvent, Embedding, Person
from webcam_tracker.database.store import ProfileStore

__all__ = [
    "AuditEntry",
    "ConsentEvent",
    "DatabaseError",
    "DatabaseLockedError",
    "Embedding",
    "InvalidPassphraseError",
    "KdfParams",
    "Person",
    "PersonNotFoundError",
    "ProfileStore",
    "StoreAlreadyInitializedError",
    "StoreNotInitializedError",
    "WeakPassphraseError",
    "create_profile_store",
]
