"""Identity-store exceptions."""

from __future__ import annotations


class DatabaseError(Exception):
    """Base class for identity-store errors."""


class StoreAlreadyInitializedError(DatabaseError):
    """initialize() called on a store that already has a key vault."""


class StoreNotInitializedError(DatabaseError):
    """unlock() called before the store has been initialized."""


class DatabaseLockedError(DatabaseError):
    """A data operation was attempted before initialize()/unlock()."""


class PersonNotFoundError(DatabaseError):
    """Referenced a person id that isn't in the store."""


class WeakPassphraseError(DatabaseError):
    """Passphrase shorter than the configured minimum length."""
