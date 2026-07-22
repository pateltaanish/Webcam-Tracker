"""Builds a ProfileStore from config.

Returns a store that is NOT yet unlocked -- the caller must call
`initialize(passphrase)` (first run) or `unlock(passphrase)` (subsequent runs)
with the operator passphrase, which is never read from config or disk.
"""

from __future__ import annotations

from webcam_tracker.config import AppConfig
from webcam_tracker.database.crypto import KdfParams
from webcam_tracker.database.store import ProfileStore


def create_profile_store(config: AppConfig) -> ProfileStore:
    identity = config.identity
    store_dir = config.resolve_path(identity.store_dir)
    return ProfileStore(
        db_path=store_dir / identity.db_filename,
        keyvault_path=store_dir / identity.keyvault_filename,
        kdf_params=KdfParams(
            time_cost=identity.argon2_time_cost,
            memory_kib=identity.argon2_memory_kib,
            parallelism=identity.argon2_parallelism,
        ),
        min_passphrase_length=identity.min_passphrase_length,
    )
