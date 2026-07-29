"""Builds a ProfileStore from config.

Returns a store that is NOT yet unlocked -- the caller must call
`initialize(passphrase)` (first run) or `unlock(passphrase)` (subsequent runs)
with the operator passphrase, which is never read from config or disk.
"""

from __future__ import annotations

from webcam_tracker.config import AppConfig
from webcam_tracker.database.accounts import AccountManager
from webcam_tracker.database.crypto import KdfParams
from webcam_tracker.database.store import ProfileStore


def _kdf_params(config: AppConfig) -> KdfParams:
    identity = config.identity
    return KdfParams(
        time_cost=identity.argon2_time_cost,
        memory_kib=identity.argon2_memory_kib,
        parallelism=identity.argon2_parallelism,
    )


def create_profile_store(config: AppConfig) -> ProfileStore:
    identity = config.identity
    store_dir = config.resolve_path(identity.store_dir)
    return ProfileStore(
        db_path=store_dir / identity.db_filename,
        keyvault_path=store_dir / identity.keyvault_filename,
        kdf_params=_kdf_params(config),
        min_passphrase_length=identity.min_passphrase_length,
    )


def create_account_manager(config: AppConfig) -> AccountManager:
    """Build the multi-user login manager (Stage 2.5) for the accounts file
    alongside the profile store."""
    identity = config.identity
    store_dir = config.resolve_path(identity.store_dir)
    return AccountManager(
        accounts_path=store_dir / identity.accounts_filename,
        kdf_params=_kdf_params(config),
        min_passphrase_length=identity.min_passphrase_length,
    )
