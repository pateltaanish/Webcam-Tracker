"""Encryption primitives for the identity store (Stage 2.1).

Implements the envelope-encryption scheme from docs/database_design.md sec 5:

    operator passphrase --Argon2id--> KEK --unwraps--> DEK --encrypts--> data

Two keys:
  - KEK (key-encryption key): derived from the passphrase, never stored.
  - DEK (data-encryption key): random, encrypts the actual data, stored only
    in KEK-wrapped form in the key-vault sidecar.

Everything here is stateless functions plus small value objects; the stateful
holder of an unlocked DEK is ProfileStore. All authenticated encryption is
AES-256-GCM, so tampering (or a wrong key) is detected on decrypt rather than
silently producing garbage -- that's also how a wrong passphrase is rejected
(no separately-stored password hash).
"""

from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from typing import Any

from argon2.low_level import Type, hash_secret_raw
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

_KEY_BYTES = 32  # AES-256
_NONCE_BYTES = 12  # standard GCM nonce
_SALT_BYTES = 16
_KEYVAULT_VERSION = 1


class InvalidPassphraseError(Exception):
    """Raised when a passphrase fails to unwrap the data key (wrong passphrase,
    or a tampered/corrupt key vault)."""


@dataclass(frozen=True)
class KdfParams:
    """Argon2id cost parameters. Stored (non-secret) in the key vault so an
    existing store keeps the params it was created with even if defaults rise."""

    time_cost: int
    memory_kib: int
    parallelism: int


def _derive_kek(passphrase: str, salt: bytes, params: KdfParams) -> bytes:
    """Argon2id( passphrase, salt ) -> 32-byte key-encryption key."""
    return hash_secret_raw(
        secret=passphrase.encode("utf-8"),
        salt=salt,
        time_cost=params.time_cost,
        memory_cost=params.memory_kib,
        parallelism=params.parallelism,
        hash_len=_KEY_BYTES,
        type=Type.ID,
    )


def create_keyvault(passphrase: str, params: KdfParams) -> tuple[dict[str, Any], bytes]:
    """Set up a brand-new store's crypto. Returns (keyvault_dict, dek):
    the dict is JSON-serializable and safe to store in the clear; the dek is
    the raw data key to keep in memory for this session."""
    salt = os.urandom(_SALT_BYTES)
    kek = _derive_kek(passphrase, salt, params)
    dek = os.urandom(_KEY_BYTES)
    nonce = os.urandom(_NONCE_BYTES)
    wrapped_dek = AESGCM(kek).encrypt(nonce, dek, None)
    keyvault = {
        "version": _KEYVAULT_VERSION,
        "kdf": "argon2id",
        "kdf_params": {
            "time_cost": params.time_cost,
            "memory_kib": params.memory_kib,
            "parallelism": params.parallelism,
        },
        "salt": _b64(salt),
        "dek_nonce": _b64(nonce),
        "wrapped_dek": _b64(wrapped_dek),
    }
    return keyvault, dek


def unlock_keyvault(keyvault: dict[str, Any], passphrase: str) -> bytes:
    """Derive the KEK from the passphrase and unwrap the DEK. Raises
    InvalidPassphraseError if the passphrase is wrong or the vault is corrupt."""
    params = KdfParams(
        time_cost=keyvault["kdf_params"]["time_cost"],
        memory_kib=keyvault["kdf_params"]["memory_kib"],
        parallelism=keyvault["kdf_params"]["parallelism"],
    )
    kek = _derive_kek(passphrase, _unb64(keyvault["salt"]), params)
    try:
        return AESGCM(kek).decrypt(
            _unb64(keyvault["dek_nonce"]), _unb64(keyvault["wrapped_dek"]), None
        )
    except InvalidTag as exc:
        raise InvalidPassphraseError("passphrase did not unlock the identity store") from exc


def rewrap_dek(dek: bytes, passphrase: str, params: KdfParams) -> dict[str, Any]:
    """Re-wrap an existing DEK under a new passphrase (fresh salt/nonce) --
    used by change-passphrase, which never re-encrypts the actual data."""
    salt = os.urandom(_SALT_BYTES)
    kek = _derive_kek(passphrase, salt, params)
    nonce = os.urandom(_NONCE_BYTES)
    wrapped_dek = AESGCM(kek).encrypt(nonce, dek, None)
    return {
        "version": _KEYVAULT_VERSION,
        "kdf": "argon2id",
        "kdf_params": {
            "time_cost": params.time_cost,
            "memory_kib": params.memory_kib,
            "parallelism": params.parallelism,
        },
        "salt": _b64(salt),
        "dek_nonce": _b64(nonce),
        "wrapped_dek": _b64(wrapped_dek),
    }


def encrypt_field(dek: bytes, plaintext: bytes) -> bytes:
    """Encrypt one field value. Returns nonce || ciphertext(+tag), self-contained."""
    nonce = os.urandom(_NONCE_BYTES)
    return nonce + AESGCM(dek).encrypt(nonce, plaintext, None)


def decrypt_field(dek: bytes, blob: bytes) -> bytes:
    """Inverse of encrypt_field. Raises InvalidPassphraseError if the blob was
    written under a different key or has been tampered with."""
    nonce, ciphertext = blob[:_NONCE_BYTES], blob[_NONCE_BYTES:]
    try:
        return AESGCM(dek).decrypt(nonce, ciphertext, None)
    except InvalidTag as exc:
        raise InvalidPassphraseError(
            "could not decrypt field (wrong key or tampered data)"
        ) from exc


def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def _unb64(text: str) -> bytes:
    return base64.b64decode(text)
