"""AES-256-GCM authenticated encryption for DNA strings."""

from __future__ import annotations

import base64
import binascii
from collections.abc import Mapping

from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes

KEY_SIZE_BYTES = 32
NONCE_SIZE_BYTES = 12


def generate_key() -> bytes:
    """Generate a random AES-256 key."""
    return get_random_bytes(KEY_SIZE_BYTES)


def encrypt_dna(dna_string: str, key: bytes) -> dict[str, str]:
    """Encrypt a DNA string and return a JSON-friendly authenticated payload."""
    _validate_key(key)
    if not isinstance(dna_string, str):
        raise TypeError("dna_string must be a string")

    cipher = AES.new(key, AES.MODE_GCM, nonce=get_random_bytes(NONCE_SIZE_BYTES))
    ciphertext, tag = cipher.encrypt_and_digest(dna_string.encode("ascii"))
    return {
        "nonce": _encode_base64(cipher.nonce),
        "tag": _encode_base64(tag),
        "ciphertext": _encode_base64(ciphertext),
    }


def decrypt_dna(encrypted_payload: Mapping[str, str], key: bytes) -> str:
    """Decrypt and authenticate a JSON-friendly DNA payload."""
    _validate_key(key)
    if not isinstance(encrypted_payload, Mapping):
        raise TypeError("encrypted_payload must be a mapping")

    required_fields = {"nonce", "tag", "ciphertext"}
    if set(encrypted_payload) != required_fields:
        raise ValueError(f"encrypted_payload must contain exactly {required_fields}")

    nonce = _decode_base64(encrypted_payload["nonce"], "nonce")
    tag = _decode_base64(encrypted_payload["tag"], "tag")
    ciphertext = _decode_base64(encrypted_payload["ciphertext"], "ciphertext")
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    return cipher.decrypt_and_verify(ciphertext, tag).decode("ascii")


def _validate_key(key: bytes) -> None:
    if not isinstance(key, bytes):
        raise TypeError("key must be bytes")
    if len(key) != KEY_SIZE_BYTES:
        raise ValueError("AES-256 key must be exactly 32 bytes")


def _encode_base64(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def _decode_base64(value: str, field_name: str) -> bytes:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a base64 string")
    try:
        return base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"{field_name} must be valid base64") from exc
