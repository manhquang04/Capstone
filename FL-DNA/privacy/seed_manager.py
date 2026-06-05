"""Dynamic seed helpers for reproducible defense randomness."""

from __future__ import annotations

import hashlib
import secrets

MAX_SEED = 2**31 - 1


def generate_run_seed() -> int:
    """Generate a fresh per-run seed from a secure random source."""
    return secrets.randbelow(MAX_SEED) + 1


def derive_seed(run_seed: int, *parts: object) -> int:
    """Derive a deterministic child seed from a recorded run seed."""
    payload = "|".join([str(run_seed), *(str(part) for part in parts)]).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "big") % MAX_SEED + 1
