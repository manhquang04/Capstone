"""Basic runtime and bandwidth benchmark for the independent DNA Encoder."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dna_encoder.encoder import DNAEncoder

SIZES = (100, 1_000, 10_000, 100_000)


def payload_size_bytes(payload: dict[str, str]) -> int:
    """Return compact JSON payload size in bytes."""
    return len(json.dumps(payload, separators=(",", ":")).encode("utf-8"))


def main() -> None:
    rng = np.random.default_rng(seed=42)
    encoder = DNAEncoder()

    header = (
        f"{'Values':>10} | {'Encode ms':>10} | {'Decode ms':>10} | "
        f"{'Original B':>12} | {'DNA length':>12} | {'Payload B':>10} | "
        f"{'Overhead %':>10}"
    )
    print(header)
    print("-" * len(header))

    for size in SIZES:
        original = rng.standard_normal(size, dtype=np.float32)

        started = perf_counter()
        payload = encoder.encode_array(original)
        encode_ms = (perf_counter() - started) * 1_000

        started = perf_counter()
        decoded = encoder.decode_array(payload, original.shape)
        decode_ms = (perf_counter() - started) * 1_000

        assert decoded.dtype == np.float32
        assert decoded.shape == original.shape
        assert np.array_equal(decoded.view(np.uint32), original.view(np.uint32))

        original_bytes = original.nbytes
        dna_length = original.size * 16
        encrypted_payload_bytes = payload_size_bytes(payload)
        overhead_percent = (
            (encrypted_payload_bytes - original_bytes) / original_bytes * 100
        )

        print(
            f"{size:>10,} | {encode_ms:>10.3f} | {decode_ms:>10.3f} | "
            f"{original_bytes:>12,} | {dna_length:>12,} | "
            f"{encrypted_payload_bytes:>10,} | {overhead_percent:>9.2f}%"
        )


if __name__ == "__main__":
    main()
