"""Directly executable smoke test for the independent DNA Encoder."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dna_encoder.binary_mapper import float32_array_to_binary
from dna_encoder.dna_mapper import binary_to_dna
from dna_encoder.encoder import DNAEncoder


def main() -> None:
    original = np.array(
        [0.1234, -1.25, 3.14159, 0.0, -0.0001],
        dtype=np.float32,
    )
    encoder = DNAEncoder()

    payload = encoder.encode_array(original)
    decoded = encoder.decode_array(payload, original.shape)
    dna_length = len(binary_to_dna(float32_array_to_binary(original)))
    payload_size = len(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    max_error = float(np.max(np.abs(original - decoded)))

    print(f"Original array: {original}")
    print(f"Decoded array:  {decoded}")
    print(f"Max error: {max_error}")
    print(f"Original byte size: {original.nbytes}")
    print(f"DNA length: {dna_length}")
    print(f"Encrypted payload size: {payload_size}")

    assert decoded.dtype == np.float32
    assert decoded.shape == original.shape
    assert np.allclose(decoded, original)
    assert np.array_equal(decoded.view(np.uint32), original.view(np.uint32))

    print("DNA Encoder Test: PASSED")


if __name__ == "__main__":
    main()
