"""High-level encrypted DNA encoder for NumPy float32 arrays."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from .aes_crypto import decrypt_dna, encrypt_dna, generate_key
from .binary_mapper import binary_to_float32_array, float32_array_to_binary
from .dna_mapper import binary_to_dna, dna_to_binary


class DNAEncoder:
    """Encode float32 arrays as encrypted DNA symbols and decode them losslessly."""

    def __init__(self, key: bytes | None = None) -> None:
        self.key = generate_key() if key is None else key

    def encode_array(self, array: np.ndarray) -> dict[str, str]:
        """Flatten and encrypt a NumPy-compatible float32 array."""
        binary_string = float32_array_to_binary(array)
        dna_string = binary_to_dna(binary_string)
        return encrypt_dna(dna_string, self.key)

    def decode_array(
        self,
        payload: Mapping[str, str],
        shape: Sequence[int],
    ) -> np.ndarray:
        """Decrypt an array payload and restore shape."""
        dna_string = decrypt_dna(payload, self.key)
        binary_string = dna_to_binary(dna_string)
        array = binary_to_float32_array(binary_string)

        try:
            return array.reshape(tuple(shape))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"shape {tuple(shape)} is incompatible with {array.size} values"
            ) from exc
