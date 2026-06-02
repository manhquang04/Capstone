"""Bit-exact conversion between NumPy float32 arrays and binary strings."""

from __future__ import annotations

import numpy as np


def float32_array_to_binary(array: np.ndarray) -> str:
    """Serialize an array as an IEEE 754 float32 binary string."""
    float32_array = np.asarray(array, dtype=np.float32)
    little_endian_array = np.ascontiguousarray(float32_array, dtype="<f4")
    return "".join(f"{byte:08b}" for byte in little_endian_array.tobytes())


def binary_to_float32_array(binary_string: str) -> np.ndarray:
    """Deserialize an IEEE 754 float32 binary string into a 1D array."""
    if not isinstance(binary_string, str):
        raise TypeError("binary_string must be a string")
    if any(bit not in "01" for bit in binary_string):
        raise ValueError("binary_string must contain only '0' and '1'")
    if len(binary_string) % 32 != 0:
        raise ValueError("float32 binary string length must be divisible by 32")

    raw_bytes = bytes(
        int(binary_string[index : index + 8], 2)
        for index in range(0, len(binary_string), 8)
    )
    return np.frombuffer(raw_bytes, dtype="<f4").astype(np.float32, copy=False)
