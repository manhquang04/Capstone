"""Conversion between binary strings and DNA symbol strings."""

from __future__ import annotations

BINARY_TO_DNA = {
    "00": "A",
    "01": "T",
    "10": "G",
    "11": "C",
}
DNA_TO_BINARY = {dna: binary for binary, dna in BINARY_TO_DNA.items()}


def binary_to_dna(binary_string: str) -> str:
    """Map each two-bit group to one DNA symbol."""
    if not isinstance(binary_string, str):
        raise TypeError("binary_string must be a string")
    if any(bit not in "01" for bit in binary_string):
        raise ValueError("binary_string must contain only '0' and '1'")
    if len(binary_string) % 2 != 0:
        raise ValueError("binary_string length must be divisible by 2")

    return "".join(
        BINARY_TO_DNA[binary_string[index : index + 2]]
        for index in range(0, len(binary_string), 2)
    )


def dna_to_binary(dna_string: str) -> str:
    """Map each DNA symbol back to its two-bit group."""
    if not isinstance(dna_string, str):
        raise TypeError("dna_string must be a string")
    if any(symbol not in DNA_TO_BINARY for symbol in dna_string):
        raise ValueError("dna_string must contain only 'A', 'T', 'G', and 'C'")

    return "".join(DNA_TO_BINARY[symbol] for symbol in dna_string)
