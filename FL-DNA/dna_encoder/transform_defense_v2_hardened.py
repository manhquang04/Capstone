"""Opt-in keyed v2; no key or coordinate list in public metadata.

Keys stay out of dataclasses and must be supplied out-of-band. This is not
encryption, authenticated transport, or differential privacy.
"""
from dataclasses import dataclass
import hashlib
import hmac
import math
import secrets

import numpy as np

VERSION = "dna-v2-hkdf-sha256-v1"


def generate_v2_key() -> bytes:
    """Generate a fresh 128-bit upload key; never serialize it with the upload."""
    return secrets.token_bytes(16)


def _key(key):
    if not isinstance(key, bytes) or len(key) != 16:
        raise ValueError("hardened v2 requires a separate 16-byte key")
    return key


def _hkdf(ikm, salt, info, length=32):
    """RFC5869 extract/expand, also exercised against its SHA256 test vector."""
    if not 0 < length <= 255 * 32:
        raise ValueError("invalid HKDF length")
    prk = hmac.new(salt or bytes(32), ikm, hashlib.sha256).digest()
    out, block = b"", b""
    for i in range(1, math.ceil(length / 32) + 1):
        block = hmac.new(prk, block + info + bytes([i]), hashlib.sha256).digest()
        out += block
    return out[:length]


def _subkey(key, tensor_index, purpose, context=0):
    _key(key)
    if type(tensor_index) is not int or not 0 <= tensor_index < 2**64:
        raise ValueError("invalid tensor index")
    if type(context) is not int or not 0 <= context < 2**64:
        raise ValueError("invalid quantization context")
    info = (VERSION.encode() + b"/tensor/" + tensor_index.to_bytes(8, "big")
            + b"/purpose/" + purpose + b"/context/" + context.to_bytes(8, "big"))
    return _hkdf(key, VERSION.encode(), info)


class _Stream:
    def __init__(self, key):
        self.key, self.counter, self.buffer = key, 0, b""

    def read(self, n):
        while len(self.buffer) < n:
            self.buffer += hmac.new(self.key, b"stream/" + self.counter.to_bytes(16, "big"), hashlib.sha256).digest()
            self.counter += 1
        out, self.buffer = self.buffer[:n], self.buffer[n:]
        return out

    def below(self, n):
        width = max(1, (n.bit_length() + 7) // 8)
        limit = 256**width - 256**width % n
        while True:
            value = int.from_bytes(self.read(width), "big")
            if value < limit:
                return value % n


@dataclass(frozen=True)
class HardenedV2Metadata:
    original_shape: tuple[int, ...]
    original_size: int
    padded_size: int
    sketch_size: int
    compression_ratio: float
    quantization_eta: float
    quantization_delta: float
    tensor_index: int
    version: str = VERSION


def _projection(n, k, key, index):
    signs_stream = _Stream(_subkey(key, index, b"signs"))
    signs = np.array([1.0 if x & 1 else -1.0 for x in signs_stream.read(n)])
    stream = _Stream(_subkey(key, index, b"sampling"))
    permutation = np.arange(n)
    for i in range(k):
        j = i + stream.below(n - i)
        permutation[i], permutation[j] = permutation[j], permutation[i]
    return signs, np.sort(permutation[:k])


def _finite(value):
    if not np.isfinite(value).all():
        raise ValueError("non-finite hardened v2 value")


def transform(array, config, tensor_index=0, quantization_seed=None, *, key=None):
    from .transform_defense_v2 import _validate_config, _next_power_of_two, _sketch_size, _fwht_normalized, _quantization_delta
    _key(key)
    if config.seed is not None:
        raise ValueError("hardened mode does not accept legacy seed")
    _validate_config(config)
    if not np.isfinite([config.compression_ratio, config.quantization_eta]).all():
        raise ValueError("non-finite config")
    original = np.asarray(array, dtype=np.float32)
    _finite(original)
    flat = original.reshape(-1).astype(np.float64)
    n = _next_power_of_two(max(1, flat.size))
    k = _sketch_size(n, config.compression_ratio)
    signs, sampled = _projection(n, k, key, tensor_index)
    padded = np.zeros(n)
    padded[:flat.size] = flat
    projected = np.sqrt(n / k) * _fwht_normalized(padded * signs)[sampled]
    delta = _quantization_delta(flat, config.quantization_eta)
    context = 0 if quantization_seed is None else quantization_seed
    stream = _Stream(_subkey(key, tensor_index, b"quantization", context))
    if delta > 0:
        scaled = projected / delta
        lower = np.floor(scaled)
        draws = np.array([(int.from_bytes(stream.read(8), "big") >> 11) * 2.0**-53 for _ in range(k)])
        projected = (lower + (draws < scaled - lower)) * delta
    q = projected.astype(np.float32)
    _finite(q)
    metadata = HardenedV2Metadata(tuple(original.shape), int(flat.size), n, k,
                                  float(config.compression_ratio), float(config.quantization_eta), float(delta), tensor_index)
    return q, metadata


def reconstruct(sketch, metadata, *, key=None):
    from .transform_defense_v2 import _next_power_of_two, _sketch_size, _fwht_normalized, _stats
    _key(key)
    if not isinstance(metadata, HardenedV2Metadata) or metadata.version != VERSION:
        raise ValueError("invalid hardened metadata version")
    shape = metadata.original_shape
    if (not isinstance(shape, tuple) or any(type(x) is not int or x < 0 for x in shape)
            or math.prod(shape) != metadata.original_size
            or metadata.padded_size != _next_power_of_two(max(1, metadata.original_size))
            or not np.isfinite([metadata.compression_ratio, metadata.quantization_eta, metadata.quantization_delta]).all()
            or not 0 < metadata.compression_ratio <= 1 or metadata.quantization_eta < 0
            or metadata.quantization_delta < 0
            or metadata.sketch_size != _sketch_size(metadata.padded_size, metadata.compression_ratio)):
        raise ValueError("inconsistent hardened metadata")
    q = np.asarray(sketch, dtype=np.float64).reshape(-1)
    _finite(q)
    if q.size != metadata.sketch_size:
        raise ValueError("sketch length mismatch")
    signs, sampled = _projection(metadata.padded_size, metadata.sketch_size, key, metadata.tensor_index)
    lifted = np.zeros(metadata.padded_size)
    lifted[sampled] = np.sqrt(metadata.padded_size / metadata.sketch_size) * q
    recovered = (_fwht_normalized(lifted) * signs)[:metadata.original_size].reshape(shape)
    recovered = recovered.astype(np.float32)
    _finite(recovered)
    return recovered, _stats(recovered, metadata)
