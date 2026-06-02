"""Protect an extracted MNIST gradient with DNA encoding and AES-GCM."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dna_encoder.encoder import DNAEncoder

RESULTS_DIR = PROJECT_ROOT / "results" / "mnist"
RAW_GRADIENT_PATH = RESULTS_DIR / "raw_gradient.pt"
DNA_GRADIENT_PATH = RESULTS_DIR / "dna_gradient.pt"
METRICS_PATH = RESULTS_DIR / "dna_gradient_metrics.json"


def payload_size(payload: dict[str, str]) -> int:
    return len(json.dumps(payload, separators=(",", ":")).encode("utf-8"))


def main() -> None:
    if not RAW_GRADIENT_PATH.is_file():
        raise FileNotFoundError(
            f"Raw gradient not found at '{RAW_GRADIENT_PATH}'. Run "
            "'python attacks/gradient_extraction.py' first."
        )

    raw_bundle = torch.load(RAW_GRADIENT_PATH, map_location="cpu", weights_only=True)
    encoder = DNAEncoder()
    decoded_gradients = []
    encode_seconds = 0.0
    decode_seconds = 0.0
    original_bytes = 0
    encrypted_payload_bytes = 0

    for gradient in raw_bundle["gradients"]:
        gradient_array = gradient.numpy().astype(np.float32, copy=False)
        original_bytes += gradient_array.nbytes

        started = perf_counter()
        payload = encoder.encode_array(gradient_array)
        encode_seconds += perf_counter() - started
        encrypted_payload_bytes += payload_size(payload)

        started = perf_counter()
        decoded_array = encoder.decode_array(payload, gradient.shape)
        decode_seconds += perf_counter() - started
        if not np.array_equal(decoded_array.view(np.uint32), gradient_array.view(np.uint32)):
            raise ValueError("DNA round trip did not preserve gradient bits")
        decoded_gradients.append(torch.from_numpy(decoded_array.copy()))

    overhead = (encrypted_payload_bytes - original_bytes) / original_bytes * 100
    dna_bundle = {
        **raw_bundle,
        "gradients": decoded_gradients,
        "source": "dna",
    }
    metrics = {
        "encode_time_ms": encode_seconds * 1_000,
        "decode_time_ms": decode_seconds * 1_000,
        "original_bytes": original_bytes,
        "encrypted_payload_bytes": encrypted_payload_bytes,
        "bandwidth_overhead_percent": overhead,
    }
    torch.save(dna_bundle, DNA_GRADIENT_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Saved gradient: {DNA_GRADIENT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
