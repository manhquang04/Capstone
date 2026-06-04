"""Differential-privacy-style noise configuration helpers.

This project does not include a privacy accountant, so these settings are
reported as clipping/noise defenses rather than formal epsilon-delta DP.
"""

from __future__ import annotations

import os

DP_NOISE_PRESETS: dict[str, float] = {
    "utility": 0.0001,
    "weak": 0.0005,
    "mild": 0.001,
    "medium": 0.005,
    "strong": 0.01,
}
DEFAULT_DP_CLIP_NORM = 100.0
DEFAULT_DP_NOISE_PRESET = "medium"


def dp_clip_norm_from_env() -> float:
    return float(os.environ.get("DP_CLIP_NORM", str(DEFAULT_DP_CLIP_NORM)))


def dp_noise_multiplier_from_env() -> tuple[float, str]:
    if "DP_NOISE_MULTIPLIER" in os.environ:
        return float(os.environ["DP_NOISE_MULTIPLIER"]), "custom"
    preset = os.environ.get("DP_NOISE_PRESET", DEFAULT_DP_NOISE_PRESET).lower()
    if preset not in DP_NOISE_PRESETS:
        raise ValueError(
            f"DP_NOISE_PRESET must be one of {sorted(DP_NOISE_PRESETS)}, got {preset!r}"
        )
    return DP_NOISE_PRESETS[preset], preset


def dp_accounting_note() -> str:
    return (
        "No privacy accountant is implemented; this is a client-update clipping "
        "and Gaussian-noise defense, not a formal epsilon-delta DP guarantee."
    )
