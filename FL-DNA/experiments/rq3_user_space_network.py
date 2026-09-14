"""Versioned user-space network emulator for the frozen RQ3 benchmark."""

from __future__ import annotations

import hashlib
import math
import socket
import threading
import time
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NetworkProfile:
    name: str
    uplink_mbps: float
    rtt_ms: float
    packet_loss_fraction: float


@dataclass(frozen=True)
class TransferResult:
    payload_bytes: int
    packets: int
    initially_lost_packets: int
    second_loss_packets: int
    retransmitted_bytes: int
    modeled_delay_seconds: float
    actual_elapsed_seconds: float
    payload_sha256: str
    received_sha256: str
    correct: bool
    timed_out: bool


def transfer_payload(
    payload: bytes,
    profile: NetworkProfile,
    *,
    seed: int,
    timeout_seconds: float,
    packet_bytes: int = 1200,
) -> TransferResult:
    """Move bytes over loopback and impose a seeded reliable-upload delay."""
    if profile.uplink_mbps <= 0 or profile.rtt_ms < 0:
        raise ValueError("bandwidth must be positive and RTT nonnegative")
    if not 0.0 <= profile.packet_loss_fraction < 1.0:
        raise ValueError("packet loss must be in [0, 1)")
    if packet_bytes <= 0 or timeout_seconds <= 0:
        raise ValueError("packet_bytes and timeout_seconds must be positive")

    payload = bytes(payload)
    packets = max(1, math.ceil(len(payload) / packet_bytes))
    rng = np.random.default_rng(seed)
    if profile.packet_loss_fraction:
        retries = rng.geometric(1.0 - profile.packet_loss_fraction, size=packets) - 1
    else:
        retries = np.zeros(packets, dtype=np.int64)
    lost = int((retries > 0).sum())
    second_losses = int((retries > 1).sum())
    retransmitted_bytes = sum(
        min(packet_bytes, max(0, len(payload) - index * packet_bytes)) * int(retry_count)
        for index, retry_count in enumerate(retries)
    )
    maximum_retry_rounds = int(retries.max()) if retries.size else 0
    modeled_delay = (
        profile.rtt_ms / 1000.0
        + (len(payload) + retransmitted_bytes) * 8.0 / (profile.uplink_mbps * 1_000_000.0)
        + maximum_retry_rounds * profile.rtt_ms / 1000.0
    )

    sender, receiver = socket.socketpair()
    received = bytearray()
    receive_error: list[BaseException] = []

    def _receive() -> None:
        try:
            while True:
                chunk = receiver.recv(1 << 20)
                if not chunk:
                    break
                received.extend(chunk)
        except BaseException as exc:  # retained and raised in caller
            receive_error.append(exc)
        finally:
            receiver.close()

    thread = threading.Thread(target=_receive, daemon=True)
    started = time.perf_counter()
    thread.start()
    try:
        sender.sendall(payload)
        sender.shutdown(socket.SHUT_WR)
    finally:
        sender.close()
    thread.join(timeout_seconds)
    if thread.is_alive():
        raise TimeoutError("loopback receiver exceeded the frozen timeout")
    if receive_error:
        raise RuntimeError("loopback receiver failed") from receive_error[0]

    remaining = modeled_delay - (time.perf_counter() - started)
    if modeled_delay <= timeout_seconds and remaining > 0:
        time.sleep(remaining)
    elapsed = time.perf_counter() - started
    expected_hash = hashlib.sha256(payload).hexdigest()
    actual_hash = hashlib.sha256(received).hexdigest()
    timed_out = modeled_delay > timeout_seconds or elapsed > timeout_seconds
    return TransferResult(
        payload_bytes=len(payload),
        packets=packets,
        initially_lost_packets=lost,
        second_loss_packets=second_losses,
        retransmitted_bytes=retransmitted_bytes,
        modeled_delay_seconds=modeled_delay,
        actual_elapsed_seconds=elapsed,
        payload_sha256=expected_hash,
        received_sha256=actual_hash,
        correct=expected_hash == actual_hash and len(received) == len(payload),
        timed_out=timed_out,
    )
