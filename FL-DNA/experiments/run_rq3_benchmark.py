"""Run the frozen RQ3 deployment-cost benchmark, one cell at a time."""

from __future__ import annotations

import argparse
import base64
import gc
import hashlib
import json
import os
import platform
import random
import struct
import subprocess
import sys
import time
import tracemalloc
from collections import OrderedDict
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import psutil
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dna_encoder.encoder import DNAEncoder
from dna_encoder.transform_defense import DNATransformConfig, transform_update_array
from experiments.rq3_user_space_network import NetworkProfile, transfer_payload
from models.fraud_mlp import FraudMLP

torch.set_num_threads(1)

METHODS = ("RAW_FLOAT32", "LOSSLESS_DNA_AES_GCM", "DNA_TRANSFORM_TRANSPORT")


def _compact_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _serialize_binary(tensors: OrderedDict[str, np.ndarray]) -> bytes:
    header = []
    chunks = []
    for name, value in tensors.items():
        array = np.ascontiguousarray(value, dtype=np.float32)
        raw = array.tobytes(order="C")
        header.append({"name": name, "shape": list(array.shape), "dtype": "float32", "nbytes": len(raw)})
        chunks.append(raw)
    encoded_header = _compact_json(header)
    return struct.pack(">Q", len(encoded_header)) + encoded_header + b"".join(chunks)


def _deserialize_binary(payload: bytes) -> OrderedDict[str, np.ndarray]:
    if len(payload) < 8:
        raise ValueError("binary payload lacks header length")
    header_size = struct.unpack(">Q", payload[:8])[0]
    header = json.loads(payload[8 : 8 + header_size])
    cursor = 8 + header_size
    output: OrderedDict[str, np.ndarray] = OrderedDict()
    for item in header:
        stop = cursor + int(item["nbytes"])
        raw = payload[cursor:stop]
        if len(raw) != int(item["nbytes"]):
            raise ValueError("binary payload is truncated")
        output[item["name"]] = np.frombuffer(raw, dtype=np.float32).copy().reshape(item["shape"])
        cursor = stop
    if cursor != len(payload):
        raise ValueError("binary payload contains trailing bytes")
    return output


def _serialize_lossless(tensors: OrderedDict[str, np.ndarray], encoder: DNAEncoder) -> bytes:
    entries = []
    for name, value in tensors.items():
        array = np.ascontiguousarray(value, dtype=np.float32)
        entries.append({
            "name": name,
            "shape": list(array.shape),
            "dtype": "float32",
            "encrypted_dna": encoder.encode_array(array),
        })
    return _compact_json({"format": "dna-aes-256-gcm-base64-json-v1", "tensors": entries})


def _deserialize_lossless(payload: bytes, encoder: DNAEncoder) -> OrderedDict[str, np.ndarray]:
    document = json.loads(payload)
    if document.get("format") != "dna-aes-256-gcm-base64-json-v1":
        raise ValueError("unexpected lossless DNA wire format")
    output: OrderedDict[str, np.ndarray] = OrderedDict()
    for item in document["tensors"]:
        output[item["name"]] = encoder.decode_array(item["encrypted_dna"], item["shape"])
    return output


def _transform(
    tensors: OrderedDict[str, np.ndarray], config: DNATransformConfig, seed: int
) -> OrderedDict[str, np.ndarray]:
    output: OrderedDict[str, np.ndarray] = OrderedDict()
    for tensor_index, (name, value) in enumerate(tensors.items()):
        transformed, _ = transform_update_array(value, replace(config, seed=seed), tensor_index=tensor_index)
        output[name] = transformed
    return output


def _aggregate(states: list[OrderedDict[str, np.ndarray]]) -> OrderedDict[str, np.ndarray]:
    output: OrderedDict[str, np.ndarray] = OrderedDict()
    for name in states[0]:
        output[name] = np.mean(np.stack([state[name] for state in states], axis=0), axis=0, dtype=np.float32)
    return output


def _equal_bits(left: OrderedDict[str, np.ndarray], right: OrderedDict[str, np.ndarray]) -> bool:
    return list(left) == list(right) and all(
        left[name].shape == right[name].shape and left[name].tobytes() == right[name].tobytes()
        for name in left
    )


def _input_updates(client_count: int, input_dim: int, seed: int) -> list[OrderedDict[str, np.ndarray]]:
    model = FraudMLP(input_dim)
    shapes = [(name, tuple(parameter.shape)) for name, parameter in model.named_parameters()]
    clients = []
    for client_index in range(client_count):
        rng = np.random.default_rng(seed + client_index)
        state: OrderedDict[str, np.ndarray] = OrderedDict()
        for name, shape in shapes:
            state[name] = rng.normal(0.0, 0.01, size=shape).astype(np.float32)
        clients.append(state)
    return clients


def _seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:4], "big")


def _measure_cell(config: dict, profile_name: str, method: str, client_count: int) -> dict:
    profile_cfg = config["network_profiles"][profile_name]
    profile = NetworkProfile(profile_name, profile_cfg["uplink_mbps"], profile_cfg["rtt_ms"], profile_cfg["packet_loss_fraction"])
    benchmark = config["benchmark"]
    updates = _input_updates(client_count, benchmark["input_dim"], benchmark["input_seed"])
    key = bytes.fromhex(benchmark["aes_256_key_hex"])
    encoder = DNAEncoder(key)
    transform_cfg = DNATransformConfig(**config["dna_transform"])
    rows = []
    total = benchmark["warmup_repetitions"] + benchmark["measured_repetitions"]
    for repetition in range(total):
        phase = "warmup" if repetition < benchmark["warmup_repetitions"] else "measured"
        measured_index = repetition - benchmark["warmup_repetitions"]
        gc.collect()
        tracemalloc.start()
        tracemalloc.reset_peak()
        round_started = time.perf_counter()
        payloads: list[bytes] = []
        expected_states: list[OrderedDict[str, np.ndarray]] = []
        client_times = []
        for client_index, state in enumerate(updates):
            started = time.perf_counter()
            if method == "RAW_FLOAT32":
                expected = state
                payload = _serialize_binary(state)
            elif method == "LOSSLESS_DNA_AES_GCM":
                expected = state
                payload = _serialize_lossless(state, encoder)
            elif method == "DNA_TRANSFORM_TRANSPORT":
                expected = _transform(state, transform_cfg, _seed(benchmark["transform_seed"], client_index, repetition))
                payload = _serialize_binary(expected)
            else:
                raise ValueError(f"unknown method {method}")
            client_times.append(time.perf_counter() - started)
            payloads.append(payload)
            expected_states.append(expected)
        client_current, client_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        transfer_results = []
        received_payloads = []
        for client_index, payload in enumerate(payloads):
            result = transfer_payload(
                payload,
                profile,
                seed=_seed(config["config_id"], profile_name, client_count, repetition, client_index),
                timeout_seconds=benchmark["timeout_seconds"],
                packet_bytes=benchmark["emulation_packet_bytes"],
            )
            transfer_results.append(result)
            received_payloads.append(payload)

        gc.collect()
        tracemalloc.start()
        tracemalloc.reset_peak()
        server_started = time.perf_counter()
        if method == "LOSSLESS_DNA_AES_GCM":
            decoded = [_deserialize_lossless(payload, encoder) for payload in received_payloads]
        else:
            decoded = [_deserialize_binary(payload) for payload in received_payloads]
        aggregated = _aggregate(decoded)
        server_seconds = time.perf_counter() - server_started
        server_current, server_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        expected_aggregate = _aggregate(expected_states)
        correct = all(result.correct for result in transfer_results) and _equal_bits(aggregated, expected_aggregate)
        timed_out = any(result.timed_out for result in transfer_results)
        row = {
            "profile": profile_name,
            "method": method,
            "client_count": client_count,
            "phase": phase,
            "repetition": repetition,
            "measured_index": measured_index if phase == "measured" else None,
            "client_encode_serialize_seconds_max": max(client_times),
            "client_encode_serialize_seconds_total": sum(client_times),
            "server_decode_aggregate_seconds": server_seconds,
            "transfer_seconds": sum(item.actual_elapsed_seconds for item in transfer_results),
            "modeled_transfer_seconds": sum(item.modeled_delay_seconds for item in transfer_results),
            "end_to_end_seconds": time.perf_counter() - round_started,
            "payload_bytes_per_client": [len(payload) for payload in payloads],
            "payload_bytes_total": sum(map(len, payloads)),
            "client_peak_alloc_bytes": client_peak,
            "server_peak_alloc_bytes": server_peak,
            "initially_lost_packets": sum(item.initially_lost_packets for item in transfer_results),
            "second_loss_packets": sum(item.second_loss_packets for item in transfer_results),
            "retransmitted_bytes": sum(item.retransmitted_bytes for item in transfer_results),
            "authentication_or_correctness_failure": not correct,
            "timed_out": timed_out,
        }
        rows.append(row)
    return {"profile": profile_name, "method": method, "client_count": client_count, "rows": rows}


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _run_cell_subprocess(script: Path, config_path: Path, output: Path, cell: tuple[str, str, int]) -> None:
    profile, method, client_count = cell
    cell_path = output / "cells" / f"{profile.lower()}__{method.lower()}__c{client_count}.json"
    command = [sys.executable, str(script), "--config", str(config_path), "--cell", profile, method, str(client_count), "--cell-output", str(cell_path)]
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    log = {"command": command, "returncode": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr}
    _write_json(cell_path.with_suffix(".log.json"), log)
    if completed.returncode != 0:
        raise RuntimeError(f"RQ3 cell failed: {cell}; see {cell_path.with_suffix('.log.json')}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cell", nargs=3)
    parser.add_argument("--cell-output", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))

    if args.cell:
        if args.cell_output is None:
            parser.error("--cell-output is required with --cell")
        profile, method, count = args.cell
        _write_json(args.cell_output, _measure_cell(config, profile, method, int(count)))
        return

    if args.output is None:
        parser.error("--output is required for an orchestrated run")
    if args.output.exists():
        raise FileExistsError("official output directory already exists; refusing a second execution")
    args.output.mkdir(parents=True)
    _write_json(args.output / "environment.json", {
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cpu_count": os.cpu_count(),
        "physical_cpu_count": psutil.cpu_count(logical=False),
        "memory_bytes": psutil.virtual_memory().total,
        "torch_num_threads": torch.get_num_threads(),
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
    })
    cells = [(profile, method, count) for profile in config["network_profiles"] for method in METHODS for count in config["benchmark"]["client_counts"]]
    random.Random(config["benchmark"]["method_order_seed"]).shuffle(cells)
    _write_json(args.output / "cell_order.json", {"seed": config["benchmark"]["method_order_seed"], "cells": cells})
    script = Path(__file__).resolve()
    for index, cell in enumerate(cells, start=1):
        print(f"[{index}/{len(cells)}] {cell}", flush=True)
        _run_cell_subprocess(script, args.config.resolve(), args.output.resolve(), cell)
    _write_json(args.output / "COMPLETED.json", {"cells": len(cells), "completed_at_unix": time.time()})


if __name__ == "__main__":
    main()
