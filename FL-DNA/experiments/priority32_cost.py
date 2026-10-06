"""Frozen optimized RQ3 cells plus descriptive DP branch (no old runner edits).

measure_dp follows run_rq3_benchmark._measure_cell's timing/serialization/transfer/
allocation/correctness contract, adding whole-update clip+noise inside client time.
"""
import argparse
import gc
import json
import random
import sys
import time
import tracemalloc
from collections import OrderedDict
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from experiments import run_rq3_benchmark as bench
from experiments.priority32_multidataset import OUT, DATASETS, write, command, progress, event, sha

torch.set_num_threads(1)


def clipped_noisy(state, seed, clip=100., sigma=.001):
    norm = np.sqrt(sum(np.sum(v.astype(np.float64)**2) for v in state.values()))
    factor = min(1., clip/max(norm, 1e-30))
    rng = np.random.default_rng(seed)
    return OrderedDict((key, (v*factor+rng.normal(0., sigma*clip, v.shape)).astype(np.float32)) for key, v in state.items())


def measure_dp(config, profile_name, count):
    profile = bench.NetworkProfile(profile_name, **config["network_profiles"][profile_name])
    settings = config["benchmark"]
    updates = bench._input_updates(count, settings["input_dim"], settings["input_seed"])
    rows = []
    for repetition in range(settings["warmup_repetitions"]+settings["measured_repetitions"]):
        phase = "warmup" if repetition < settings["warmup_repetitions"] else "measured"
        gc.collect(); tracemalloc.start(); tracemalloc.reset_peak()
        round_started = time.perf_counter()
        payloads, expected_states, client_times = [], [], []
        for client, state in enumerate(updates):
            started = time.perf_counter()
            expected = clipped_noisy(state, bench._seed(settings["transform_seed"], client, repetition))
            payload = bench._serialize_binary(expected)
            client_times.append(time.perf_counter()-started)
            payloads.append(payload); expected_states.append(expected)
        _, client_peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
        transfers = [bench.transfer_payload(payload, profile,
                     seed=bench._seed(config["config_id"], profile_name, count, repetition, client),
                     timeout_seconds=settings["timeout_seconds"], packet_bytes=settings["emulation_packet_bytes"])
                     for client, payload in enumerate(payloads)]
        gc.collect(); tracemalloc.start(); tracemalloc.reset_peak()
        server_started = time.perf_counter()
        decoded = [bench._deserialize_binary(payload) for payload in payloads]
        aggregated = bench._aggregate(decoded)
        server_seconds = time.perf_counter()-server_started
        _, server_peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
        correct = all(item.correct for item in transfers) and bench._equal_bits(aggregated, bench._aggregate(expected_states))
        rows.append({"profile": profile_name, "method": "DP_CLIP_NOISE", "client_count": count,
                     "phase": phase, "repetition": repetition,
                     "measured_index": repetition-settings["warmup_repetitions"] if phase == "measured" else None,
                     "client_encode_serialize_seconds_max": max(client_times),
                     "client_encode_serialize_seconds_total": sum(client_times),
                     "server_decode_aggregate_seconds": server_seconds,
                     "transfer_seconds": sum(v.actual_elapsed_seconds for v in transfers),
                     "modeled_transfer_seconds": sum(v.modeled_delay_seconds for v in transfers),
                     "end_to_end_seconds": time.perf_counter()-round_started,
                     "payload_bytes_per_client": list(map(len, payloads)), "payload_bytes_total": sum(map(len, payloads)),
                     "client_peak_alloc_bytes": client_peak, "server_peak_alloc_bytes": server_peak,
                     "initially_lost_packets": sum(v.initially_lost_packets for v in transfers),
                     "second_loss_packets": sum(v.second_loss_packets for v in transfers),
                     "retransmitted_bytes": sum(v.retransmitted_bytes for v in transfers),
                     "authentication_or_correctness_failure": not correct,
                     "timed_out": any(v.timed_out for v in transfers)})
    return {"profile": profile_name, "method": "DP_CLIP_NOISE", "client_count": count, "rows": rows}


def valid_cell(path, config, cell):
    if not path.exists():
        return False
    doc = json.loads(path.read_text())
    assert (doc["profile"], doc["method"], doc["client_count"]) == tuple(cell)
    assert len(doc["rows"]) == 60 and len([r for r in doc["rows"] if r["phase"] == "measured"]) == 50
    assert doc["config_sha256"] == sha(config)
    return True


def orchestrate():
    original = json.loads((ROOT/"protocols/config/rq3_v2_benchmark.json").read_text())
    jobs = []
    for dataset in DATASETS:
        settings = json.loads(json.dumps(original))
        settings["benchmark"]["input_dim"] = json.loads((OUT/"prepared"/dataset/"audit.json").read_text())["feature_count"]
        # Do NOT alter config_id: it participates in frozen emulated packet-loss seeds.
        if dataset == "paysim":
            settings["benchmark"]["methods"] = ["RAW_FLOAT32", "DP_CLIP_NOISE"]
        else:
            settings["benchmark"]["methods"].append("DP_CLIP_NOISE")
        settings["benchmark"]["cells"] = 9*len(settings["benchmark"]["methods"])
        settings["dp_cost_only"] = {"clip_norm": 100., "noise_multiplier": .001, "scope": "whole synthetic trainable vector"}
        config_path = OUT/"rq3"/dataset/"config.json"
        if config_path.exists():
            assert json.loads(config_path.read_text()) == settings
        else:
            write(config_path, settings)
        cells = [(p, m, k) for p in settings["network_profiles"] for m in settings["benchmark"]["methods"] for k in (3, 5, 10)]
        random.Random(271828).shuffle(cells)
        write(OUT/"rq3"/dataset/"cell_order.json", {"seed": 271828, "cells": cells})
        jobs.extend((dataset, config_path, cell) for cell in cells)
    start = time.time()
    for index, (dataset, cfg, cell) in enumerate(jobs, 1):
        p, m, k = cell
        path = OUT/"rq3"/dataset/"cells"/f"{p.lower()}__{m.lower()}__c{k}.json"
        if valid_cell(path, cfg, cell):
            event("rq3_validated_skip", path=str(path))
        else:
            command([sys.executable, "-B", str(Path(__file__).resolve()), "--config", str(cfg), "--cell", p, m, str(k), "--output", str(path)])
            assert valid_cell(path, cfg, cell)
        progress("6_RQ3", dataset, m, index, len(jobs), start=start)
    for dataset in DATASETS:
        command([sys.executable, "-B", str(ROOT/"experiments/analyze_rq3_benchmark.py"),
                 "--config", str(OUT/"rq3"/dataset/"config.json"), "--run", str(OUT/"rq3"/dataset),
                 "--results", str(OUT/"rq3"/dataset/"analysis")])
    write(OUT/"rq3/COMPLETED.json", {"cells": len(jobs), "elapsed_seconds": time.time()-start,
                                   "note": "54 frozen DNA/RAW cells IEEE+BAF;27 DP cells;9 PaySim RAW references"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--orchestrate", action="store_true")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--cell", nargs=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.orchestrate:
        orchestrate()
    else:
        config = json.loads(args.config.read_text())
        profile, method, count = args.cell
        value = measure_dp(config, profile, int(count)) if method == "DP_CLIP_NOISE" else bench._measure_cell(config, profile, method, int(count))
        value["config_sha256"] = sha(args.config)
        write(args.output, value)


if __name__ == "__main__":
    main()
