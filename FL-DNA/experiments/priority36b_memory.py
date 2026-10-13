"""Supplementary RSS harness; fresh exec per repetition/phase, old helpers reused."""
import argparse
from collections import OrderedDict
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import resource
import subprocess
import sys
import threading
import time
import tracemalloc

import numpy as np
import psutil
import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
from experiments import run_rq3_benchmark as bench
from experiments.priority32_cost import clipped_noisy

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/priority36b"
METHODS = ["RAW_FLOAT32", "DNA_TRANSFORM_TRANSPORT", "DNA_TRANSFORM_V2_TRANSPORT", "DP_CLIP_NOISE"]
DATASETS = ["paysim", "ieee_cis", "baf"]


def sha(path):
    with path.open("rb") as handle:
        digest = hashlib.sha256()
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def event(stage, **fields):
    value = {"at": datetime.now(timezone.utc).isoformat(), "stage": stage, "complete": False, **fields}
    write(OUT / "progress.json", value)
    with (OUT / "progress.log").open("a") as log:
        log.write(json.dumps(value) + "\n")


def rss_bytes(value, platform=sys.platform):
    if platform == "darwin":
        return int(value)
    if platform.startswith("linux"):
        return int(value * 1024)
    raise ValueError("ru_maxrss units unsupported on this OS")


def state_hash(state):
    digest = hashlib.sha256()
    for name, value in state.items():
        digest.update(name.encode())
        digest.update(np.ascontiguousarray(value, dtype=np.float32).tobytes())
    return digest.hexdigest()


def finite(states):
    for state in states:
        for name, value in state.items():
            if not np.isfinite(value).all():
                raise ValueError("non-finite tensor " + name)
            if name.endswith("running_var") and (value < 0).any():
                raise ValueError("negative BN running variance")


def prepare_client(config, method, count, repetition):
    b = config["benchmark"]
    updates = bench._input_updates(count, b["input_dim"], b["input_seed"])
    finite(updates)
    v1_config = bench.DNATransformConfig(**config["dna_transform"])
    def work():
        payloads, expected = [], []
        for client, state in enumerate(updates):
            seed = bench._seed(b["transform_seed"], client, repetition)
            if method == "RAW_FLOAT32":
                output, payload = state, bench._serialize_binary(state)
            elif method == "DNA_TRANSFORM_TRANSPORT":
                output = bench._transform(state, v1_config, seed)
                payload = bench._serialize_binary(output)
            elif method == "DNA_TRANSFORM_V2_TRANSPORT":
                payload, output = bench._serialize_v2(state, bench.DNATransformV2Config(**config["dna_transform_v2"], seed=seed), repetition)
            elif method == "DP_CLIP_NOISE":
                output = clipped_noisy(state, seed)
                payload = bench._serialize_binary(output)
            else:
                raise ValueError("unknown method")
            payloads.append(payload)
            expected.append(output)
        return {"updates": updates, "payloads": payloads, "expected": expected}
    return work


def phase(args):
    config = json.loads((OUT / "configs" / (args.dataset + ".json")).read_text())
    if args.phase == "client":
        work = prepare_client(config, args.method, args.count, args.rep)
    else:
        with args.bundle.open("rb") as handle:
            bundle = pickle.load(handle)  # trusted local evidence only
        finite(bundle["updates"] + bundle["expected"])
        def work():
            decoded = [bench._deserialize_v2(p) if args.method == "DNA_TRANSFORM_V2_TRANSPORT" else bench._deserialize_binary(p) for p in bundle["payloads"]]
            return {"decoded": decoded, "aggregate": bench._aggregate(decoded)}
    gc.collect()
    before = rss_bytes(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    baseline = psutil.Process().memory_info().rss
    print("READY", flush=True)
    if sys.stdin.readline().strip() != "GO":
        raise RuntimeError("sampler handshake missing")
    tracemalloc.start()
    tracemalloc.reset_peak()
    started = time.perf_counter()
    result = work()
    seconds = time.perf_counter() - started
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    maximum = rss_bytes(raw)
    print("DONE", flush=True)
    # Evidence serialization/correctness audit lies outside measured phase.
    if args.phase == "client":
        finite(result["expected"])
        with args.bundle.open("xb") as handle:
            pickle.dump(result, handle, protocol=4)
        digest = state_hash(bench._aggregate(result["expected"]))
    else:
        finite(result["decoded"] + [result["aggregate"]])
        expected = bench._aggregate(bundle["expected"])
        if not bench._equal_bits(expected, result["aggregate"]):
            raise ValueError("decoded aggregate bytes differ")
        digest = state_hash(result["aggregate"])
    print(json.dumps({"phase": args.phase, "pid": os.getpid(), "seconds": seconds, "rss_peak_bytes": maximum,
                      "rss_before_peak_bytes": before, "rss_baseline_bytes": baseline, "ru_maxrss_raw": raw,
                      "rss_units": "bytes" if sys.platform == "darwin" else "KiB", "trace_peak_bytes": peak,
                      "trace_current_bytes": current, "aggregate_sha256": digest,
                      "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads()}), flush=True)


def execute_phase(dataset, profile, method, count, rep, name, folder):
    cmd = [sys.executable, "-B", "-m", "experiments.priority36b_memory", "--phase", name,
           "--dataset", dataset, "--method", method, "--count", str(count), "--rep", str(rep), "--bundle", str(folder / "bundle.pkl")]
    stderr = folder / (name + ".stderr.log")
    with stderr.open("x") as err:
        child = subprocess.Popen(cmd, cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err, text=True)
        creation = psutil.Process(child.pid).create_time()
        write(folder / (name + "_process.json"), {"pid": child.pid, "created": creation, "command": cmd})
        if child.stdout.readline().strip() != "READY":
            child.wait()
            raise RuntimeError("phase initialization failed; see " + str(stderr))
        target = psutil.Process(child.pid)
        samples = [(time.perf_counter(), target.memory_info().rss)]
        stop = threading.Event()
        def sample():
            while not stop.is_set():
                try:
                    samples.append((time.perf_counter(), target.memory_info().rss))
                except psutil.NoSuchProcess:
                    break
                stop.wait(.0005)
        sampler = threading.Thread(target=sample)
        sampler.start()
        child.stdin.write("GO\n")
        child.stdin.flush()
        end = child.stdout.readline().strip()
        try:
            samples.append((time.perf_counter(), target.memory_info().rss))
        except psutil.NoSuchProcess:
            pass  # child can exit after DONE before parent consumes the pipe
        stop.set()
        sampler.join()
        if end != "DONE":
            child.wait()
            raise RuntimeError("phase failed during measurement; see " + str(stderr))
        body = child.stdout.read()
        code = child.wait()
        child.stdin.close()
        child.stdout.close()
        if code:
            raise RuntimeError("phase correctness failure; see " + str(stderr))
    result = json.loads(body)
    write(folder / (name + "_psutil_samples.json"), samples)
    assert result["pid"] == child.pid and result["torch_threads"] == result["torch_interop_threads"] == 1
    result.update({"psutil_peak_bytes": max(x[1] for x in samples), "psutil_samples": len(samples),
                   "psutil_max_gap_seconds": max((b[0]-a[0] for a,b in zip(samples,samples[1:])), default=0),
                   "returncode": code, "exited": True, "process_created": creation})
    if not all(np.isfinite(v) for v in result.values() if isinstance(v, (int,float))):
        raise ValueError("non-finite observation")
    write(folder / (name + "_result.json"), result)
    return result


def verify_freeze():
    freeze = json.loads((OUT / "freeze.json").read_text())
    for section in ["sources", "inputs"]:
        for name, expected in freeze[section].items():
            if sha(ROOT / name) != expected:
                raise ValueError("frozen hash drift: " + name)


def supervise():
    verify_freeze()
    start = time.time()
    done = 0
    for dataset in DATASETS:
        config = json.loads((OUT / "configs" / (dataset + ".json")).read_text())
        cells = [(p,m,k) for p in config["network_profiles"] for m in METHODS for k in [3,5,10]]
        random.Random(271828).shuffle(cells)
        for cell_index, (profile, method, count) in enumerate(cells):
            verify_freeze()
            rows = []
            cell = OUT / "cells" / dataset / (profile + "__" + method + "__" + str(count))
            for rep in range(60):
                folder = cell / ("rep_%02d" % rep)
                folder.mkdir(parents=True, exist_ok=False)
                client = execute_phase(dataset, profile, method, count, rep, "client", folder)
                with (folder / "bundle.pkl").open("rb") as handle:
                    bundle = pickle.load(handle)
                network = bench.NetworkProfile(profile, **config["network_profiles"][profile])
                b = config["benchmark"]
                transfers = [bench.transfer_payload(payload, network, seed=bench._seed(config["config_id"], profile, count, rep, i),
                             timeout_seconds=b["timeout_seconds"], packet_bytes=b["emulation_packet_bytes"]) for i,payload in enumerate(bundle["payloads"])]
                if any(not t.correct or t.timed_out for t in transfers):
                    raise ValueError("network timeout or correctness failure")
                write(folder / "transfer.json", [asdict(t) for t in transfers])
                del bundle
                gc.collect()
                server = execute_phase(dataset, profile, method, count, rep, "server", folder)
                assert client["aggregate_sha256"] == server["aggregate_sha256"]
                row = {"dataset": dataset, "profile": profile, "method": method, "client_count": count,
                       "repetition": rep, "phase": "warmup" if rep < 10 else "measured", "measured_index": rep-10 if rep >= 10 else -1,
                       "bundle_sha256": sha(folder / "bundle.pkl"), "client": client, "server": server}
                write(folder / "observation.json", row)
                rows.append(row)
                done += 1
                elapsed = time.time()-start
                event("MEASUREMENTS_RUNNING", rounds_done=done, rounds_total=6480, cells_done=done//60,
                      dataset=dataset, profile=profile, method=method, repetition=rep,
                      elapsed_seconds=elapsed, provisional_eta_seconds=elapsed/done*(6480-done))
            write(cell / "cell.json", {"dataset": dataset, "profile": profile, "method": method, "client_count": count, "rows": rows})
    verify_freeze()
    write(OUT / "MEASUREMENTS_FINISHED.json", {"rounds": done, "cells": 108, "all_phase_children_waited": True, "supervisor_pid": os.getpid()})
    event("MEASUREMENTS_FINISHED_AWAITING_POSTEXIT_VERIFIER", rounds_done=done, rounds_total=6480)


def wrapper():
    cmd = [sys.executable, "-B", "-m", "experiments.priority36b_memory", "--supervise"]
    with (OUT / "supervisor.stdout.log").open("x") as out, (OUT / "supervisor.stderr.log").open("x") as err:
        child = subprocess.Popen(cmd, cwd=ROOT, stdout=out, stderr=err)
        write(OUT / "supervisor_launch.json", {"pid": child.pid, "created": psutil.Process(child.pid).create_time(), "command": cmd})
        code = child.wait()
    if code:
        event("REQUIRES_DIRECTION", supervisor_returncode=code, failure="supervisor.stderr.log; no replay authorized")
        return
    write(OUT / "supervisor_exit.json", {"pid": child.pid, "returncode": code, "waited": True})
    subprocess.run([sys.executable, "-B", "-m", "experiments.priority36b_finish"], cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["client", "server"])
    parser.add_argument("--dataset")
    parser.add_argument("--method")
    parser.add_argument("--count", type=int)
    parser.add_argument("--rep", type=int)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--supervise", action="store_true")
    parser.add_argument("--wrapper", action="store_true")
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    try:
        if args.phase:
            phase(args)
        elif args.supervise:
            supervise()
        elif args.wrapper:
            wrapper()
        elif args.launch:
            verify_freeze()
            if (OUT / "launch.json").exists():
                raise FileExistsError("no duplicate launch")
            env = dict(os.environ)
            for name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]:
                env[name] = "1"
            with (OUT / "detached.stdout.log").open("x") as out, (OUT / "detached.stderr.log").open("x") as err:
                cmd = [sys.executable, "-B", "-m", "experiments.priority36b_memory", "--wrapper"]
                proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=out, stderr=err, start_new_session=True)
            write(OUT / "launch.json", {"pid": proc.pid, "created": psutil.Process(proc.pid).create_time(), "command": cmd, "env_threads": {k:v for k,v in env.items() if "THREADS" in k}, "detached": True})
            print(json.dumps({"launched": proc.pid, "rounds": 6480, "cells": 108}))
        else:
            parser.error("choose phase/supervise/wrapper/launch")
    except BaseException as exc:
        if not args.phase:
            write(OUT / ("FAILURE_%d.json" % time.time_ns()), {"error": repr(exc), "pid": os.getpid()})
        raise


if __name__ == "__main__":
    main()
