"""Frozen Priority34A: local BN, no BN transmission, CPU/thread1."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import multiprocessing as mp
import os
import platform
import random
import sys
import time
import traceback
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import priority32_multidataset as p

np, torch = p.np, p.torch
OUT = ROOT / "artifacts/priority34a"
PROTOCOL = ROOT / "protocols/amendments/2026-10-03_priority34a_local_bn.md"
RAW_BN = p.OUT / "repair_trainable_raw_bn_20261002"
SOURCES = ["experiments/priority34a_local_bn.py", "experiments/priority34a_analysis.py",
           "tests/test_priority34a_local_bn.py", "experiments/priority32_multidataset.py",
           "experiments/fraud_fl_common.py", "models/fraud_mlp.py", "data/load_creditcard.py",
           "experiments/run_fraud_fl_dna_transform.py", "experiments/run_fraud_fl_dna_transform_v2.py",
           "dna_encoder/transform_defense.py", "dna_encoder/transform_defense_v2.py",
           "privacy/seed_manager.py", str(PROTOCOL.relative_to(ROOT))]


def write(path, value):
    path = Path(path)
    if OUT.resolve() not in path.resolve().parents:
        raise ValueError("P34A writes restricted to its own namespace")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def event(kind, **fields):
    record = {"at": p.now(), "event": kind, **fields}
    with (OUT / "runs.jsonl").open("a") as handle:
        handle.write(json.dumps(record, allow_nan=False) + "\n")
    print(json.dumps(record), flush=True)


def progress(part, done=0, failed=0, dataset="all", defense="all", start=None):
    elapsed = time.time() - start if start else 0
    doc = {"part": part, "dataset": dataset, "defense": defense,
           "comparator": "local_bn_baseline", "done": done, "total": 189,
           "failed": failed, "last_update": p.now(),
           "eta_minutes": elapsed / (done + failed) * (189-done-failed) / 60 if start and done+failed else None}
    write(OUT / "progress.json", doc)
    with (OUT / "progress.log").open("a") as handle:
        handle.write(json.dumps(doc) + "\n")


def checklist(item, status):
    doc = json.loads((OUT / "checklist.json").read_text())
    doc[str(item)].update(status=status, last_update=p.now())
    write(OUT / "checklist.json", doc)


def domains(model):
    bn = set()
    for prefix, module in model.named_modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            bn.update(prefix + "." + name for name in module.state_dict())
    allowed = tuple(name for name, _ in model.named_parameters() if name not in bn)
    if not bn or not allowed:
        raise ValueError("unexpected model domains")
    return frozenset(bn), allowed


def assert_payload(state, bn, allowed):
    if tuple(state) != tuple(allowed) or set(state) & bn:
        raise ValueError("BN/payload firewall violation")
    for name, value in state.items():
        if value.device.type != "cpu" or not torch.isfinite(value).all():
            raise ValueError("nonfinite/nonCPU payload: " + name)


def guard(model, location):
    minima = {}
    for name, value in model.state_dict().items():
        if torch.is_floating_point(value) and not torch.isfinite(value).all():
            raise ValueError("nonfinite state " + name + " at " + location)
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            minimum = float(module.running_var.min())
            if minimum < 0:
                raise ValueError(f"negative BN running_var {name}={minimum} at {location}")
            minima[name] = minimum
    return minima


def broadcast(model, global_parameters, bn, allowed):
    assert_payload(global_parameters, bn, allowed)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name in global_parameters:
                parameter.copy_(global_parameters[name])


def upload(model, global_parameters, method, seed, round_number, client, bn, allowed):
    state = OrderedDict((name, model.state_dict()[name].detach().clone()) for name in allowed)
    assert_payload(state, bn, allowed)
    if method == "dna_v1_conservative":
        key = p.derive_seed(p.derive_seed(seed, "rq2-dna-transform"), "dna_transform", round_number, client)
        state, _ = p.dna_transform_state(state, global_parameters, p.DNATransformConfig(
            block_size=256, mix_ratio=.08, keep_ratio=.88, shrink_factor=.45, seed=key))
    elif method == "dna_v2_0p95":
        key = p.derive_seed(p.derive_seed(seed, "rq2-dna-transform-v2"), "dna_transform_v2", round_number, client)
        state, _ = p.dna_transform_v2_state(state, global_parameters, p.DNATransformV2Config(
            compression_ratio=.95, quantization_eta=.01, seed=key), p.derive_seed(key, "quantization"))
    elif method != "baseline":
        raise ValueError("unknown method")
    assert_payload(state, bn, allowed)
    return OrderedDict((name, value.detach().clone()) for name, value in state.items())


def checked_probabilities(model, array):
    guard(model, "evaluation")
    model.eval()
    blocks = []
    with torch.no_grad():
        for begin in range(0, len(array), 4096):
            logits = model(torch.from_numpy(np.array(array[begin:begin+4096], copy=True)))
            if not torch.isfinite(logits).all():
                raise ValueError("nonfinite evaluation logits")
            blocks.append(torch.sigmoid(logits).reshape(-1).numpy())
    result = np.concatenate(blocks)
    if not np.isfinite(result).all():
        raise ValueError("nonfinite probabilities")
    return result


def valid_result(path, config):
    path = Path(path)
    if not path.exists():
        return False
    doc = json.loads(path.read_text())
    if doc.get("status") != "COMPLETED" or doc.get("config_sha256") != p.canonical_sha(config):
        raise ValueError("existing output/config mismatch; do not overwrite " + str(path))
    if not doc["no_bn_transmitted"] or doc["payload_assertions"] != config["training"]["rounds"]*3:
        raise ValueError("payload assertion count failed")
    if len(doc["clients"]) != 3 or doc["torch_threads"] != 1:
        raise ValueError("client/thread contract failed")
    for split in ("validation", "test"):
        for endpoint in ("f1", "auc_roc", "pr_auc"):
            values = [client[split][endpoint] for client in doc["clients"]]
            if not np.isfinite(values).all() or not np.isclose(np.mean(values), doc[split][endpoint], atol=1e-14, rtol=0):
                raise ValueError("nonfinite or incorrect mean-client metrics")
    for relative, digest in doc["artifact_sha256"].items():
        if p.sha(OUT / relative) != digest:
            raise ValueError("changed output " + relative)
    return True


def train_job(config, path_string):
    path = Path(path_string)
    if valid_result(path, config):
        return {"path": str(path), "skipped": True}
    started = time.time()
    seed, dataset, method = config["seed"], config["dataset"], config["method"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(1)
    prepared = p.OUT / "prepared" / dataset
    x = np.load(prepared / "train_x.npy", mmap_mode="r")
    y = np.load(prepared / "train_y.npy", mmap_mode="r")
    categories = np.load(prepared / "train_categories.npy")
    partitions = p._mild_non_iid_client_indices(p.pd.DataFrame({"type": categories}), y, 3, seed)
    if np.unique(np.concatenate(partitions)).size != len(y) or sum(map(len, partitions)) != len(y):
        raise ValueError("partition coverage failed")
    if any(len(indices)%1024 == 1 for indices in partitions):
        raise ValueError("singleton batch; requires direction")
    loaders = [p.DataLoader(p.TensorDataset(torch.from_numpy(np.array(x[indices], copy=True)),
              torch.from_numpy(np.array(y[indices], copy=True)[:, None])), batch_size=1024,
              shuffle=True, num_workers=0, drop_last=False,
              generator=torch.Generator().manual_seed(seed+client)) for client, indices in enumerate(partitions)]
    initial = p.FraudMLP(x.shape[1]).cpu()
    bn, allowed = domains(initial)
    guard(initial, "initialization")
    clients = [copy.deepcopy(initial) for _ in range(3)]
    global_parameters = OrderedDict((name, initial.state_dict()[name].clone()) for name in allowed)
    loss_function = p.BinaryFocalLoss(alpha=config["training"]["alpha"], gamma=2.)
    # New attempt journal retains any interrupted partial rather than truncating it.
    attempt = path.parent / (path.stem + ".attempt_" + str(time.time_ns()))
    attempt.mkdir(parents=True, exist_ok=False)
    journal = attempt / "rounds.jsonl"
    for round_number in range(1, config["training"]["rounds"]+1):
        states, bn_minima, trajectory_minima = [], [], []
        for client, (local, loader) in enumerate(zip(clients, loaders)):
            broadcast(local, global_parameters, bn, allowed)
            local.train()
            minimum = guard(local, f"round{round_number}/client{client}/broadcast")
            optimizer = torch.optim.Adam(local.parameters(), lr=config["training"]["lr"])
            for batch, (features, labels) in enumerate(loader):
                optimizer.zero_grad()
                logits = local(features)
                if not torch.isfinite(logits).all():
                    raise ValueError(f"nonfinite logits round{round_number}/client{client}/batch{batch}")
                loss = loss_function(logits, labels)
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite loss")
                loss.backward()
                if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all() for parameter in local.parameters()):
                    raise ValueError("nonfinite gradient")
                optimizer.step()
                current = guard(local, f"round{round_number}/client{client}/batch{batch}")
                minimum = {name: min(minimum[name], value) for name, value in current.items()}
            bn_minima.append(guard(local, f"round{round_number}/client{client}/trained"))
            trajectory_minima.append(minimum)
            states.append(upload(local, global_parameters, method, seed, round_number, client, bn, allowed))
        global_parameters = p.fed_avg(states, list(map(len, partitions)))
        assert_payload(global_parameters, bn, allowed)
        with journal.open("a") as handle:
            handle.write(json.dumps({"at": p.now(), "round": round_number,
                "bn_minima_per_client": bn_minima, "bn_minima_all_steps": trajectory_minima,
                "transmitted_keys": list(allowed), "bn_keys": sorted(bn),
                "no_bn_transmitted": True, "upload_assertions": 3,
                "elapsed_seconds": time.time()-started}, allow_nan=False) + "\n")
    client_docs, artifacts = [], [journal]
    for client, local in enumerate(clients):
        broadcast(local, global_parameters, bn, allowed)
        values = {}
        for split in ("validation", "test"):
            features = np.load(prepared / (split+"_x.npy"), mmap_mode="r")
            labels = np.load(prepared / (split+"_y.npy"))
            probability = checked_probabilities(local, features)
            probability_path = attempt / f"client{client}_{split}_probabilities.npy"
            np.save(probability_path, probability, allow_pickle=False)
            artifacts.append(probability_path)
            if split == "validation":
                threshold = float(p.tune_threshold(labels, probability))
            values[split] = p.metrics(labels, probability, threshold)
        client_docs.append({"client": client, "threshold": threshold, **values,
                            "bn_minima": guard(local, "final")})
    checkpoint = attempt / "final_checkpoint.pt"
    torch.save({"global_non_bn": global_parameters,
                "client_bn": [OrderedDict((name, local.state_dict()[name].clone()) for name in sorted(bn)) for local in clients]}, checkpoint)
    artifacts.append(checkpoint)
    doc = {"status": "COMPLETED", "config": config, "config_sha256": p.canonical_sha(config),
           "dataset": dataset, "method": method, "seed": seed, "clients": client_docs,
           "client_rows": list(map(len, partitions)), "client_fraud_counts": [int(y[indices].sum()) for indices in partitions],
           "partition_sha256": [hashlib.sha256(indices.tobytes()).hexdigest() for indices in partitions],
           "payload_keys": list(allowed), "bn_keys": sorted(bn), "no_bn_transmitted": True,
           "payload_assertions": config["training"]["rounds"]*3, "torch_threads": torch.get_num_threads(),
           "artifact_sha256": {str(item.relative_to(OUT)): p.sha(item) for item in artifacts},
           "elapsed_seconds": time.time()-started, "completed_at": p.now()}
    for split in ("validation", "test"):
        doc[split] = {endpoint: float(np.mean([client[split][endpoint] for client in client_docs]))
                      for endpoint in ("f1", "auc_roc", "pr_auc")}
    write(path, doc)
    valid_result(path, config)
    return {"path": str(path), "skipped": False}


def freeze():
    if (OUT / "execution_freeze.json").exists():
        raise ValueError("freeze already exists; immutable")
    original = json.loads((p.OUT / "execution_freeze.json").read_text())
    inputs = {str((p.OUT / "execution_freeze.json").relative_to(ROOT)): p.sha(p.OUT / "execution_freeze.json")}
    for dataset in p.DATASETS:
        prepared = p.OUT / "prepared" / dataset
        audit = json.loads((prepared / "audit.json").read_text())
        pairs = {**audit["raw_sha256"], **{str((prepared / item).relative_to(ROOT)): digest for item, digest in audit["prepared_sha256"].items()}}
        pairs[str((prepared / "audit.json").relative_to(ROOT))] = p.sha(prepared / "audit.json")
        inputs.update(pairs)
    for result in sorted((RAW_BN / "rq2_jobs").rglob("*.json")):
        inputs[str(result.relative_to(ROOT))] = p.sha(result)
    inputs[str((ROOT / "reports/priority32_repaired_trainable_rq2_rq3_report_20261002.md").relative_to(ROOT))] = p.sha(ROOT / "reports/priority32_repaired_trainable_rq2_rq3_report_20261002.md")
    for relative, digest in inputs.items():
        if p.sha(ROOT / relative) != digest:
            raise ValueError("input integrity failure " + relative)
    sources = {relative: p.sha(ROOT / relative) for relative in SOURCES}
    jobs = []
    for dataset in p.DATASETS:
        for method in p.METHODS:
            for seed in p.SEEDS:
                config = p.job_config(dataset, method, seed, original["chosen"][dataset]["training"])
                config.update(protocol_variant="local_bn_no_transmitted_bn", evaluation="mean_of_three_client_metrics_per_seed",
                              local_thresholds="per_client_validation_F1", source_sha256=sources,
                              p34_protocol_sha256=p.sha(PROTOCOL))
                jobs.append({"config": config, "result": f"jobs/{dataset}/{method}/{seed}.json"})
    doc = {"status": "FROZEN", "frozen_at": p.now(), "jobs": jobs,
           "inputs": inputs, "sources": sources, "workers": 4, "paired_n": 21,
           "environment": {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
                           "platform": platform.platform(), "device": "cpu", "torch_threads": 1}}
    write(OUT / "execution_freeze.json", doc)
    checklist(2, "complete")
    event("execution_frozen", jobs=len(jobs), input_files=len(inputs), sources=len(sources))


def verify_freeze():
    doc = json.loads((OUT / "execution_freeze.json").read_text())
    for relative, digest in {**doc["inputs"], **doc["sources"]}.items():
        if p.sha(ROOT / relative) != digest:
            raise ValueError("frozen input/source changed: " + relative)
    return doc


def supervise():
    OUT.mkdir(parents=True, exist_ok=True)
    # flock owns supervisor lifetime; stale file contents never justify a duplicate.
    import fcntl
    with (OUT / "supervisor.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write(OUT / "supervisor.lock.json", {"pid": os.getpid(), "started_at": p.now(), "executable": sys.executable})
        manifest = verify_freeze()
        jobs, completed = [], 0
        for job in manifest["jobs"]:
            if valid_result(OUT / job["result"], job["config"]):
                completed += 1
            else:
                if (OUT / job["result"]).with_suffix(".failure.json").exists():
                    raise ValueError("recorded failure requires human direction, not automatic replay")
                jobs.append(job)
        random.Random(320032).shuffle(jobs)
        start, failed = time.time(), []
        event("supervisor_started", completed=completed, remaining=len(jobs), command=sys.argv)
        checklist(3, "in_progress")
        progress("training", completed, start=start)
        with ProcessPoolExecutor(max_workers=4, mp_context=mp.get_context("spawn")) as pool:
            futures = {pool.submit(train_job, job["config"], str(OUT / job["result"])): job for job in jobs}
            for future in as_completed(futures):
                job = futures[future]
                try:
                    receipt = future.result()
                    completed += 1
                    event("job_completed", **receipt)
                except Exception:
                    failure = {"config": job["config"], "result": job["result"], "traceback": traceback.format_exc(), "at": p.now()}
                    failed.append(failure)
                    write((OUT / job["result"]).with_suffix(".failure.json"), failure)
                    event("job_failed", result=job["result"], error=failure["traceback"])
                progress("training", completed, len(failed), job["config"]["dataset"], job["config"]["method"], start)
        if failed:
            write(OUT / "REQUIRES_DIRECTION.json", {"completed": completed, "missing": 189-completed, "failed": failed})
            checklist(3, "requires_direction")
            progress("requires_direction", completed, len(failed))
            return
        checklist(3, "complete")
        progress("analysis_and_checks", completed)
        from experiments.priority34a_analysis import finish
        finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--supervise", action="store_true")
    args = parser.parse_args()
    if args.freeze:
        freeze()
    elif args.supervise:
        try:
            supervise()
        except Exception:
            write(OUT / "SUPERVISOR_INTERRUPTION.json", {"at": p.now(), "traceback": traceback.format_exc()})
            raise
    else:
        parser.error("choose --freeze or --supervise")


if __name__ == "__main__":
    main()
