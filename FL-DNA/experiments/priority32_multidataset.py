"""Priority32 immutable-input preprocessing, CPU FL jobs and resumable orchestration.

Reuses frozen FraudMLP, focal loss, FedAvg, threshold and transform functions.
Only new Priority32 artifacts are writable. No reconstruction attacks.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import OrderedDict
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_variable] = "1"

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader, TensorDataset

from data.load_creditcard import BASE_FEATURE_COLUMNS, NUMERIC_COLUMNS, _build_features, _mild_non_iid_client_indices
from dna_encoder.transform_defense import DNATransformConfig
from dna_encoder.transform_defense_v2 import DNATransformV2Config
from experiments.fraud_fl_common import BinaryFocalLoss, fed_avg, tune_threshold
from experiments.run_fraud_fl_dna_transform import dna_transform_state
from experiments.run_fraud_fl_dna_transform_v2 import dna_transform_v2_state
from models.fraud_mlp import FraudMLP
from privacy.seed_manager import derive_seed

torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

OUT = ROOT / "artifacts/priority32_multidataset"
AMENDMENT = ROOT / "protocols/amendments/2026-10-02_priority32_multidataset_rq2_rq3.md"
DATASETS = ("paysim", "ieee_cis", "baf")
METHODS = ("baseline", "dna_v1_conservative", "dna_v2_0p95")
SEEDS = list(range(321000, 321021))
QUALITY_CONFIGS = [
    {"rounds": 50, "lr": .001, "alpha": .95},
    {"rounds": 50, "lr": .0003, "alpha": .95},
    {"rounds": 50, "lr": .003, "alpha": .95},
    {"rounds": 100, "lr": .001, "alpha": .95},
    {"rounds": 50, "lr": .001, "alpha": .90},
    {"rounds": 100, "lr": .0003, "alpha": .99},
]
RAW = {
    "paysim": ["datasets/creditcard.csv"],
    "ieee_cis": ["datasets/ieee-fraud-detection/train_transaction.csv", "datasets/ieee-fraud-detection/train_identity.csv"],
    "baf": ["datasets/baf/Base.csv"],
}


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write(path, value):
    path = Path(path)
    if not path.resolve().is_relative_to(OUT.resolve()):
        raise ValueError("writes restricted to new Priority32 artifacts")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def event(kind, **fields):
    OUT.mkdir(parents=True, exist_ok=True)
    record = {"at": now(), "event": kind, **fields}
    with (OUT / "runs.jsonl").open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    print(json.dumps(record), flush=True)


def progress(stage, dataset="all", method="all", done=0, total=1, failed=0, start=None):
    start = start or time.time()
    elapsed = time.time() - start
    doc = {"stage": stage, "dataset": dataset, "method": method, "done": done, "total": total,
           "failed": failed, "started_at": datetime.fromtimestamp(start, timezone.utc).isoformat(),
           "last_update": now(), "eta_minutes": (elapsed / done * (total-done) / 60) if done else None}
    write(OUT / "progress.json", doc)
    with (OUT / "progress.log").open("a") as handle:
        handle.write(json.dumps(doc) + "\n")


def checklist(stage, status):
    path = OUT / "checklist.json"
    items = json.loads(path.read_text()) if path.exists() else {
        str(i): {"title": title, "status": "pending"} for i, title in enumerate([
            "Data audit: all three datasets", "Train-only preprocessing and splits", "Validation baseline quality gates",
            "Execution freeze before confirmatory", "189 paired RQ2 jobs", "RQ2 analysis and independent recomputation",
            "RQ3 Bundle B and descriptive DP cost", "Report, hashes and final checks"])}
    items[str(stage)]["status"] = status
    items[str(stage)]["last_update"] = now()
    write(path, items)


def capped_parts(parts, labels, cap=500000, seed=320032):
    total = sum(map(len, parts))
    if total <= cap:
        return parts
    desired = np.array([len(p) * cap / total for p in parts])
    counts = np.floor(desired).astype(int)
    for i in np.argsort(-(desired-counts), kind="stable")[:cap-int(counts.sum())]:
        counts[i] += 1
    result = []
    for i, (indices, count) in enumerate(zip(parts, counts)):
        chosen, _ = train_test_split(indices, train_size=int(count), stratify=labels[indices], random_state=seed+i)
        # Preserve partition order, not numeric ID order (IEEE time / BAF month).
        selected = set(chosen.tolist())
        result.append(np.array([j for j in indices if j in selected], dtype=np.int64))
    assert sum(map(len, result)) == cap
    return result


def preprocess(frame, parts, categorical):
    """Fit all transforms on train; return ordered arrays and reproducible metadata."""
    train = frame.iloc[parts[0]]
    numeric = [c for c in frame if c not in categorical]
    medians = train[numeric].median().fillna(0.)
    scaler = RobustScaler()
    train_numeric = train[numeric].fillna(medians).to_numpy(dtype=np.float64)
    scaler.fit(train_numeric)
    names = list(numeric)
    levels, frequencies = {}, {}
    for column in categorical:
        values = train[column].fillna("__MISSING__").astype(str)
        categories = sorted(values.unique().tolist())
        if len(categories) <= 20:
            levels[column] = categories
            names.extend([f"{column}={value}" for value in categories])
        else:
            frequencies[column] = {str(k): float(v) for k, v in values.value_counts(normalize=True).items()}
            names.append(f"{column}:frequency")
    arrays = []
    for indices in parts:
        subset = frame.iloc[indices]
        blocks = [scaler.transform(subset[numeric].fillna(medians).to_numpy(dtype=np.float64)).astype(np.float32)]
        for column in categorical:
            values = subset[column].fillna("__MISSING__").astype(str)
            if column in levels:
                blocks.append(np.column_stack([(values == value).to_numpy(dtype=np.float32) for value in levels[column]]))
            else:
                blocks.append(values.map(frequencies[column]).fillna(0.).to_numpy(dtype=np.float32)[:, None])
        array = np.concatenate(blocks, axis=1)
        if not np.isfinite(array).all():
            raise ValueError("nonfinite features")
        arrays.append(array)
    meta = {"feature_names": names, "numeric": numeric, "categorical": categorical,
            "medians": medians.to_dict(), "robust_center": scaler.center_.tolist(),
            "robust_scale": scaler.scale_.tolist(), "one_hot_levels": levels, "frequencies": frequencies}
    return arrays, meta


def prepare_dataset(dataset):
    destination = OUT / "prepared" / dataset
    complete = destination / "audit.json"
    if complete.exists():
        doc = json.loads(complete.read_text())
        if any(sha(ROOT / p) != h for p, h in doc["raw_sha256"].items()):
            raise RuntimeError("raw input changed on resume")
        if any(sha(destination / p) != h for p, h in doc["prepared_sha256"].items()):
            raise RuntimeError("prepared data changed on resume")
        event("validated_preparation_skip", dataset=dataset)
        return doc
    raw_hashes = {p: sha(ROOT/p) for p in RAW[dataset]}
    dropped, sentinel = [], []
    if dataset == "paysim":
        df = pd.read_csv(ROOT/RAW[dataset][0], usecols=BASE_FEATURE_COLUMNS+["isFraud"])
        y = df.isFraud.to_numpy(dtype=np.int64)
        source = np.arange(len(df), dtype=np.int64)
        train, temporary = train_test_split(source, test_size=.35, stratify=y, random_state=320032)
        val, test = train_test_split(temporary, test_size=.20/.35, stratify=y[temporary], random_state=320032)
        parts = [train, val, test]
        # The unchanged loader's feature helper is safe only with no missing numeric data.
        if not df[[c for c in BASE_FEATURE_COLUMNS if c != "type"]].isna().any().any():
            features = _build_features(df)
        else:
            features = df[BASE_FEATURE_COLUMNS].copy()
            features["balance_diff_orig"] = features.oldbalanceOrg-features.newbalanceOrig
            features["balance_diff_dest"] = features.newbalanceDest-features.oldbalanceDest
        category = "type"
        categorical = [category]
        split_contract = "stratified65/15/20"
    elif dataset == "ieee_cis":
        transactions = pd.read_csv(ROOT/RAW[dataset][0], low_memory=False)
        identity = pd.read_csv(ROOT/RAW[dataset][1], low_memory=False)
        if transactions.TransactionID.duplicated().any() or identity.TransactionID.duplicated().any():
            raise ValueError("duplicate IEEE source identifiers")
        df = transactions.merge(identity, on="TransactionID", how="left", validate="one_to_one", sort=False)
        del transactions, identity
        y = df.isFraud.to_numpy(dtype=np.int64)
        source = df.TransactionID.to_numpy(dtype=np.int64)
        order = np.argsort(df.TransactionDT.to_numpy(), kind="stable")
        n = len(df)
        parts = [order[:int(.65*n)], order[int(.65*n):int(.8*n)], order[int(.8*n):]]
        features = df.drop(columns=["TransactionID", "isFraud"])
        category = "ProductCD"
        categorical = features.select_dtypes(include=["object", "string"]).columns.tolist()
        split_contract = "stable TransactionDT65/15/20; left join identity"
    else:
        df = pd.read_csv(ROOT/RAW[dataset][0], low_memory=False)
        y = df.fraud_bool.to_numpy(dtype=np.int64)
        source = np.arange(len(df), dtype=np.int64)
        if not df.month.isin(range(8)).all():
            raise ValueError("unexpected BAF month")
        order = np.argsort(df.month.to_numpy(), kind="stable")
        months = df.month.to_numpy()[order]
        parts = [order[months <= 4], order[months == 5], order[months >= 6]]
        features = df.drop(columns="fraud_bool").copy()
        categorical = ["payment_type", "employment_status", "housing_status", "source", "device_os"]
        for column in list(features.columns):
            if column not in categorical and (features[column] == -1).any():
                sentinel.append(column)
                features[column+":missing"] = (features[column] == -1).astype(np.float32)
                features[column] = features[column].replace(-1, np.nan)
        category = "payment_type"
        split_contract = "months0-4 train;5 validation;6-7 test"
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("binary label absent or invalid")
    original_counts = list(map(len, parts))
    parts = capped_parts(parts, y)
    if dataset == "ieee_cis":
        dropped = features.columns[(features.iloc[parts[0]].isna().mean() > .90)].tolist()
        features = features.drop(columns=dropped)
        categorical = [c for c in categorical if c not in dropped]
    arrays, preprocessing = preprocess(features, parts, categorical)
    if dataset == "ieee_cis" and arrays[0].shape[1] > 512:
        raise ValueError(f"IEEE dimension {arrays[0].shape[1]} exceeds512")
    source_sets = [set(source[p].tolist()) for p in parts]
    overlaps = {f"{i}:{j}": len(source_sets[i]&source_sets[j]) for i in range(3) for j in range(i+1, 3)}
    assert not any(overlaps.values())
    destination.mkdir(parents=True, exist_ok=True)
    prepared_files = []
    for name, indices, array in zip(("train", "validation", "test"), parts, arrays):
        for suffix, values in (("x", array), ("y", y[indices].astype(np.float32)), ("source_ids", source[indices])):
            path = destination / f"{name}_{suffix}.npy"
            np.save(path, values, allow_pickle=False)
            prepared_files.append(path)
    categories = features.iloc[parts[0]][category].fillna("__MISSING__").astype(str)
    np.save(destination/"train_categories.npy", categories.to_numpy(dtype=str), allow_pickle=False)
    prepared_files.append(destination/"train_categories.npy")
    write(destination/"preprocessing.json", preprocessing)
    prepared_files.append(destination/"preprocessing.json")
    doc = {"dataset": dataset, "raw_sha256": raw_hashes, "row_count": len(y), "fraud_rate": float(y.mean()),
           "feature_count": arrays[0].shape[1], "split_contract": split_contract, "original_partition_counts": original_counts,
           "selected_partitions": {name: {"rows": len(p), "fraud_rate": float(y[p].mean())} for name, p in zip(("train", "validation", "test"), parts)},
           "source_overlap": overlaps, "dropped_gt90pct_missing": dropped, "minus_one_sentinel_columns": sentinel,
           "natural_category": category, "primary_client_mapping": {c: i%3 for i, c in enumerate(sorted(categories.unique()))},
           "prepared_sha256": {str(p.relative_to(destination)): sha(p) for p in prepared_files}, "completed_at": now()}
    write(complete, doc)
    return doc


def job_config(dataset, method, seed, training, quality=False):
    return {"dataset": dataset, "method": method, "seed": seed, "training": training,
            "quality_validation_only": quality, "batch_size": 1024, "local_epochs": 1, "clients": 3,
            "focal_gamma": 2., "torch_threads": 1, "device": "cpu", "selected_checkpoint": "final_round",
            "audit_sha256": sha(OUT/"prepared"/dataset/"audit.json"), "amendment_sha256": sha(AMENDMENT)}


def probabilities(model, array):
    model.eval()
    blocks = []
    with torch.no_grad():
        for start in range(0, len(array), 4096):
            blocks.append(torch.sigmoid(model(torch.from_numpy(np.array(array[start:start+4096], copy=True)))).reshape(-1).numpy())
    return np.concatenate(blocks)


def metrics(y, probability, threshold):
    return {"f1": float(f1_score(y, probability >= threshold, zero_division=0)),
            "auc_roc": float(roc_auc_score(y, probability)), "pr_auc": float(average_precision_score(y, probability)),
            "rows": len(y), "fraud_rate": float(np.mean(y))}


def valid_result(path, config):
    if not path.exists():
        return False
    doc = json.loads(path.read_text())
    if doc.get("config_sha256") != canonical_sha(config) or doc.get("status") != "COMPLETED":
        raise ValueError(f"existing result does not match frozen configuration: {path}")
    for endpoint in ("f1", "auc_roc", "pr_auc"):
        assert np.isfinite(doc["validation"][endpoint])
        if not config["quality_validation_only"]:
            assert np.isfinite(doc["test"][endpoint])
    return True


def train_job(config, result_path):
    torch.set_num_threads(1)
    path = Path(result_path)
    if valid_result(path, config):
        return {"path": str(path), "skipped": True}
    started = time.time()
    seed, dataset, method = config["seed"], config["dataset"], config["method"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    prepared = OUT/"prepared"/dataset
    x = np.load(prepared/"train_x.npy", mmap_mode="r")
    y = np.load(prepared/"train_y.npy", mmap_mode="r")
    categories = np.load(prepared/"train_categories.npy")
    partitions = _mild_non_iid_client_indices(pd.DataFrame({"type": categories}), y, 3, seed)
    assert np.unique(np.concatenate(partitions)).size == len(y)
    loaders = [DataLoader(TensorDataset(torch.from_numpy(np.array(x[p], copy=True)),
                                       torch.from_numpy(np.array(y[p], copy=True)[:, None])),
                          batch_size=1024, shuffle=True, num_workers=0, drop_last=False,
                          generator=torch.Generator().manual_seed(seed+client))
               for client, p in enumerate(partitions)]
    # Fail closed rather than silently dropping singleton batches (BatchNorm).
    if any(len(p)%1024 == 1 for p in partitions):
        raise ValueError("singleton final batch: requires a technical amendment, not silent drop")
    model = FraudMLP(x.shape[1])
    loss_function = BinaryFocalLoss(alpha=config["training"]["alpha"], gamma=2.)
    round_log = path.with_suffix(".rounds.jsonl")
    round_log.parent.mkdir(parents=True, exist_ok=True)
    for round_number in range(1, config["training"]["rounds"]+1):
        global_state = copy.deepcopy(model.state_dict())
        states = []
        for client, loader in enumerate(loaders):
            local = copy.deepcopy(model).train()
            optimizer = torch.optim.Adam(local.parameters(), lr=config["training"]["lr"])
            for features, labels in loader:
                optimizer.zero_grad()
                loss = loss_function(local(features), labels)
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite training loss")
                loss.backward(); optimizer.step()
            state = local.state_dict()
            if method == "dna_v1_conservative":
                client_seed = derive_seed(derive_seed(seed, "rq2-dna-transform"), "dna_transform", round_number, client)
                state, _ = dna_transform_state(state, global_state, DNATransformConfig(block_size=256, mix_ratio=.08, keep_ratio=.88, shrink_factor=.45, seed=client_seed))
            elif method == "dna_v2_0p95":
                client_seed = derive_seed(derive_seed(seed, "rq2-dna-transform-v2"), "dna_transform_v2", round_number, client)
                state, _ = dna_transform_v2_state(state, global_state, DNATransformV2Config(compression_ratio=.95, quantization_eta=.01, seed=client_seed), derive_seed(client_seed, "quantization"))
            elif method != "baseline":
                raise ValueError("unknown method")
            states.append(OrderedDict((k, v.detach().clone()) for k, v in state.items()))
        model.load_state_dict(fed_avg(states, list(map(len, partitions))))
        with round_log.open("a") as handle:
            handle.write(json.dumps({"at": now(), "round": round_number, "elapsed_seconds": time.time()-started})+"\n")
    val_x = np.load(prepared/"validation_x.npy", mmap_mode="r")
    val_y = np.load(prepared/"validation_y.npy")
    val_probability = probabilities(model, val_x)
    threshold = tune_threshold(val_y, val_probability)
    doc = {"status": "COMPLETED", "config": config, "config_sha256": canonical_sha(config),
           "seed": seed, "dataset": dataset, "method": method, "threshold": threshold,
           "validation": metrics(val_y, val_probability, threshold), "client_rows": list(map(len, partitions)),
           "client_fraud_counts": [int(y[p].sum()) for p in partitions], "torch_threads": torch.get_num_threads(),
           "elapsed_seconds": time.time()-started, "completed_at": now()}
    if config["quality_validation_only"]:
        all_fraud_f1 = float(2*val_y.mean()/(1+val_y.mean()))
        doc["predict_all_fraud_f1"] = all_fraud_f1
        doc["gate_pass"] = doc["validation"]["auc_roc"] >= .7 and doc["validation"]["f1"] >= all_fraud_f1+.02
    else:
        test_x = np.load(prepared/"test_x.npy", mmap_mode="r")
        test_y = np.load(prepared/"test_y.npy")
        doc["test"] = metrics(test_y, probabilities(model, test_x), threshold)
    write(path, doc)
    return {"path": str(path), "skipped": False}


def run_pool(jobs, stage, workers=4):
    start, done = time.time(), 0
    progress(stage, total=len(jobs), start=start)
    with ProcessPoolExecutor(max_workers=workers, mp_context=__import__("multiprocessing").get_context("spawn")) as executor:
        futures = {}
        for cfg, path in jobs:
            if valid_result(path, cfg):
                done += 1
                event("validated_job_skip", stage=stage, path=str(path))
                progress(stage, cfg["dataset"], cfg["method"], done, len(jobs), start=start)
            else:
                event("job_launched", stage=stage, config_sha256=canonical_sha(cfg), path=str(path))
                futures[executor.submit(train_job, cfg, str(path))] = (cfg, path)
        for future in as_completed(futures):
            cfg, path = futures[future]
            try:
                future.result()
                assert valid_result(path, cfg)
            except Exception:
                event("job_failed", stage=stage, path=str(path), traceback=traceback.format_exc())
                progress(stage, cfg["dataset"], cfg["method"], done, len(jobs), failed=1, start=start)
                raise
            done += 1
            event("job_completed", stage=stage, path=str(path), sha256=sha(path))
            progress(stage, cfg["dataset"], cfg["method"], done, len(jobs), start=start)


def audit_seeds():
    hits = []
    for directory in (ROOT/"protocols/config", ROOT/"protocols/amendments"):
        for path in directory.rglob("*"):
            if path.suffix not in (".json", ".yaml", ".yml", ".md") or "priority32" in path.name:
                continue
            content = path.read_text(errors="replace")
            import re
            present = sorted(set(int(v) for v in re.findall(r"(?<!\d)3210(?:0\d|1\d|20)(?!\d)", content)))
            if present:
                hits.append({"path": str(path.relative_to(ROOT)), "seeds": present})
    write(OUT/"seed_audit.json", {"seeds": SEEDS, "previous_registered_seed_mentions": hits,
                                "scope": "registered protocol/config text; all three methods intentionally paired"})
    if hits:
        raise ValueError("registered training seed collision; ask before changing seeds")


def supervise():
    OUT.mkdir(parents=True, exist_ok=True)
    # Exclusive supervisor lock prevents accidental duplicate long workloads.
    lock = OUT / "supervisor.lock.json"
    if lock.exists():
        prior = json.loads(lock.read_text())
        try:
            os.kill(prior["pid"], 0)
        except ProcessLookupError:
            event("stale_supervisor_lock_resume", previous=prior)
            lock.rename(OUT/f"supervisor.lock.superseded.{time.time_ns()}.json")
        else:
            raise RuntimeError("another supervisor is still alive; refusing duplicate workload")
    with lock.open("x") as handle:
        json.dump({"pid": os.getpid(), "at": now()}, handle)
    start = time.time()
    event("supervisor_start_or_resume", pid=os.getpid(), command=sys.argv, started_at=now())
    write(OUT/"environment.json", {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
                                  "platform": platform.platform(), "workers": 4, "torch_threads": torch.get_num_threads()})
    checklist(0, "in_progress"); checklist(1, "in_progress")
    audits = {}
    for i, dataset in enumerate(DATASETS, 1):
        event("prepare_start", dataset=dataset)
        audits[dataset] = prepare_dataset(dataset)
        progress("0-1_data_preparation", dataset, "none", i, 3, start=start)
    write(OUT/"data_audit.json", audits)
    checklist(0, "completed"); checklist(1, "completed")
    audit_seeds()
    checklist(2, "in_progress")
    chosen = {}
    for attempt, training in enumerate(QUALITY_CONFIGS):
        pending = [d for d in DATASETS if d not in chosen]
        if not pending:
            break
        jobs = [(job_config(d, "baseline", 320100, training, True), OUT/"quality"/d/f"attempt_{attempt+1}.json") for d in pending]
        run_pool(jobs, f"2_quality_attempt_{attempt+1}")
        for cfg, path in jobs:
            result = json.loads(path.read_text())
            if result["gate_pass"]:
                chosen[cfg["dataset"]] = {"training": training, "quality_result": str(path.relative_to(ROOT)), "quality_sha256": sha(path)}
                event("dataset_quality_gate_pass", dataset=cfg["dataset"], attempt=attempt+1, validation=result["validation"])
    if len(chosen) != 3:
        write(OUT/"STOPPED_QUALITY_GATE.json", {"chosen": chosen, "failed_datasets": [d for d in DATASETS if d not in chosen], "at": now()})
        event("stopped_quality_gate", qualified_datasets=list(chosen))
        return
    checklist(2, "completed"); checklist(3, "in_progress")
    freeze = {"status": "FROZEN/AUTHORIZED", "chosen": chosen, "seeds": SEEDS, "methods": METHODS,
              "paired_n": 21, "jobs": 189, "amendment_sha256": sha(AMENDMENT),
              "audit_sha256": sha(OUT/"data_audit.json"), "code_sha256": sha(__file__)}
    freeze_path = OUT/"execution_freeze.json"
    if freeze_path.exists():
        previous = json.loads(freeze_path.read_text())
        assert {k: v for k, v in previous.items() if k != "frozen_at"} == freeze
    else:
        freeze["frozen_at"] = now()
        write(freeze_path, freeze)
    checklist(3, "completed"); checklist(4, "in_progress")
    jobs = [(job_config(d, m, s, chosen[d]["training"]), OUT/"rq2_jobs"/d/m/f"seed_{s}.json")
            for d in DATASETS for s in SEEDS for m in METHODS]
    # Fixed shuffle avoids scheduling all methods/large datasets in a single block.
    random.Random(320032).shuffle(jobs)
    run_pool(jobs, "4_RQ2", workers=4)
    checklist(4, "completed"); checklist(5, "in_progress")
    command([sys.executable, "-B", str(ROOT/"experiments/analyze_priority32_multidataset.py")])
    command([sys.executable, "-B", str(ROOT/"experiments/verify_priority32_statistics.py")])
    checklist(5, "completed"); checklist(6, "in_progress")
    command([sys.executable, "-B", str(ROOT/"experiments/priority32_cost.py"), "--orchestrate"])
    checklist(6, "completed"); checklist(7, "in_progress")
    command([sys.executable, "-B", str(ROOT/"experiments/finalize_priority32.py")])
    checklist(7, "completed")
    progress("COMPLETED", done=189, total=189, start=start)
    event("supervisor_completed", elapsed_seconds=time.time()-start)
    # Finalize once more after the LAST journal/progress writes so hashes are stable.
    command([sys.executable, "-B", str(ROOT/"experiments/finalize_priority32.py"), "--seal"], log=False)


def command(args, log=True):
    if log:
        event("command_launched", command=args)
    result = subprocess.run(args, cwd=ROOT, check=False)
    if log:
        event("command_finished", command=args, returncode=result.returncode)
    if result.returncode:
        raise RuntimeError(f"command failed: {args}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--supervise", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.prepare_only:
        for d in DATASETS:
            prepare_dataset(d)
    elif args.supervise:
        try:
            supervise()
        except Exception:
            event("supervisor_failed", traceback=traceback.format_exc())
            raise
    else:
        parser.error("select --supervise or --prepare-only")


if __name__ == "__main__":
    main()
