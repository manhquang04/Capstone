"""Historical target-ID audit, isolated from official TabLeak import namespaces."""
from __future__ import annotations
import hashlib
import argparse
import json
import numbers
import os
import sys
from pathlib import Path
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
import torch
import numpy as np
torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    if torch.get_num_interop_threads() != 1:
        raise

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "artifacts/priority34c"


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def flatten_ids(value):
    if isinstance(value, str):
        return flatten_ids(json.loads(value))
    if isinstance(value, dict):
        return [i for child in value.values() for i in flatten_ids(child)]
    if torch.is_tensor(value):
        return flatten_ids(value.tolist())
    if isinstance(value, np.ndarray):
        return flatten_ids(value.tolist())
    if isinstance(value, (list, tuple)):
        return [i for child in value for i in flatten_ids(child)]
    if isinstance(value, numbers.Integral) and not isinstance(value, bool):
        return [int(value)]
    raise ValueError("invalid historical source-ID leaf")


def extract(value):
    ids = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "sources" and isinstance(child, (list, tuple)) and all(isinstance(i, numbers.Integral) for i in child):
                ids.extend(flatten_ids(child))
            elif key in ("source_ids", "source_rows", "new_source_rows", "transaction_ids", "source_ids_json") and not isinstance(child, dict):
                ids.extend(flatten_ids(child))
            elif key in ("source_ids", "source_rows", "new_source_rows", "transaction_ids", "source_ids_json"):
                ids.extend(flatten_ids(child))
            elif isinstance(child, (dict, list, tuple)):
                ids.extend(extract(child))
    elif isinstance(value, (list, tuple)):
        for child in value:
            if isinstance(child, (dict, list, tuple)):
                ids.extend(extract(child))
    return sorted(set(ids))


def candidates(dataset):
    artifacts = ROOT / "artifacts"
    paths = set()
    aliases = {"paysim": ("paysim",), "ieee_cis": ("ieee",), "baf": ("baf",)}[dataset]
    for path in artifacts.rglob("*"):
        if not path.is_file() or OUT in path.parents:
            continue
        relative = str(path.relative_to(artifacts)).lower()
        generic_paysim = (dataset == "paysim" and path.suffix == ".pt" and "target" in path.name.lower()
                          and not any(other in relative for other in ("ieee", "baf", "cifar", "mnist", "adult")))
        if not any(alias in relative for alias in aliases) and not generic_paysim:
            continue
        if path.suffix == ".pt" and ("target" in path.name.lower() or "bundle" in path.name.lower()):
            paths.add(path)
        if path.suffix == ".json" and ("target" in path.name.lower() or "source_id" in path.name.lower()):
            paths.add(path)
    return sorted(paths)


def audit(dataset):
    rows, excluded, cache = [], set(), {}
    references, additional_inputs = {}, {}
    for path in candidates(dataset):
        digest = sha(path)
        if digest not in cache:
            if path.suffix == ".json":
                value = json.loads(path.read_text())
            else:
                # Trusted project-owned historical bundles only; no external
                # pickle or downloaded file is loaded. Never log bundle values.
                value = torch.load(path, map_location="cpu", weights_only=False)
            try:
                cache[digest] = extract(value)
            except Exception as error:
                raise ValueError("historical ID parse failed: " + str(path.relative_to(ROOT))) from error
        ids = cache[digest]
        reconstruction = None
        if not ids and dataset == "paysim" and "phase3/full_" in str(path.relative_to(ROOT)) and path.name == "evaluator_targets.pt":
            protocol = path.parent.parent / "protocol_lock.json"
            config = json.loads(protocol.read_text())
            key = (int(config["data_seed"]), int(config["max_rows"]))
            if key not in references:
                import pandas as pd
                from sklearn.model_selection import train_test_split
                raw = ROOT / "datasets/creditcard.csv"
                labels = pd.read_csv(raw, usecols=["isFraud"])["isFraud"].to_numpy()
                source = np.arange(len(labels), dtype=np.int64)
                if key[1] >= len(labels):
                    sampled = source
                else:
                    _, sampled = train_test_split(source, test_size=key[1], random_state=key[0], stratify=labels)
                references[key] = sorted(int(i) for i in sampled)
                additional_inputs[str(raw.relative_to(ROOT))] = sha(raw)
            ids = references[key]
            additional_inputs[str(protocol.relative_to(ROOT))] = sha(protocol)
            reconstruction = dict(method="conservative entire sampled reference population",
                                  data_seed=key[0], max_rows=key[1], protocol=str(protocol.relative_to(ROOT)),
                                  source_ids_sha256=hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest())
        if not ids:
            raise ValueError("historical target candidate has no audited IDs: " + str(path.relative_to(ROOT)))
        excluded.update(ids)
        receipt = dict(path=str(path.relative_to(ROOT)), sha256=digest, count=len(ids))
        if reconstruction is None:
            receipt["source_ids"] = ids
        else:
            receipt["conservative_reconstruction"] = reconstruction
        rows.append(receipt)
    if not rows:
        raise ValueError("no historical provenance found for " + dataset)
    return dict(dataset=dataset, exclusions=sorted(excluded), provenance=rows,
                candidate_count=len(rows), unique_bundle_count=len(cache),
                additional_inputs=additional_inputs,
                method="explicit target/bundle names; source_ids/transaction_ids recursively extracted")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=("paysim", "ieee_cis", "baf"), default=["paysim", "ieee_cis", "baf"])
    args = parser.parse_args()
    for dataset in args.datasets:
        doc = audit(dataset)
        path = OUT / "history" / (dataset + ".json")
        if path.exists():
            if json.loads(path.read_text()) != doc:
                raise ValueError("historical provenance drift")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(doc, indent=2, sort_keys=True, allow_nan=False) + "\n")
        print(json.dumps(dict(dataset=dataset, historical_ids=len(doc["exclusions"]),
                              candidate_count=doc["candidate_count"])))
