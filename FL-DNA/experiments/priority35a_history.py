"""Read-only, hash-verified P24–34 target inventory; no measurement.

Original source IDs are dataset-namespaced. Earlier verified P34C exclusions
are retained conservatively; every newly discovered target must be accounted
for, not silently skipped because its provenance is ambiguous.
"""
from __future__ import annotations
import hashlib
import json
import os
import re
from pathlib import Path
import numpy as np
import torch
from experiments.priority35a_firewall import integer_ids, digest

ROOT = Path(__file__).resolve().parents[1]
FIELDS = {"source_ids", "source_rows", "new_source_rows", "transaction_ids", "source_ids_json"}


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            result.update(block)
    return result.hexdigest()


def flatten_ids(value):
    if isinstance(value, str):
        return flatten_ids(json.loads(value))
    if isinstance(value, dict):
        return [i for child in value.values() for i in flatten_ids(child)]
    if torch.is_tensor(value) or isinstance(value, np.ndarray):
        return flatten_ids(value.tolist())
    if isinstance(value, (list, tuple)):
        return [i for child in value for i in flatten_ids(child)]
    return integer_ids([value])


def extract_ids(value):
    result = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key in FIELDS or key == "sources" and isinstance(child, (list, tuple)):
                result.extend(flatten_ids(child))
            elif isinstance(child, (dict, list, tuple)):
                result.extend(extract_ids(child))
    elif isinstance(value, (list, tuple)):
        for child in value:
            if isinstance(child, (dict, list, tuple)):
                result.extend(extract_ids(child))
    return sorted(set(result))


def discover(root=ROOT):
    paths = []
    for top in sorted((root / "artifacts").iterdir()):
        match = re.match(r"priority(\d+)", top.name)
        if not top.is_dir() or not match or not 24 <= int(match.group(1)) <= 34:
            continue
        for folder, directories, files in os.walk(top):
            directories[:] = sorted(d for d in directories if d != "__pycache__")
            for name in sorted(files):
                if name.endswith((".json", ".pt")) and any(s in name.lower() for s in ("target", "source_id", "firewall", "reservation")):
                    paths.append(Path(folder) / name)
    return sorted(paths)


def classify(relative, document):
    low = relative.lower()
    # These are separately inspected non-record audit schemas, not broad
    # permission to disregard any unparseable target candidate.
    if low.startswith("artifacts/priority34a/partial_audits/round_firewall_"):
        if isinstance(document, dict) and "upload_assertions" in document and "jobs" in document:
            return "NON_TARGET_UPLOAD_FIREWALL"
    if low == "artifacts/priority34c/firewall_preflight.json":
        if isinstance(document, dict) and document.get("target_records_accessed") is False:
            return "NON_TARGET_PREFLIGHT"
    if low.startswith("artifacts/priority30_native_defenses/") and low.endswith("target_match_check.json"):
        if isinstance(document, dict) and "image_source_ids" in document and "adult_source_ids" in document:
            return "NON_FRAUD_IMAGE_AND_ADULT"
    if (low == "artifacts/priority33c/image_targets.json"
        or low.startswith("artifacts/priority34b/recovery/")
        or low == "artifacts/priority34b/image_targets.json"
        or low.startswith("artifacts/priority34d/image_recovery/")
        or low == "artifacts/priority34d/audits/independent_reservation_preflight.json"
        or "cifar" in low or "mnist" in low or "adult" in low):
        return "NON_FRAUD_IMAGE_OR_ADULT"
    explicit = document.get("dataset") if isinstance(document, dict) else None
    if explicit in ("baf", "ieee_cis", "paysim"):
        return explicit
    choices = [d for d, aliases in (("baf", ("baf",)), ("ieee_cis", ("ieee",)), ("paysim", ("paysim",)))
               if any(a in low for a in aliases)]
    if len(choices) != 1:
        raise ValueError("ambiguous historical dataset namespace: " + relative)
    return choices[0]


def audit(root=ROOT):
    root = Path(root).resolve()
    excluded = {d: set() for d in ("baf", "ieee_cis", "paysim")}
    provenance = []
    inherited_ids_by_path = {}
    verified_hashes = {}

    def checked_hash(path, expected=None):
        relative = str(path.relative_to(root))
        if relative not in verified_hashes:
            verified_hashes[relative] = sha(path)
        value = verified_hashes[relative]
        if expected is not None and value != expected:
            raise ValueError("historical evidence hash drift: " + relative)
        return value

    def safe_path(relative):
        path = (root / relative).resolve()
        if not path.is_relative_to(root / "artifacts") and not path.is_relative_to(root / "datasets"):
            raise ValueError("historical input outside project evidence namespaces")
        return path

    for dataset in excluded:
        path = root / "artifacts/priority34c/history" / (dataset + ".json")
        doc = json.loads(path.read_text())
        if doc["dataset"] != dataset or len(doc["provenance"]) != doc["candidate_count"]:
            raise ValueError("inherited historical audit schema mismatch")
        inherited = integer_ids(doc["exclusions"])
        if len(inherited) != len(set(inherited)):
            raise ValueError("duplicate inherited exclusion list")
        excluded[dataset].update(inherited)
        checked_hash(path)
        for row in doc["provenance"]:
            checked_hash(safe_path(row["path"]), row["sha256"])
            if "source_ids" in row:
                ids = integer_ids(row["source_ids"])
                if len(ids) != row["count"] or not set(ids) <= excluded[dataset]:
                    raise ValueError("historical audit exclusion omission")
            elif "conservative_reconstruction" in row:
                ids = inherited  # Conservatively retain the entire validated union.
            else:
                raise ValueError("missing inherited source-ID proof")
            inherited_ids_by_path[row["path"]] = (dataset, ids)
        for relative, expected in doc["additional_inputs"].items():
            checked_hash(safe_path(relative), expected)
        provenance.append(dict(path=str(path.relative_to(root)), sha256=checked_hash(path),
                               dataset=dataset, prior_exclusions=len(inherited),
                               verified_provenance_count=len(doc["provenance"])))

    candidates = []
    parsed_by_hash = {}
    for path in discover(root):
        relative = str(path.relative_to(root))
        hashed = checked_hash(path)
        if relative in inherited_ids_by_path:
            dataset, ids = inherited_ids_by_path[relative]
            method = "hash-validated inherited source IDs"
        else:
            if hashed not in parsed_by_hash:
                parsed_by_hash[hashed] = (json.loads(path.read_text()) if path.suffix == ".json"
                    else torch.load(path, map_location="cpu", weights_only=False))
            doc = parsed_by_hash[hashed]
            dataset = classify(relative, doc)
            ids = extract_ids(doc) if dataset in excluded else []
            method = "recursive original-source-ID manifest parse" if ids else "explicit non-fraud/non-target namespace"
            if dataset in excluded and not ids:
                raise ValueError("fraud target manifest has no audited source IDs: " + relative)
        if dataset in excluded:
            excluded[dataset].update(ids)
        candidates.append(dict(path=relative, sha256=hashed, dataset=dataset,
                               source_count=len(ids), source_ids_sha256=digest(sorted(set(ids))),
                               method=method))
    if not candidates:
        raise ValueError("no P24–34 candidate manifests found")
    return dict(scope="read-only target-ID inventory, not a leakage measurement",
                inventory_count=len(candidates), inventory=candidates,
                inherited_history=provenance, verified_input_hashes=verified_hashes,
                exclusions={d: sorted(ids) for d, ids in excluded.items()},
                exclusion_counts={d: len(ids) for d, ids in excluded.items()},
                new_real_target_allocations=0)
