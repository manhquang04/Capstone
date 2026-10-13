"""Scoped evidence I/O and qualification-only state tracking."""
from pathlib import Path
import hashlib
import json
import os
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/priority35a"
Q = OUT / "qualification"
PROTOCOL = ROOT / "protocols/amendments/2026-10-10_priority35a_config_c_protocol.md"
PROTOCOL_SHA = "0a9cd3c1e9b9ccdce2fbacfdc2f4d326b535b109e63af8163f0cdbf2b575a0f7"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def scoped(path):
    path = Path(path).resolve()
    if OUT.resolve() not in path.parents:
        raise ValueError("Priority35A artifact write outside allowlist")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write(path, doc, private=False, exclusive=False):
    path = scoped(path)
    if exclusive and path.exists():
        raise ValueError("immutable receipt already exists: " + str(path))
    temp = path.with_name(path.name + ".tmp." + str(os.getpid()))
    descriptor = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600 if private else 0o644)
    with os.fdopen(descriptor, "w") as handle:
        json.dump(doc, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    temp.replace(path)


def load(path):
    return json.loads(Path(path).read_text())


def now():
    return datetime.now(timezone.utc).isoformat()


def stage(name, **fields):
    doc = load(OUT / "progress.json")
    doc.update(stage=name, status="QUALIFICATION_IN_PROGRESS", complete=False, last_update=now())
    doc.update(fields)
    write(OUT / "progress.json", doc)
    with scoped(OUT / "progress.log").open("a") as handle:
        handle.write(json.dumps(dict(at=now(), stage=name, **fields), allow_nan=False) + "\n")
    check = load(OUT / "checklist.json")
    check.update(stage=name, complete=False)
    write(OUT / "checklist.json", check)
    print(json.dumps(dict(at=now(), stage=name, **fields), allow_nan=False), flush=True)


def tensor_save(path, value, private=False):
    import torch
    path = scoped(path)
    if path.exists():
        raise ValueError("never overwrite a tensor artifact")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600 if private else 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        torch.save(value, stream)


def check_hashes(hashes):
    for relative, expected in hashes.items():
        if sha(ROOT / relative) != expected:
            raise ValueError("hash drift: " + relative)


def freeze_check():
    if sha(PROTOCOL) != PROTOCOL_SHA:
        raise ValueError("protocol hash drift")
    doc = load(Q / "execution_freeze.json")
    check_hashes(doc["hashes"])
    return doc
