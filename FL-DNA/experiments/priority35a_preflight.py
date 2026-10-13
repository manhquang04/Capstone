"""Protocol/test stage ONLY: no training, target generation or measurements."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/priority35a"
PROTOCOL = ROOT / "protocols/amendments/2026-10-10_priority35a_config_c_protocol.md"
sys.path.insert(0, str(ROOT))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def now():
    return datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).isoformat()


def save(path, value):
    path = Path(path).resolve()
    if not path.is_relative_to(OUT.resolve()):
        raise ValueError("P35A output firewall")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def stage(name, **fields):
    progress_path = OUT / "progress.json"
    doc = json.loads(progress_path.read_text())
    doc.update(stage=name, last_update=now(), scientific_runs_launched=0,
               complete=False, **fields)
    save(progress_path, doc)
    with (OUT / "progress.log").open("a") as stream:
        stream.write(json.dumps(dict(at=now(), stage=name, **fields), allow_nan=False) + "\n")
    print(json.dumps(dict(stage=name, **fields)), flush=True)


def execute():
    if Path(sys.executable).resolve().parent != (ROOT / ".venv-phase1/bin").resolve():
        # Virtualenv interpreters may be symlinks, so compare sys.prefix too.
        if Path(sys.prefix).resolve() != (ROOT / ".venv-phase1").resolve():
            raise ValueError("must use .venv-phase1")
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE="1")
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = "1"
        os.environ[key] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    protocol_hash = sha(PROTOCOL)
    lock = OUT / "protocol_lock.json"
    lock_doc = dict(protocol=str(PROTOCOL.relative_to(ROOT)), sha256=protocol_hash,
                    scope="protocol/tests only; no measurement authorization")
    if lock.exists():
        if json.loads(lock.read_text()) != lock_doc:
            raise ValueError("locked protocol drift")
    else:
        save(lock, lock_doc)
    sources = sorted(list((ROOT / "experiments").glob("priority35a_*.py"))
                     + list((ROOT / "tests").glob("test_priority35a*.py")))
    attempt = OUT / "preflight_attempts" / str(time.time_ns())
    attempt.mkdir(parents=True, exist_ok=False)
    save(attempt / "inputs.json", dict(protocol_sha256=protocol_hash,
         sources={str(p.relative_to(ROOT)): sha(p) for p in sources},
         started_at=now(), measurement_runs=0))
    stage("protocol_locked_before_tests", status="PREFLIGHT_RUNNING", protocol_sha256=protocol_hash,
          preflight_attempt=str(attempt.relative_to(ROOT)), required_direction=None)
    commands = [
        [sys.executable, "-B", "-m", "unittest", "discover", "-s", "tests", "-p", "test_priority35a*.py", "-v"],
        [sys.executable, "-m", "py_compile", *[str(p.relative_to(ROOT)) for p in sources]],
        ["git", "diff", "--check"],
    ]
    results = []
    for index, command in enumerate(commands):
        stage(("unit_tests", "py_compile", "git_diff_check")[index], status="PREFLIGHT_RUNNING")
        command_env = dict(env)
        if index == 1:
            command_env["PYTHONPYCACHEPREFIX"] = str(OUT / "bytecode")
        result = subprocess.run(command, cwd=ROOT, env=command_env, capture_output=True, text=True)
        row = dict(command=command, returncode=result.returncode,
                   stdout=result.stdout, stderr=result.stderr, at=now())
        save(attempt / (f"command_{index}.json"), row)
        results.append(row)
        print(json.dumps(dict(command_index=index, returncode=result.returncode,
                             output=(result.stdout + result.stderr)[-14000:])), flush=True)
        if result.returncode:
            save(attempt / "FAILURE.json", dict(stage=index, preserved_command_log=f"command_{index}.json"))
            stage("preflight_failed", status="PREFLIGHT_FAILED", failed_command=index)
            return 1
    # Independently preserve the complete read-only history audit for review.
    stage("historical_manifest_hash_audit", status="PREFLIGHT_RUNNING")
    from experiments.priority35a_history import audit
    history = audit()
    save(attempt / "history_audit.json", history)
    # git diff ignores new untracked files; explicitly check those as well.
    local_docs = sources + sorted((ROOT / "protocols/amendments").glob("2026-10-10_priority35a_*.md"))
    for path in local_docs:
        content = path.read_text()
        if not content.endswith("\n") or any(line.rstrip(" \t") != line for line in content.splitlines()):
            raise ValueError("new-file whitespace check failed: " + str(path.relative_to(ROOT)))
    match = re.search(r"Ran (\d+) tests?", results[0]["stderr"])
    if not match:
        raise ValueError("missing unittest count")
    test_count = int(match.group(1))
    checklist = json.loads((OUT / "checklist.json").read_text())
    checklist.update(protocol_frozen=True, singleton_reference_resolved=True,
                     simulation_label_firewall_metric_tests_pass=True,
                     historical_manifest_audit_pass=True, test_count=test_count,
                     protocol_sha256=protocol_hash,
                     stage="PROTOCOL_TESTS_READY_WAITING_USER", complete=False)
    save(OUT / "checklist.json", checklist)
    receipt = dict(status="PROTOCOL_TESTS_READY_WAITING_USER", at=now(), test_count=test_count,
                   protocol_sha256=protocol_hash, scientific_runs_launched=0,
                   new_real_target_allocations=0, history_counts=history["exclusion_counts"],
                   historical_candidates=history["inventory_count"], py_compile="PASS",
                   git_diff_check="PASS", new_file_whitespace_check="PASS",
                   commands=[r["command"] for r in results], full_priority35a_complete=False)
    save(attempt / "READY.json", receipt)
    save(OUT / "protocol_tests_ready.json", dict(**receipt, attempt=str(attempt.relative_to(ROOT))))
    stage("protocol_tests_ready_waiting_user", status="WAITING_USER_NO_MEASUREMENTS", tests_run=test_count,
          protocol_sha256=protocol_hash, pending=["User authorization for measurements",
              "Freeze execution sources/inputs", "Reconstruct checkpoints", "Qualification gates",
              "Conditional confirmation", "Independent statistics/report/manifest"])
    manifest = {str(p.relative_to(ROOT)): sha(p) for p in local_docs}
    manifest.update({str(p.relative_to(ROOT)): sha(p) for p in attempt.iterdir() if p.is_file()})
    for name in ("protocol_lock.json", "protocol_tests_ready.json", "progress.json", "progress.log", "checklist.json"):
        manifest[str((OUT / name).relative_to(ROOT))] = sha(OUT / name)
    save(OUT / ("protocol_tests_manifest_" + attempt.name + ".json"), dict(
        scope="protocol/test-stage seal, not a final scientific manifest", entries=manifest,
        historical_input_hashes=history["verified_input_hashes"]))
    print(json.dumps(receipt), flush=True)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tests", action="store_true", required=True)
    parser.parse_args()
    try:
        code = execute()
    except Exception as error:
        stage("preflight_exception", status="PREFLIGHT_FAILED", error_type=type(error).__name__, error=str(error))
        raise
    raise SystemExit(code)
