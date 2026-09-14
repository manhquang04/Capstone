"""Run raw-only RQ1 scope-boundary coarse screening.

This runner intentionally executes only the raw branch.  It reuses the frozen
Phase 4 raw attacker implementation and baseline gate files, while allowing the
target set to vary by records_per_group according to a pre-approved screening
config.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_RUN = ROOT / "artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _command(run: Path, group: int, python: str, config: dict, target: Path) -> list[str]:
    attack = config["attack"]
    return [
        python,
        "experiments/run_phase4_harddiff_reparam_for_misselected.py",
        str(run),
        "--target-file",
        target.name,
        "--groups",
        str(group),
        "--restarts",
        str(attack["restarts"]),
        "--init-mode",
        attack["init_mode"],
        "--nonnegative-lambda",
        str(attack["nonnegative_lambda"]),
    ]


def _job(job: dict, python: str, output: Path, config: dict, target: Path) -> dict:
    group = int(job["group"])
    run = output / "raw" / f"group_{group}_run"
    run.mkdir(parents=True, exist_ok=False)
    shutil.copy2(SOURCE_RUN / "baseline_gate.json", run / "baseline_gate.json")
    shutil.copy2(SOURCE_RUN / "frozen.json", run / "frozen.json")
    shutil.copy2(target, run / target.name)

    command = _command(run, group, python, config, target)
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(ROOT),
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    started = time.perf_counter()
    with (run / "stdout.log").open("w") as stdout, (run / "stderr.log").open("w") as stderr:
        process = subprocess.run(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr, check=False)
    row = {
        **job,
        "branch": "raw",
        "command": command,
        "returncode": process.returncode,
        "seconds": time.perf_counter() - started,
        "status": "SUCCESS" if process.returncode == 0 else "FAILED",
    }
    (run / "job_record.json").write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=9)
    parser.add_argument("--python", default=str(ROOT / ".venv-phase1/bin/python"))
    args = parser.parse_args()

    if not 1 <= args.workers <= 9:
        raise ValueError("workers must be in [1, 9]")

    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("scope_screening_authorized") is not True:
        raise RuntimeError("scope screening is not authorized")
    if config.get("branch") != "raw":
        raise RuntimeError("this runner only supports branch=raw")

    target = (ROOT / config["target"]["path"]).resolve()
    overlap = (ROOT / config["target"]["overlap_matrix"]).resolve()
    if _sha256(target) != config["target"]["sha256"]:
        raise RuntimeError("target hash mismatch")
    disjoint = json.loads(overlap.read_text(encoding="utf-8"))
    if disjoint["disjointness_gate"] != "PASS" or disjoint["max_overlap"] != 0:
        raise RuntimeError("target disjointness gate failed")
    for path, expected in config["implementation_hashes"].items():
        if _sha256(ROOT / path) != expected:
            raise RuntimeError(f"implementation hash mismatch: {path}")

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    jobs = [{"group": group} for group in range(int(config["target"]["groups"]))]
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": str(args.config),
        "config_sha256": _sha256(args.config),
        "target_sha256": _sha256(target),
        "records_per_group": config["screening"]["records_per_group"],
        "fraud_records_per_group": config["screening"]["fraud_records_per_group"],
        "jobs": jobs,
        "workers": args.workers,
    }
    (output / "execution_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(_job, job, args.python, output, config, target) for job in jobs]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(json.dumps({key: row[key] for key in ("branch", "group", "status", "seconds")}), flush=True)

    summary = {
        "jobs": len(rows),
        "success": sum(row["status"] == "SUCCESS" for row in rows),
        "failed": sum(row["status"] == "FAILED" for row in rows),
        "records": sorted(rows, key=lambda row: row["group"]),
    }
    (output / "execution_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("jobs", "success", "failed")}, indent=2))
    raise SystemExit(0 if summary["failed"] == 0 else 1)


if __name__ == "__main__":
    main()
