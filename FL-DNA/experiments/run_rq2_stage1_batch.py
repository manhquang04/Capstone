"""Run the frozen eight-seed RQ2 Stage-1 development/calibration batch."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SEEDS = [101, 202, 303, 404, 505, 606, 707, 808]
UTILITY_DP_MULTIPLIERS = [0.0001, 0.0005, 0.001, 0.005, 0.01]
CONTEXTUAL_DP_MULTIPLIERS = [0.00025]
METHODS = [
    "baseline",
    "dna_lossless",
    "dna_transform",
    *[f"dp_{value:g}" for value in UTILITY_DP_MULTIPLIERS],
    *[f"dp_{value:g}" for value in CONTEXTUAL_DP_MULTIPLIERS],
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def job_command(method: str) -> tuple[str, str]:
    if method == "baseline":
        return "experiments/run_fraud_fl_baseline.py", "BASELINE_OUTPUT_PATH"
    if method == "dna_lossless":
        return "experiments/run_fraud_fl_dna.py", "DNA_OUTPUT_PATH"
    if method == "dna_transform":
        return "experiments/run_fraud_fl_dna_transform.py", "DNA_TRANSFORM_OUTPUT_PATH"
    if method.startswith("dp_"):
        return "experiments/run_fraud_fl_dp.py", "DP_OUTPUT_PATH"
    raise ValueError(method)


def run_job(job: dict[str, object], python: str, output_root: Path) -> dict[str, object]:
    seed = int(job["seed"])
    method = str(job["method"])
    folder = output_root / f"seed_{seed}" / method
    folder.mkdir(parents=True, exist_ok=True)
    result_path = folder / "metrics.json"
    stdout_path = folder / "stdout.log"
    stderr_path = folder / "stderr.log"
    if result_path.exists() and (folder / "SUCCESS").exists():
        return {**job, "status": "SKIPPED_EXISTING_SUCCESS", "seconds": 0.0}

    prior_files = [folder / name for name in ("job_record.json", "stdout.log", "stderr.log", "metrics.json")]
    if any(path.exists() for path in prior_files):
        attempts = folder / "attempts"
        attempts.mkdir(exist_ok=True)
        attempt_id = 1
        while (attempts / f"attempt_{attempt_id}").exists():
            attempt_id += 1
        archive = attempts / f"attempt_{attempt_id}"
        archive.mkdir()
        for path in prior_files:
            if path.exists():
                shutil.copy2(path, archive / path.name)

    script, output_variable = job_command(method)
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(ROOT),
            "FL_RUN_SEED": str(seed),
            "MAX_ROWS": "500000",
            "NUM_ROUNDS": "50",
            "LOCAL_EPOCHS": "1",
            "FL_NUM_CLIENTS": "3",
            "LOSS_TYPE": "focal",
            "FOCAL_ALPHA": "0.95",
            "FOCAL_GAMMA": "2.0",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            output_variable: str(result_path),
        }
    )
    if method == "dna_transform":
        env.update(
            {
                "DNA_TRANSFORM_BLOCK_SIZE": "256",
                "DNA_TRANSFORM_MIX": "0.08",
                "DNA_TRANSFORM_KEEP": "0.88",
                "DNA_TRANSFORM_SHRINK": "0.45",
            }
        )
    if method.startswith("dp_"):
        env.update({"DP_CLIP_NORM": "100.0", "DP_NOISE_MULTIPLIER": method.removeprefix("dp_")})

    command = [python, script]
    started = time.perf_counter()
    with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
        process = subprocess.run(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr, check=False)
    elapsed = time.perf_counter() - started
    record = {
        **job,
        "status": "SUCCESS" if process.returncode == 0 and result_path.exists() else "FAILED",
        "returncode": process.returncode,
        "seconds": elapsed,
        "command": command,
        "environment": {key: env[key] for key in sorted(env) if key in {
            "FL_RUN_SEED", "MAX_ROWS", "NUM_ROUNDS", "LOCAL_EPOCHS", "FL_NUM_CLIENTS",
            "LOSS_TYPE", "FOCAL_ALPHA", "FOCAL_GAMMA", "DP_CLIP_NORM",
            "DP_NOISE_MULTIPLIER", "DNA_TRANSFORM_BLOCK_SIZE", "DNA_TRANSFORM_MIX",
            "DNA_TRANSFORM_KEEP", "DNA_TRANSFORM_SHRINK",
        }},
        "metrics": str(result_path),
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
    }
    (folder / "job_record.json").write_text(json.dumps(record, indent=2) + "\n")
    if record["status"] == "SUCCESS":
        (folder / "SUCCESS").write_text("success\n")
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--python", default=str(ROOT / ".venv-phase1/bin/python"))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/rq2/stage1_development_20260912")
    args = parser.parse_args()
    if not 1 <= args.workers <= 9:
        raise ValueError("workers must be between 1 and the frozen maximum 9")

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Balanced Latin rotation: at each scheduling wave, all methods and seeds
    # occur exactly once. This order was fixed before utility outcomes existed.
    jobs = []
    for wave in range(len(METHODS)):
        for seed_index, seed in enumerate(SEEDS):
            jobs.append({"seed": seed, "method": METHODS[(seed_index + wave) % len(METHODS)], "wave": wave})
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "RQ2 Stage-1 development only; seeds not reusable for confirmation",
        "seeds": SEEDS,
        "methods": METHODS,
        "utility_candidate_multipliers": UTILITY_DP_MULTIPLIERS,
        "contextual_distortion_matched_multipliers": CONTEXTUAL_DP_MULTIPLIERS,
        "workers": args.workers,
        "ordering": "balanced_latin_rotation",
        "jobs": jobs,
        "code_hashes": {
            path: sha256(ROOT / path)
            for path in [
                "data/load_creditcard.py",
                "experiments/fraud_fl_common.py",
                "experiments/run_fraud_fl_baseline.py",
                "experiments/run_fraud_fl_dna.py",
                "experiments/run_fraud_fl_dna_transform.py",
                "experiments/run_fraud_fl_dp.py",
                "privacy/dp_engine.py",
            ]
        },
    }
    (output / "batch_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_job, job, args.python, output) for job in jobs]
        for future in as_completed(futures):
            record = future.result()
            results.append(record)
            print(json.dumps({key: record.get(key) for key in ("seed", "method", "status", "seconds")}))
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "jobs": len(results),
        "success": sum(row["status"] in ("SUCCESS", "SKIPPED_EXISTING_SUCCESS") for row in results),
        "failed": sum(row["status"] == "FAILED" for row in results),
        "records": sorted(results, key=lambda row: (int(row["seed"]), str(row["method"]))),
    }
    (output / "batch_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: summary[key] for key in ("jobs", "success", "failed")}, indent=2))
    raise SystemExit(0 if summary["failed"] == 0 else 1)


if __name__ == "__main__":
    main()
