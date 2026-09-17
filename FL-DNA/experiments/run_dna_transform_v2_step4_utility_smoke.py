"""Run DNA Transform v2 Step 4 development-scale utility smoke.

This is not confirmatory.  It checks whether v2 preserves enough F1/AUC to
justify any later privacy calibration.  No security attack is run here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SEEDS = [101, 202, 303]
MAX_ROWS = 100_000
NUM_ROUNDS = 10
NUM_CLIENTS = 3
METHODS = [
    {"method": "baseline", "script": "experiments/run_fraud_fl_baseline.py", "output_var": "BASELINE_OUTPUT_PATH"},
    {
        "method": "v2_ratio0p95_eta0p01",
        "script": "experiments/run_fraud_fl_dna_transform_v2.py",
        "output_var": "DNA_TRANSFORM_V2_OUTPUT_PATH",
        "compression_ratio": 0.95,
        "quantization_eta": 0.01,
    },
    {
        "method": "v2_ratio0p9_eta0p01",
        "script": "experiments/run_fraud_fl_dna_transform_v2.py",
        "output_var": "DNA_TRANSFORM_V2_OUTPUT_PATH",
        "compression_ratio": 0.90,
        "quantization_eta": 0.01,
    },
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_job(job: dict[str, object], python: str, output_root: Path) -> dict[str, object]:
    seed = int(job["seed"])
    method = str(job["method"])
    folder = output_root / f"seed_{seed}" / method
    folder.mkdir(parents=True, exist_ok=False)
    result_path = folder / "metrics.json"

    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(ROOT),
            "FL_RUN_SEED": str(seed),
            "MAX_ROWS": str(MAX_ROWS),
            "NUM_ROUNDS": str(NUM_ROUNDS),
            "LOCAL_EPOCHS": "1",
            "FL_NUM_CLIENTS": str(NUM_CLIENTS),
            "LOSS_TYPE": "focal",
            "FOCAL_ALPHA": "0.95",
            "FOCAL_GAMMA": "2.0",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            str(job["output_var"]): str(result_path),
        }
    )
    if method.startswith("v2_"):
        env.update(
            {
                "DNA_TRANSFORM_V2_COMPRESSION_RATIO": str(job["compression_ratio"]),
                "DNA_TRANSFORM_V2_QUANTIZATION_ETA": str(job["quantization_eta"]),
            }
        )

    command = [python, str(job["script"])]
    started = time.perf_counter()
    with (folder / "stdout.log").open("w") as stdout, (folder / "stderr.log").open("w") as stderr:
        process = subprocess.run(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr, check=False)
    elapsed = time.perf_counter() - started

    record = {
        "seed": seed,
        "method": method,
        "status": "SUCCESS" if process.returncode == 0 and result_path.exists() else "FAILED",
        "returncode": process.returncode,
        "seconds": elapsed,
        "command": command,
        "metrics": str(result_path),
        "stdout": str(folder / "stdout.log"),
        "stderr": str(folder / "stderr.log"),
        "environment": {
            key: env[key]
            for key in sorted(env)
            if key
            in {
                "FL_RUN_SEED",
                "MAX_ROWS",
                "NUM_ROUNDS",
                "LOCAL_EPOCHS",
                "FL_NUM_CLIENTS",
                "LOSS_TYPE",
                "FOCAL_ALPHA",
                "FOCAL_GAMMA",
                "DNA_TRANSFORM_V2_COMPRESSION_RATIO",
                "DNA_TRANSFORM_V2_QUANTIZATION_ETA",
            }
        },
    }
    (folder / "job_record.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--python", default=str(ROOT / ".venv-phase1/bin/python"))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/dna_transform_v2/step4_utility_smoke_20260916")
    args = parser.parse_args()
    if not 1 <= args.workers <= 9:
        raise ValueError("workers must be between 1 and 9")

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    jobs: list[dict[str, object]] = []
    for seed in SEEDS:
        for method in METHODS:
            jobs.append({"seed": seed, **method})

    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "DNA Transform v2 Step 4 development-scale utility smoke; not confirmatory; no attack",
        "seeds": SEEDS,
        "max_rows": MAX_ROWS,
        "num_rounds": NUM_ROUNDS,
        "num_clients": NUM_CLIENTS,
        "methods": METHODS,
        "utility_collapse_rule": "if v2 F1/AUC is materially lower than paired baseline, stop before Step 5 security calibration",
        "jobs": jobs,
        "code_hashes": {
            path: sha256(ROOT / path)
            for path in [
                "data/load_creditcard.py",
                "experiments/fraud_fl_common.py",
                "experiments/run_fraud_fl_baseline.py",
                "experiments/run_fraud_fl_dna_transform_v2.py",
                "dna_encoder/transform_defense_v2.py",
            ]
        },
    }
    (output / "smoke_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    records = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_job, job, args.python, output) for job in jobs]
        for future in as_completed(futures):
            row = future.result()
            records.append(row)
            print(json.dumps({key: row.get(key) for key in ("seed", "method", "status", "seconds")}), flush=True)

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "jobs": len(records),
        "success": sum(row["status"] == "SUCCESS" for row in records),
        "failed": sum(row["status"] == "FAILED" for row in records),
        "records": sorted(records, key=lambda row: (int(row["seed"]), str(row["method"]))),
    }
    (output / "smoke_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: summary[key] for key in ("jobs", "success", "failed")}, indent=2))
    raise SystemExit(0 if summary["failed"] == 0 else 1)


if __name__ == "__main__":
    main()
