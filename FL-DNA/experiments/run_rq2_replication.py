"""Run one pre-frozen RQ2 conservative replication exactly once."""

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
METHODS = ["baseline", "dna_lossless", "dna_transform", "dp_0.00025"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def script_for(method: str) -> tuple[str, str]:
    return {
        "baseline": ("experiments/run_fraud_fl_baseline.py", "BASELINE_OUTPUT_PATH"),
        "dna_lossless": ("experiments/run_fraud_fl_dna.py", "DNA_OUTPUT_PATH"),
        "dna_transform": ("experiments/run_fraud_fl_dna_transform.py", "DNA_TRANSFORM_OUTPUT_PATH"),
    }.get(method, ("experiments/run_fraud_fl_dp.py", "DP_OUTPUT_PATH"))


def run_job(job: dict, python: str, output: Path) -> dict:
    seed, method = int(job["seed"]), job["method"]
    folder = output / f"seed_{seed}" / method
    folder.mkdir(parents=True, exist_ok=False)
    result = folder / "metrics.json"
    script, output_variable = script_for(method)
    env = os.environ.copy()
    env.update({
        "PYTHONPATH": str(ROOT), "FL_RUN_SEED": str(seed), "MAX_ROWS": "500000",
        "NUM_ROUNDS": "50", "LOCAL_EPOCHS": "1", "FL_NUM_CLIENTS": "3",
        "LOSS_TYPE": "focal", "FOCAL_ALPHA": "0.95", "FOCAL_GAMMA": "2.0",
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
        output_variable: str(result),
    })
    if method == "dna_transform":
        env.update({"DNA_TRANSFORM_BLOCK_SIZE": "256", "DNA_TRANSFORM_MIX": "0.08",
                    "DNA_TRANSFORM_KEEP": "0.88", "DNA_TRANSFORM_SHRINK": "0.45"})
    if method.startswith("dp_"):
        env.update({"DP_CLIP_NORM": "100.0", "DP_NOISE_MULTIPLIER": "0.00025"})
    command = [python, script]
    started = time.perf_counter()
    with (folder / "stdout.log").open("w") as stdout, (folder / "stderr.log").open("w") as stderr:
        process = subprocess.run(command, cwd=ROOT, env=env, stdout=stdout,
                                 stderr=stderr, check=False)
    row = {**job, "command": command, "returncode": process.returncode,
           "seconds": time.perf_counter() - started,
           "status": "SUCCESS" if process.returncode == 0 and result.exists() else "FAILED"}
    (folder / "job_record.json").write_text(json.dumps(row, indent=2) + "\n")
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
    config = json.loads(args.config.read_text())
    if not config.get("execution_authorized"):
        raise RuntimeError("replication execution is not authorized")
    contract = ROOT / config["contract"]["path"]
    if sha256(contract) != config["contract"]["sha256"]:
        raise RuntimeError("contract checksum mismatch")
    seeds = config["seeds"]
    if json.loads(contract.read_text())["seeds"] != seeds:
        raise RuntimeError("seed order/contract mismatch")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    jobs = [{"seed": seed, "method": METHODS[(index + wave) % 4], "wave": wave}
            for wave in range(4) for index, seed in enumerate(seeds)]
    manifest = {"created_at": datetime.now(timezone.utc).isoformat(),
                "config_sha256": sha256(args.config), "contract_sha256": sha256(contract),
                "seeds": seeds, "methods": METHODS, "workers": args.workers, "jobs": jobs}
    (output / "execution_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_job, job, args.python, output) for job in jobs]
        for future in as_completed(futures):
            row = future.result(); rows.append(row)
            print(json.dumps({key: row[key] for key in ("seed", "method", "status", "seconds")}), flush=True)
    summary = {"created_at": datetime.now(timezone.utc).isoformat(), "jobs": len(rows),
               "success": sum(row["status"] == "SUCCESS" for row in rows),
               "failed": sum(row["status"] == "FAILED" for row in rows),
               "records": sorted(rows, key=lambda row: (row["seed"], row["method"]))}
    (output / "execution_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: summary[key] for key in ("jobs", "success", "failed")}, indent=2))
    raise SystemExit(0 if summary["failed"] == 0 else 1)


if __name__ == "__main__":
    main()
