"""Run Priority 14 DP comparator RQ2 utility jobs.

This runner executes only the two missing DP branches.  The paired unprotected
baseline is reused from the frozen RQ2-v2 confirmatory run specified in the
config.
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


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_job(job: dict, python: str, output: Path, config: dict) -> dict:
    seed = int(job["seed"])
    method_id = job["method_id"]
    folder = output / f"seed_{seed}" / method_id
    folder.mkdir(parents=True, exist_ok=False)
    result = folder / "metrics.json"
    training = config["training"]
    dp = job["dp"]
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(ROOT),
            "FL_RUN_SEED": str(seed),
            "MAX_ROWS": str(training["maximum_rows"]),
            "NUM_ROUNDS": str(training["num_rounds"]),
            "LOCAL_EPOCHS": str(training["local_epochs"]),
            "FL_NUM_CLIENTS": str(training["num_clients"]),
            "LOSS_TYPE": "focal",
            "FOCAL_ALPHA": str(training["focal_alpha"]),
            "FOCAL_GAMMA": str(training["focal_gamma"]),
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "DP_OUTPUT_PATH": str(result),
            "DP_CLIP_NORM": str(dp["clip_norm"]),
            "DP_NOISE_MULTIPLIER": str(dp["noise_multiplier"]),
        }
    )
    command = [python, "experiments/run_fraud_fl_dp.py"]
    started = time.perf_counter()
    with (folder / "stdout.log").open("w") as stdout, (folder / "stderr.log").open("w") as stderr:
        process = subprocess.run(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr, check=False)
    row = {
        **job,
        "command": command,
        "returncode": process.returncode,
        "seconds": time.perf_counter() - started,
        "status": "SUCCESS" if process.returncode == 0 and result.exists() else "FAILED",
    }
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

    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    if not config.get("execution_authorized"):
        raise RuntimeError("Priority 14 DP utility execution is not authorized")
    if config.get("status") != "CONFIRMATORY FROZEN":
        raise RuntimeError("Priority 14 DP utility config is not confirmatory frozen")
    seeds = [int(seed) for seed in config["confirmatory_seeds"]["values"]]
    if len(seeds) != int(config["statistics"]["required_confirmatory_seed_count"]):
        raise RuntimeError("Seed count does not match required confirmatory n")
    if len(set(seeds)) != len(seeds):
        raise RuntimeError("Duplicate confirmatory seeds")

    baseline = config["baseline_source"]
    baseline_config = ROOT / baseline["config"]
    baseline_summary = ROOT / baseline["run_dir"] / "execution_summary.json"
    if sha256(baseline_config) != baseline["config_sha256"]:
        raise RuntimeError("Baseline config checksum mismatch")
    if sha256(baseline_summary) != baseline["execution_summary_sha256"]:
        raise RuntimeError("Baseline execution summary checksum mismatch")

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    jobs = [
        {"seed": seed, "method_id": method["method_id"], "dp": method}
        for seed in seeds
        for method in config["dp_methods"]
    ]
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Priority 14 DP comparator utility confirmatory execution",
        "config": str(config_path.relative_to(ROOT)),
        "config_sha256": sha256(config_path),
        "baseline_reuse": baseline,
        "seeds": seeds,
        "dp_methods": config["dp_methods"],
        "workers": args.workers,
        "jobs": jobs,
    }
    (output / "execution_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_job, job, args.python, output, config) for job in jobs]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(json.dumps({key: row[key] for key in ("seed", "method_id", "status", "seconds")}), flush=True)
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "jobs": len(rows),
        "success": sum(row["status"] == "SUCCESS" for row in rows),
        "failed": sum(row["status"] == "FAILED" for row in rows),
        "records": sorted(rows, key=lambda row: (row["seed"], row["method_id"])),
    }
    (output / "execution_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: summary[key] for key in ("jobs", "success", "failed")}, indent=2))
    raise SystemExit(0 if summary["failed"] == 0 else 1)


if __name__ == "__main__":
    main()
