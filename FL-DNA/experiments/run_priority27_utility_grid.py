"""Run Priority 27 C2 DP-variant utility grid."""

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


def method_id(variant: str, sigma: float) -> str:
    return f"{variant}__sigma_{sigma:.0e}".replace("-", "m").replace("+", "")


def run_job(job: dict, python: str, output: Path, config: dict) -> dict:
    seed = int(job["seed"])
    method = str(job["method"])
    folder = output / f"seed_{seed}" / method
    folder.mkdir(parents=True, exist_ok=False)
    result = folder / "metrics.json"
    training = config["training"]
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
        }
    )
    if method == "baseline":
        script = "experiments/run_fraud_fl_baseline.py"
        env["BASELINE_OUTPUT_PATH"] = str(result)
    else:
        script = "experiments/run_fraud_fl_dp_priority27.py"
        env["PRIORITY27_DP_OUTPUT_PATH"] = str(result)
        env["PRIORITY27_DP_VARIANT"] = str(job["variant"])
        env["PRIORITY27_DP_NOISE_MULTIPLIER"] = str(job["noise_multiplier"])
        env["PRIORITY27_DP_CLIP_SPEC_JSON"] = json.dumps(config["clip_specs"][job["variant"]])
    command = [python, script]
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
    (folder / "job_record.json").write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")
    return row


def build_jobs(config: dict, extension_only: bool) -> list[dict]:
    seeds = [int(seed) for seed in config["seeds"]]
    variants = list(config["clip_specs"])
    sigmas = [float(x) for x in (config["extension_sigma_grid"] if extension_only else config["initial_sigma_grid"])]
    jobs: list[dict] = []
    if not extension_only:
        jobs.extend({"seed": seed, "method": "baseline"} for seed in seeds)
    for seed in seeds:
        for variant in variants:
            for sigma in sigmas:
                jobs.append({"seed": seed, "method": method_id(variant, sigma), "variant": variant, "noise_multiplier": sigma})
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=9)
    parser.add_argument("--extension-only", action="store_true")
    parser.add_argument("--python", default=str(ROOT / ".venv-phase1/bin/python"))
    args = parser.parse_args()
    if not 1 <= args.workers <= 9:
        raise ValueError("workers must be in [1, 9]")
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    if config.get("status") != "UTILITY_GRID_FROZEN" or not config.get("execution_authorized"):
        raise RuntimeError("Priority 27 utility grid config is not frozen/authorized")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    jobs = build_jobs(config, args.extension_only)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Priority 27 C2 utility grid",
        "extension_only": args.extension_only,
        "config": str(config_path.relative_to(ROOT)),
        "config_sha256": sha256(config_path),
        "workers": args.workers,
        "jobs": jobs,
    }
    (output / "execution_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_job, job, args.python, output, config) for job in jobs]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(json.dumps({key: row[key] for key in ("seed", "method", "status", "seconds")}), flush=True)
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "jobs": len(rows),
        "success": sum(row["status"] == "SUCCESS" for row in rows),
        "failed": sum(row["status"] == "FAILED" for row in rows),
        "records": sorted(rows, key=lambda row: (row["seed"], row["method"])),
    }
    (output / "execution_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: summary[key] for key in ("jobs", "success", "failed")}, indent=2))
    raise SystemExit(0 if summary["failed"] == 0 else 1)


if __name__ == "__main__":
    main()

