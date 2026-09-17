"""Frozen development utility sweep for predeclared strong update-DP budgets."""

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
SEEDS = (101, 202, 303)
MULTIPLIERS = {
    "epsilon_100": 0.09866372299196495,
    "epsilon_50": 0.15890264833373352,
    "epsilon_10": 0.5678974412358136,
    "epsilon_1": 4.900556183310786,
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def execute(job: dict, python: str, out: Path) -> dict:
    folder = out / f"seed_{job['seed']}" / job["method"]
    folder.mkdir(parents=True, exist_ok=False)
    metrics = folder / "metrics.json"
    env = os.environ.copy()
    env.update({
        "PYTHONPATH": str(ROOT), "FL_RUN_SEED": str(job["seed"]),
        "MAX_ROWS": "100000", "NUM_ROUNDS": "10", "LOCAL_EPOCHS": "1",
        "FL_NUM_CLIENTS": "3", "LOSS_TYPE": "focal", "FOCAL_ALPHA": "0.95",
        "FOCAL_GAMMA": "2.0", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
    })
    if job["method"] == "baseline":
        command = [python, "experiments/run_fraud_fl_baseline.py"]
        env["BASELINE_OUTPUT_PATH"] = str(metrics)
    else:
        command = [python, "experiments/run_fraud_fl_dp.py"]
        env["DP_OUTPUT_PATH"] = str(metrics)
        env["DP_NOISE_MULTIPLIER"] = str(job["noise_multiplier"])
        env["DP_NOISE_PRESET"] = str(job["method"])
        env["DP_CLIP_NORM"] = "100"
    started = time.perf_counter()
    with (folder / "stdout.log").open("w") as so, (folder / "stderr.log").open("w") as se:
        done = subprocess.run(command, cwd=ROOT, env=env, stdout=so, stderr=se, check=False)
    record = {"seed": job["seed"], "method": job["method"], "noise_multiplier": job.get("noise_multiplier"),
              "command": command, "returncode": done.returncode, "seconds": time.perf_counter() - started,
              "metrics": str(metrics), "status": "SUCCESS" if done.returncode == 0 and metrics.exists() else "FAILED"}
    (folder / "job_record.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--workers", type=int, default=9); p.add_argument("--python", default=str(ROOT / ".venv-phase1/bin/python")); a = p.parse_args()
    if not 1 <= a.workers <= 9: raise ValueError("workers must be in 1..9")
    out = a.output_dir.resolve(); out.mkdir(parents=True, exist_ok=False)
    jobs = [{"seed": seed, "method": "baseline"} for seed in SEEDS]
    jobs += [{"seed": seed, "method": method, "noise_multiplier": noise} for seed in SEEDS for method, noise in MULTIPLIERS.items()]
    manifest = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
      "scope": "development-only frozen strong update-DP utility sweep; not confirmatory",
      "amendment": "protocols/amendments/2026-09-16_strong_update_dp_exploratory_pareto.md",
      "seeds": SEEDS, "max_rows": 100000, "rounds": 10, "clients": 3, "clip_norm": 100.0,
      "multipliers": MULTIPLIERS, "jobs": jobs,
      "code_hashes": {x: sha256(ROOT / x) for x in ("experiments/run_fraud_fl_dp.py", "experiments/run_fraud_fl_baseline.py", "experiments/fraud_fl_common.py", "privacy/dp_engine.py", "data/load_creditcard.py")}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    records = []
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures = [pool.submit(execute, job, a.python, out) for job in jobs]
        for future in as_completed(futures):
            record = future.result(); records.append(record); print(json.dumps({k: record[k] for k in ("seed", "method", "status", "seconds")}), flush=True)
    payload = {"created_at": datetime.now(timezone.utc).isoformat(), "jobs": len(records), "success": sum(r["status"] == "SUCCESS" for r in records), "failed": sum(r["status"] == "FAILED" for r in records), "records": sorted(records, key=lambda r: (r["seed"], r["method"]))}
    (out / "execution_summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    if payload["failed"]: raise SystemExit(1)


if __name__ == "__main__": main()
