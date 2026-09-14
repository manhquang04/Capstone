"""Run the single frozen RQ1 confirmatory attack batch."""

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
TARGET = ROOT / "artifacts/rq1/confirmatory_freeze_20260913/rq1_confirmatory_targets.pt"
TARGET_SHA256 = "caee0024c318c2874037382996b843ead8024b02b132185e502e245d3266267f"
OVERLAP = ROOT / "artifacts/rq1/confirmatory_freeze_20260913/source_overlap_matrix.json"
SOURCE_RUN = ROOT / "artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z"
GROUPS = list(range(39))
DNA_RUN_SEED = 1184685071


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command(branch: str, run: Path, group: int, python: str) -> list[str]:
    if branch == "raw":
        return [python, "experiments/run_phase4_harddiff_reparam_for_misselected.py", str(run),
                "--target-file", TARGET.name, "--groups", str(group), "--restarts", "8",
                "--init-mode", "standard", "--nonnegative-lambda", "0.001"]
    if branch == "dna":
        return [python, "experiments/run_phase4_dna_level1_forward_attack.py", str(run),
                "--target-file", TARGET.name, "--groups", str(group), "--candidates", "4",
                "--restarts", "8", "--iterations", "600", "--attack-lr", "0.1",
                "--init-mode", "standard", "--nonnegative-lambda", "0.001",
                "--block-size", "256", "--mix-ratio", "0.08", "--keep-ratio", "0.88",
                "--shrink-factor", "0.45", "--dna-run-seed", str(DNA_RUN_SEED)]
    if branch == "dp":
        return [python, "experiments/run_phase4_simple_defense_attack.py", str(run),
                "--target-file", TARGET.name, "--defense", "clipping_noise_mc", "--groups", str(group),
                "--restarts", "8", "--iterations", "600", "--attack-lr", "0.1",
                "--init-mode", "standard", "--nonnegative-lambda", "0.001",
                "--clip-norm", "100", "--noise-multiplier", "0.00025",
                "--mc-noise-samples", "100", "--defense-seed", "314159265"]
    raise ValueError(branch)


def run_job(job: dict, python: str, output: Path) -> dict:
    branch, group = job["branch"], int(job["group"])
    run = output / branch / f"group_{group}_run"
    run.mkdir(parents=True, exist_ok=False)
    shutil.copy2(SOURCE_RUN / "baseline_gate.json", run / "baseline_gate.json")
    shutil.copy2(SOURCE_RUN / "frozen.json", run / "frozen.json")
    shutil.copy2(TARGET, run / TARGET.name)
    cmd = command(branch, run, group, python)
    env = os.environ.copy()
    env.update({"PYTHONPATH": str(ROOT), "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"})
    started = time.perf_counter()
    with (run / "stdout.log").open("w") as stdout, (run / "stderr.log").open("w") as stderr:
        process = subprocess.run(cmd, cwd=ROOT, env=env, stdout=stdout, stderr=stderr, check=False)
    record = {**job, "command": cmd, "returncode": process.returncode,
              "seconds": time.perf_counter() - started,
              "status": "SUCCESS" if process.returncode == 0 else "FAILED"}
    (run / "job_record.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=9)
    parser.add_argument("--python", default=str(ROOT / ".venv-phase1/bin/python"))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/rq1/confirmatory_run_20260913")
    args = parser.parse_args()
    if not 1 <= args.workers <= 9:
        raise ValueError("workers must be in [1, 9]")
    config_text = (ROOT / "protocols/config/rq1_confirmatory.yaml").read_text()
    if "confirmatory_execution_authorized: true" not in config_text:
        raise RuntimeError("RQ1 confirmatory execution is not authorized")
    if sha256(TARGET) != TARGET_SHA256:
        raise RuntimeError("frozen target checksum mismatch")
    overlap = json.loads(OVERLAP.read_text())
    if overlap["disjointness_gate"] != "PASS" or overlap["max_overlap"] != 0:
        raise RuntimeError("source-disjointness gate failed")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    jobs = [{"branch": branch, "group": group} for branch in ("raw", "dna", "dp") for group in GROUPS]
    manifest = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
                "scope": "single frozen RQ1 confirmatory execution", "target_sha256": TARGET_SHA256,
                "groups": GROUPS, "branches": ["raw", "dna", "dp"], "workers": args.workers,
                "jobs": jobs}
    (output / "execution_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_job, job, args.python, output) for job in jobs]
        for future in as_completed(futures):
            row = future.result(); results.append(row)
            print(json.dumps({key: row[key] for key in ("branch", "group", "status", "seconds")}), flush=True)
    summary = {"created_at": datetime.now(timezone.utc).isoformat(), "jobs": len(results),
               "success": sum(r["status"] == "SUCCESS" for r in results),
               "failed": sum(r["status"] == "FAILED" for r in results),
               "records": sorted(results, key=lambda r: (r["branch"], r["group"]))}
    (output / "execution_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("jobs", "success", "failed")}, indent=2))
    raise SystemExit(0 if summary["failed"] == 0 else 1)


if __name__ == "__main__":
    main()
