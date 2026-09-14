"""Run the one-time, predeclared RQ2 development utility-grid extension."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_PATH = Path(__file__).resolve().parents[1]
if str(ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(ROOT_PATH))

from experiments.run_rq2_stage1_batch import ROOT, SEEDS, run_job, sha256
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

MULTIPLIERS = [0.00001, 0.000025, 0.00005, 0.000075]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--python", default=str(ROOT / ".venv-phase1/bin/python"))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/rq2/utility_extension_20260913")
    args = parser.parse_args()
    if not 1 <= args.workers <= 9:
        raise ValueError("workers must be in [1,9]")
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=False)
    jobs = [{"seed": seed, "method": f"dp_{multiplier:g}", "wave": wave} for wave, multiplier in enumerate(MULTIPLIERS) for seed in SEEDS]
    manifest = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(), "scope": "one-time development-only utility-grid extension", "seeds": SEEDS, "multipliers": MULTIPLIERS, "strict_tolerances": {"f1": 0.01, "auc": 0.0025}, "relaxed_sensitivity_tolerances": {"f1": 0.02, "auc": 0.005}, "privacy_metrics_used_for_grid": False, "amendment": "protocols/amendments/2026-09-13_rq1_variant_and_utility_extension.md", "implementation_hashes": {path: sha256(ROOT / path) for path in ["experiments/run_rq2_stage1_batch.py", "experiments/run_fraud_fl_dp.py", "privacy/dp_engine.py", "data/load_creditcard.py", "experiments/fraud_fl_common.py"]}, "jobs": jobs}
    (output / "extension_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    records=[]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures=[executor.submit(run_job, job, args.python, output) for job in jobs]
        for future in as_completed(futures):
            row=future.result(); records.append(row); print(json.dumps({key: row.get(key) for key in ("seed","method","status","seconds")}), flush=True)
    summary={"jobs":len(records),"success":sum(r["status"] in ("SUCCESS","SKIPPED_EXISTING_SUCCESS") for r in records),"failed":sum(r["status"]=="FAILED" for r in records),"records":sorted(records,key=lambda r:(r["seed"],r["method"]))}
    (output/"extension_execution_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    raise SystemExit(0 if summary["failed"]==0 else 1)


if __name__ == "__main__":
    main()
