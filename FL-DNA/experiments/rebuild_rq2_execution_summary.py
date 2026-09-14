"""Rebuild the RQ2 registry summary from immutable per-job records.

This does not execute or retry training. It is used when the orchestration
process completes all child jobs but its final aggregate write is interrupted.
"""
from __future__ import annotations
import argparse, json, statistics
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(); p.add_argument("--run-dir",type=Path,default=ROOT/"artifacts/rq2/confirmatory_run_20260913"); a=p.parse_args()
    records=[json.loads(path.read_text()) for path in sorted(a.run_dir.glob("seed_*/**/job_record.json"))]
    metric_files=list(a.run_dir.glob("seed_*/**/metrics.json"))
    stderr_files=list(a.run_dir.glob("seed_*/**/stderr.log"))
    if len(records)!=84 or len(metric_files)!=84: raise RuntimeError(f"incomplete registry: records={len(records)}, metrics={len(metric_files)}")
    bad=[r for r in records if r.get("status")!="SUCCESS" or r.get("returncode")!=0]
    if bad: raise RuntimeError(f"failed records present: {bad}")
    seconds=[float(r["seconds"]) for r in records]
    summary={"schema_version":1,"created_at":datetime.now(timezone.utc).isoformat(),"source":"immutable per-job records; no training rerun","orchestration_note":"original parent process did not write its aggregate summary after all children completed","jobs":len(records),"success":len(records),"failed":0,"metrics":len(metric_files),"nonempty_stderr":sum(x.stat().st_size>0 for x in stderr_files),"machine_hours":sum(seconds)/3600,"median_seconds":statistics.median(seconds),"minimum_seconds":min(seconds),"maximum_seconds":max(seconds),"records":sorted(records,key=lambda r:(int(r["seed"]),str(r["method"]))) }
    (a.run_dir/"execution_summary_rebuilt.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps({k:summary[k] for k in ("jobs","success","failed","metrics","nonempty_stderr","machine_hours","median_seconds","minimum_seconds","maximum_seconds")},indent=2))

if __name__=="__main__": main()
