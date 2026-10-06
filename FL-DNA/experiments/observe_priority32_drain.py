"""Telemetry-only sidecar: retain every completion while failed pool drains.

No training, target creation, result imputation, or statistics. The original
supervisor waits for queued futures after a worker exception; this observer
keeps required progress current without interrupting/re-running those jobs.
"""
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"artifacts/priority32_multidataset"


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic(path, doc):
    temporary = path.with_suffix(".observer.tmp")
    temporary.write_text(json.dumps(doc, indent=2)+"\n")
    temporary.replace(path)


def main():
    original = json.loads((OUT/"progress.json").read_text())
    start = datetime.fromisoformat(original["started_at"]).timestamp()
    pid = json.loads((OUT/"supervisor.lock.json").read_text())["pid"]
    seen = set()
    while True:
        events = [json.loads(line) for line in (OUT/"runs.jsonl").read_text().splitlines() if line.strip()]
        jobs = {e["path"]: e for e in events if e.get("stage") == "4_RQ2" and e["event"] == "job_launched"}
        known_failures = {e["path"] for e in events if e.get("stage") == "4_RQ2" and e["event"] == "job_failed"}
        completed = {}
        for path in jobs:
            file = Path(path)
            if file.exists():
                value = json.loads(file.read_text())
                assert value["status"] == "COMPLETED" and value["config_sha256"] == jobs[path]["config_sha256"]
                completed[path] = value
        for path in sorted(set(completed)-seen):
            value = completed[path]
            doc = {"stage": "4_RQ2_draining_after_numerical_failure", "dataset": value["dataset"],
                   "method": value["method"], "done": len(completed), "total": len(jobs),
                   "failed": len(known_failures), "started_at": original["started_at"], "last_update": now(),
                   "eta_minutes": (time.time()-start)/max(len(completed),1)*(len(jobs)-len(completed))/60}
            atomic(OUT/"progress.json", doc)
            with (OUT/"progress.log").open("a") as handle:
                handle.write(json.dumps(doc)+"\n")
            with (OUT/"observer.jsonl").open("a") as handle:
                handle.write(json.dumps({"at": now(), "event": "completed_result_observed", "path": path})+"\n")
        seen.update(completed)
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            missing = sorted(set(jobs)-set(completed))
            # Missing outputs are disclosed, never assumed to have any numeric result.
            atomic(OUT/"INCOMPLETE_NUMERICAL_FAILURE.json", {
                "at": now(), "completed": len(completed), "expected": len(jobs),
                "missing_result_paths": missing, "known_failed_paths": sorted(known_failures),
                "status": "AWAITING_USER_DIRECTION; no automatic rerun of scientifically nonfinite output"})
            break
        time.sleep(1)


if __name__ == "__main__":
    main()
