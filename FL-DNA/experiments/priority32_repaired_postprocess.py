"""Post-training repair analysis/cost routing; unchanged original formulas/runners.

All scientific writes are restricted to the new repair namespace. No training.
"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments import priority32_multidataset as p

ORIGINAL = p.OUT
OUT = ORIGINAL / "repair_trainable_raw_bn_20261002"


def main():
    training_lock = json.loads((OUT / "supervisor.lock.json").read_text())
    try:
        os.kill(training_lock["pid"], 0)
    except ProcessLookupError:
        pass
    else:
        raise RuntimeError("training supervisor still alive; cost must be isolated")
    lock = OUT / "postprocess.lock.json"
    if lock.exists():
        prior = json.loads(lock.read_text())
        try:
            os.kill(prior["pid"], 0)
        except ProcessLookupError:
            lock.rename(OUT / ("postprocess.previous." + str(time.time_ns()) + ".json"))
        else:
            raise RuntimeError("duplicate postprocess prohibited")
    p.OUT = OUT
    p.write(lock, {"pid": os.getpid(), "at": p.now()})
    freeze = json.loads((OUT / "execution_freeze.json").read_text())
    for job in freeze["jobs"]:
        assert p.valid_result(Path(job["path"]), job["config"]), job["path"]
    assert len(freeze["jobs"]) == 126
    for source, checksum in freeze["baseline_hashes"].items():
        source = p.ROOT / source
        assert p.sha(source) == checksum
        assert p.sha(OUT / source.relative_to(ORIGINAL)) == checksum
    assert len(freeze["baseline_hashes"]) == 63
    p.event("postprocess_started_or_resumed", command=sys.argv, pid=os.getpid(), code_sha256=p.sha(__file__))
    from experiments import analyze_priority32_multidataset as primary
    from experiments import verify_priority32_statistics as independent
    primary.OUT = independent.OUT = OUT
    p.checklist(5, "in_progress")
    if not (OUT / "rq2_analysis/independent_verification.json").exists():
        p.event("paired_analysis_started")
        primary.main()
        independent.main()
        p.event("paired_analysis_completed")
    else:
        assert json.loads((OUT / "rq2_analysis/independent_verification.json").read_text())["pass"]
        p.event("paired_analysis_validated_skip")
    p.checklist(5, "completed")
    p.checklist(6, "in_progress")
    from experiments import priority32_cost as cost
    cost.OUT = OUT
    if not (OUT / "rq3/COMPLETED.json").exists():
        p.event("rq3_started")
        cost.orchestrate()
    else:
        assert json.loads((OUT / "rq3/COMPLETED.json").read_text())["cells"] == 90
        p.event("rq3_completed_validated_skip")
    p.checklist(6, "completed")
    p.progress("ANALYSIS_COST_COMPLETE_PENDING_FINAL_REPORT", done=90, total=90)
    p.event("analysis_cost_completed")


if __name__ == "__main__":
    main()
