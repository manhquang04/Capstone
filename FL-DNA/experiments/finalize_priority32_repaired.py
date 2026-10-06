"""Seal the authorized repair report without editing original P32 outputs."""
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments import priority32_multidataset as p

ORIGINAL = p.OUT
OUT = ORIGINAL / "repair_trainable_raw_bn_20261002"
REPORT = p.ROOT / "reports/priority32_repaired_trainable_rq2_rq3_report_20261002.md"
AMENDMENT = p.ROOT / "protocols/amendments/2026-10-02_priority32_bn_domain_repair_authorized.md"


def main():
    for filename in ("supervisor.lock.json", "postprocess.lock.json"):
        pid = json.loads((OUT / filename).read_text())["pid"]
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            pass
        else:
            raise RuntimeError(f"workload still live: {filename} pid={pid}")
    p.OUT = OUT
    from experiments import finalize_priority32 as original
    original.OUT, original.REPORT, original.AMENDMENT = OUT, REPORT, AMENDMENT
    extra = [p.ROOT / "experiments" / name for name in (
        "priority32_repaired_trainable.py", "priority32_repaired_postprocess.py",
        "finalize_priority32_repaired.py", "diagnose_priority32_numerical_failure.py")]
    extra.append(p.ROOT / "tests/test_priority32_repaired_trainable.py")
    original.CODE.extend(extra)
    for dataset in p.DATASETS:
        config = OUT / "rq3" / dataset / "config.json"
        from experiments.priority32_cost import valid_cell
        cells = json.loads((OUT / "rq3" / dataset / "cell_order.json").read_text())["cells"]
        for profile, method, count in cells:
            path = OUT / "rq3" / dataset / "cells" / f"{profile.lower()}__{method.lower()}__c{count}.json"
            assert valid_cell(path, config, (profile, method, count))
    assert json.loads((OUT / "rq3/COMPLETED.json").read_text())["cells"] == 90
    original.main(seal=False)
    previous_tests = subprocess.run([sys.executable, "-B", str(p.ROOT / "tests/test_priority32_multidataset.py")],
                                    cwd=p.ROOT, capture_output=True, text=True)
    assert previous_tests.returncode == 0, previous_tests.stdout + previous_tests.stderr
    checks = json.loads((OUT / "final_checks.json").read_text())
    checks.update({"original_pipeline_unit_tests": "PASS", "original_pipeline_unit_test_output":
                   previous_tests.stdout + previous_tests.stderr,
                   "no_workload_lock_checks": "PASS", "validated_cost_cells": 90,
                   "protocol_variant": "trainable_only_transform_raw_bn"})
    p.write(OUT / "final_checks.json", checks)
    text = REPORT.read_text()
    text = text.replace("Status: COMPLETE (external no-workload check still required after supervisor exits).",
                        "Status: COMPLETE. Training and cost supervisors have exited; live-PID checks passed.")
    text = text.replace("This is an additional shared-pipeline replication.",
                        "This is the separately authorized trainable-only transform / raw BN utility variant.")
    text = text.replace("All attempts retained under quality/.",
                        "Original validation attempts retained under artifacts/priority32_multidataset/quality/ and reused without rerunning.")
    text = text.replace("artifacts/priority32_multidataset/rq2_analysis/", str(OUT.relative_to(p.ROOT)) + "/rq2_analysis/")
    text = text.replace("artifacts/priority32_multidataset/sha256_manifest.csv", str(OUT.relative_to(p.ROOT)) + "/sha256_manifest.csv")
    text = text.replace("experiments/priority32_multidataset.py --supervise", "experiments/priority32_repaired_trainable.py --supervise")
    text += "\n## Authorized repair scope and full run disclosure\n\n"
    text += (
        "The original full-state experiment remains unchanged: 189 submitted attempts, 88 finite outputs and 101 missing outputs. "
        "One authorized diagnostic replay of BAF/v1/321001 reproduced NaN evaluation and found negative BN running variance "
        "from round 2; not all missing original jobs have independently identified causes. "
        "The diagnostic report is reports/priority32_numerical_failure_diagnosis_20261002.md.\n\n"
        "The human-approved repair transforms trainable parameters only, restoring all nontrainable local BN buffers "
        "before ordinary sample-weighted FedAvg. BN buffers remain transmitted RAW. This is NOT FedBN and does NOT "
        "protect the BN leakage channel. Results do not replace earlier full-state evidence or the registered PaySim result. "
        "All 126 transform jobs, including 25 previously finite jobs, were rerun in the new repair namespace. "
        "63 baseline files were reused as byte-identical copies; every frozen source/copy checksum was verified. "
        "No clamping, imputation, seed exclusions or scientific tuning.\n\n"
        "Training completed 126/126 with zero failures in 4272.953714 seconds (71.215895 minutes). "
        "No repaired training interruption or resume. The postprocess first attempt stopped before statistics/cost "
        "because valid_result received a string instead of a Path. The path conversion alone was fixed and the driver "
        "resumed; old stderr and the interruption entry remain retained. No training output was recomputed.\n\n"
    )
    cost_seconds = json.loads((OUT / "rq3/COMPLETED.json").read_text())["elapsed_seconds"]
    text += f"Cost stage: all 90 cells completed in {cost_seconds:.6f} seconds ({cost_seconds/60:.6f} minutes), no failed subprocess or resume.\n\n"
    text += (
        "Postprocess exact launch: .venv-phase1/bin/python -B -u experiments/priority32_repaired_postprocess.py. "
        "Finalization: .venv-phase1/bin/python -B experiments/finalize_priority32_repaired.py. "
        "Long drivers used nohup, stdin DEVNULL, separate append-only stdout/stderr and detached sessions. "
        "All original/repair unit tests passed. Prepared audit/preprocessing source files remain under the original "
        "prepared/<dataset>/ namespace; only audit metadata was copied into the repair namespace for cost shape routing.\n"
    )
    REPORT.write_text(text)
    p.checklist(7, "completed")
    p.progress("COMPLETED", done=189, total=189)
    p.event("repair_finalization_completed", command=sys.argv, report=str(REPORT))
    audits = json.loads((OUT / "data_audit.json").read_text())
    paths = set(path for path in OUT.rglob("*") if path.is_file()
                and path.name not in ("sha256_manifest.csv", "final_hashes.json")
                and not path.name.endswith(".tmp"))
    paths.update(original.CODE + [AMENDMENT, REPORT, p.AMENDMENT])
    for dataset in p.DATASETS:
        paths.update(p.ROOT / path for path in audits[dataset]["raw_sha256"])
        paths.update((ORIGINAL / "prepared" / dataset).glob("*.json"))
    manifest = OUT / "sha256_manifest.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "bytes", "sha256"])
        writer.writeheader()
        for path in sorted(paths):
            writer.writerow({"path": str(path.relative_to(p.ROOT)), "bytes": path.stat().st_size, "sha256": p.sha(path)})
    p.write(OUT / "final_hashes.json", {"sha256_manifest.csv": p.sha(manifest), "report": p.sha(REPORT),
                                       "sealed": True, "at": p.now()})
    print(json.dumps({"sealed": True, "report": str(REPORT)}), flush=True)


if __name__ == "__main__":
    main()
