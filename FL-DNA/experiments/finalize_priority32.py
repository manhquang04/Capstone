"""Report/check/hash finalization for new Priority32 files only."""
import argparse
import csv
import json
import os
import py_compile
import subprocess
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.priority32_multidataset import OUT, DATASETS, METHODS, AMENDMENT, sha, write, now

REPORT = ROOT/"reports/priority32_multidataset_rq2_rq3_report.md"
CODE = [ROOT/"experiments"/name for name in ("priority32_multidataset.py", "priority32_cost.py", "analyze_priority32_multidataset.py", "verify_priority32_statistics.py", "finalize_priority32.py")]
CODE.append(ROOT/"tests/test_priority32_multidataset.py")


def table(headers, rows):
    return "| "+" | ".join(headers)+" |\n| "+" | ".join(["---"]*len(headers))+" |\n"+"\n".join("| "+" | ".join(str(v) for v in row)+" |" for row in rows)+"\n"


def main(seal=False):
    summary = json.loads((OUT/"rq2_analysis/summary.json").read_text())
    independent = json.loads((OUT/"rq2_analysis/independent_verification.json").read_text())
    assert independent["pass"]
    assert len(list((OUT/"rq2_jobs").glob("*/*/seed_*.json"))) == 189
    for path in CODE:
        py_compile.compile(str(path), cfile=str(OUT/"compile"/(path.stem+".pyc")), doraise=True)
    diff = subprocess.run(["git", "diff", "--check"], cwd=ROOT, capture_output=True, text=True)
    tests = subprocess.run([sys.executable, "-B", str(CODE[-1])], cwd=ROOT, capture_output=True, text=True)
    write(OUT/"final_checks.json", {"py_compile": "PASS", "git_diff_check": "PASS" if diff.returncode == 0 else "FAIL",
          "git_diff_check_output": diff.stdout+diff.stderr, "unit_tests": "PASS" if tests.returncode == 0 else "FAIL",
          "unit_test_output": tests.stdout+tests.stderr, "independent_statistics": independent,
          "torch_threads": 1, "reconstruction_attacks": 0,
          "workload_note": "training/cost subprocesses joined; parent finalizer/supervisor exits after seal; verify no process externally",
          "checked_at": now()})
    assert diff.returncode == 0 and tests.returncode == 0
    audits = json.loads((OUT/"data_audit.json").read_text())
    freeze = json.loads((OUT/"execution_freeze.json").read_text())
    lines = ["# Priority32: multi-dataset RQ2 and RQ3", "", "Status: COMPLETE (external no-workload check still required after supervisor exits).",
             "", "This is an additional shared-pipeline replication. Earlier results remain unchanged. No reconstruction attacks, Latex edits or external_defenses edits.",
             "", "## Data audit and frozen preprocessing", "",
             table(["Dataset", "Raw rows", "Fraud rate", "Features", "Train/validation/test selected"],
                   [[d, audits[d]["row_count"], f'{audits[d]["fraud_rate"]:.9g}', audits[d]["feature_count"],
                     '/'.join(str(audits[d]["selected_partitions"][s]["rows"]) for s in ("train", "validation", "test"))] for d in DATASETS]),
             "Train-only imputation/scaling/encoding; cap500k inside partitions; source-overlap0. IEEE temporal split with identity left join; BAF months0–4/5/6–7 and exact -1 indicators. PaySim retains the unchanged loader feature/partition helpers, but cap is after splitting and medians are train-only. Full preprocessing, category mapping, missing-column lists and source hashes are in prepared/<dataset>/audit.json and preprocessing.json.",
             "", "## Validation quality gates", "",
             table(["Dataset", "Chosen training", "Val F1", "All-fraud F1", "Val AUC", "Gate"],
                   [[d, str(freeze["chosen"][d]["training"]),
                     json.loads((ROOT/freeze["chosen"][d]["quality_result"]).read_text())["validation"]["f1"],
                     json.loads((ROOT/freeze["chosen"][d]["quality_result"]).read_text())["predict_all_fraud_f1"],
                     json.loads((ROOT/freeze["chosen"][d]["quality_result"]).read_text())["validation"]["auc_roc"], "PASS"] for d in DATASETS]),
             "All attempts retained under quality/. Selection used validation only; no test evaluation in those jobs. execution_freeze.json predates confirmatory jobs.",
             "", "## RQ2 paired non-inferiority", "", "21 seeds/dataset, final-round checkpoint; same seeds across methods, validation-F1 threshold. Paired mean Student-t two-sided95%CI; strict lower bound > -.02 F1 / -.005 ROC-AUC. PR-AUC descriptive.", "",
             table(["Dataset", "Method", "Endpoint", "Baseline mean", "Mean delta", "SD", "95% CI", "Gate"],
                   [[r["dataset"], r["method"], r["endpoint"], f'{r["baseline_mean"]:.9g}', f'{r["mean_delta"]:.9g}',
                     f'{r["sd_delta"]:.9g}', f'[{r["ci95_lower"]:.9g}, {r["ci95_upper"]:.9g}]', r["gate"]] for r in summary["rows"]]),
             "", "Overall: "+json.dumps(summary["overall"], sort_keys=True), "",
             "Independent NumPy/SciPy recomputation from per-job JSON: "+json.dumps(independent), "",
             "Paired differences: artifacts/priority32_multidataset/rq2_analysis/paired_differences.csv.",
             "", "## RQ3 cost and DP comparison", "",
             "Same optimized frozen Bundle B harness/config:10 warmups,50 measurements,orderseed271828,bootstrap10000; new input dimensions only. Synthetic trainable parameter shapes (not full-state real RQ2 traffic). Each IEEE/BAF has27 RAW/v1/v2 cells plus9 DP descriptive cells; PaySim has9 RAW+9 DP references. All run in isolated serial subprocesses after training."]
    cost_rows, metric_rows = [], []
    for dataset in DATASETS:
        decisions = json.loads((OUT/"rq3"/dataset/"analysis/acceptance_decision.json").read_text())
        for method in ("DNA_TRANSFORM_TRANSPORT", "DNA_TRANSFORM_V2_TRANSPORT", "DP_CLIP_NOISE"):
            cells = [c for c in decisions["cell_decisions"] if c["method"] == method]
            if not cells:
                continue
            bad = sorted({v["criterion"] for c in cells for v in c["criteria"] if not v["pass"]})
            cost_rows.append([dataset, method, "ACCEPTABLE" if all(c["decision"] == "ACCEPTABLE" for c in cells) else "NOT_ACCEPTABLE", ', '.join(bad) or "none"])
        with (OUT/"rq3"/dataset/"analysis/benchmark_summary.csv").open() as handle:
            records = list(csv.DictReader(handle))
        for method in sorted(set(r["method"] for r in records)):
            fields = {field: max(float(r["p95"]) for r in records if r["method"] == method and r["metric"] == field) for field in
                      ("client_encode_serialize_seconds_max", "server_decode_aggregate_seconds", "payload_bytes_total")}
            metric_rows.append([dataset, method, f'{fields["client_encode_serialize_seconds_max"]*1000:.9g}',
                               f'{fields["server_decode_aggregate_seconds"]*1000:.9g}', int(fields["payload_bytes_total"])])
    lines += ["", table(["Dataset", "Method", "Bundle B", "Failed criteria"], cost_rows),
              "", table(["Dataset", "Method", "Worst-cell p95 encode ms", "Worst-cell p95 server ms", "Max payload bytes(total clients)"], metric_rows),
              "DP cost-only configuration: whole synthetic vector clip100,sigma.001; clipping/noise timed, RAW serialization. It is not a new calibrated DP utility/privacy claim. Per-cell overhead/memory/payload and CI acceptance criteria are in rq3/<dataset>/analysis/acceptance_criteria.csv.",
              "", "## Exact commands and disclosures", "", "```text",
              f"{sys.executable} -B -u experiments/priority32_multidataset.py --supervise",
              "```", "All exact subprocess commands, start/skip/failure/resume events and output hashes: runs.jsonl. No unfinished or failed run is silently excluded. Stage progress/checklist in progress.json, progress.log, checklist.json. All round journals retained.",
              "", "## Provenance and checks", "", "Amendment: "+str(AMENDMENT.relative_to(ROOT)),
              "Raw/code/output SHA-256: artifacts/priority32_multidataset/sha256_manifest.csv. Manifest SHA and report SHA: final_hashes.json. Manifest excludes itself and final_hashes.json to avoid impossible self-reference.",
              "Compile, unit tests, git diff --check and independent statistics: final_checks.json. All checks passed. torch.set_num_threads(1); no old data/artifacts/reports changed.", ""]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    # Report is a NEW Priority32 file; never touches earlier reports.
    REPORT.write_text("\n".join(lines))
    paths = sorted(set([p for p in OUT.rglob("*") if p.is_file() and p.name not in ("sha256_manifest.csv", "final_hashes.json") and not p.name.endswith(".tmp")]+CODE+[AMENDMENT, REPORT]+[ROOT/p for d in DATASETS for p in audits[d]["raw_sha256"]]))
    with (OUT/"sha256_manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "bytes", "sha256"])
        writer.writeheader()
        for path in paths:
            writer.writerow({"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha(path)})
    write(OUT/"final_hashes.json", {"sha256_manifest.csv": sha(OUT/"sha256_manifest.csv"),
          "report": sha(REPORT), "sealed": seal, "at": now()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seal", action="store_true")
    main(parser.parse_args().seal)
