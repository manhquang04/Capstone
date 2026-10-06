"""Priority34A analysis and independent recomputation; no training here."""
from __future__ import annotations
import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path
import numpy as np
from scipy import stats
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from experiments import priority34a_local_bn as r

ENDPOINTS = ("f1", "auc_roc", "pr_auc")
MARGINS = {"f1": -.02, "auc_roc": -.005}
REPORT = r.ROOT / "reports/priority34a_report.md"


def paired_summary(values):
    values = np.asarray(values, dtype=float)
    if len(values) != 21 or not np.isfinite(values).all():
        raise ValueError("exact21finite paired replicates required")
    mean, sd = float(values.mean()), float(values.std(ddof=1))
    half = float(stats.t.ppf(.975, 20))*sd/math.sqrt(21)
    return {"n": 21, "mean_delta": mean, "sd": sd, "ci_low": mean-half,
            "ci_high": mean+half, "median_delta": float(np.median(values)),
            "iqr": [float(item) for item in np.quantile(values, [.25, .75])]}


def independent_summary(values):
    # Separate arithmetic implementation, not a call to paired_summary.
    count = len(values)
    if count != 21:
        raise ValueError("incomplete independent pairs")
    mean = math.fsum(values)/count
    sample_variance = math.fsum((value-mean)**2 for value in values)/(count-1)
    sd = math.sqrt(sample_variance)
    half = stats.t.isf(.025, count-1)*math.sqrt(sample_variance/count)
    return {"mean_delta": mean, "sd": sd, "ci_low": mean-half, "ci_high": mean+half}


def finish():
    manifest = r.verify_freeze()
    r.checklist(4, "in_progress")
    records, independent, client_rows = {}, {}, []
    max_metric_error = 0.
    for job in manifest["jobs"]:
        path = r.OUT / job["result"]
        if not r.valid_result(path, job["config"]):
            raise ValueError("missing job, no incomplete analysis")
        doc = json.loads(path.read_text())
        key = (doc["dataset"], doc["method"], doc["seed"])
        records[key] = doc
        reloaded_clients = []
        for client in doc["clients"]:
            recomputed = {}
            for split in ("validation", "test"):
                candidate = [relative for relative in doc["artifact_sha256"] if relative.endswith(f"client{client['client']}_{split}_probabilities.npy")]
                if len(candidate) != 1:
                    raise ValueError("probability provenance")
                probability = np.load(r.OUT / candidate[0], allow_pickle=False)
                labels = np.load(r.p.OUT / "prepared" / doc["dataset"] / (split+"_y.npy"), allow_pickle=False)
                if len(probability) != len(labels) or not np.isfinite(probability).all():
                    raise ValueError("probability audit")
                threshold = client["threshold"]
                if split == "validation" and r.p.tune_threshold(labels, probability) != threshold:
                    raise ValueError("validation threshold mismatch")
                recomputed[split] = {"f1": float(f1_score(labels, probability >= threshold, zero_division=0)),
                                     "auc_roc": float(roc_auc_score(labels, probability)),
                                     "pr_auc": float(average_precision_score(labels, probability))}
                for endpoint in ENDPOINTS:
                    error = abs(recomputed[split][endpoint]-client[split][endpoint])
                    max_metric_error = max(max_metric_error, error)
                    if error > 1e-12:
                        raise ValueError("reloaded client metric mismatch")
            reloaded_clients.append(recomputed["test"])
            client_rows.append({"dataset": doc["dataset"], "method": doc["method"], "seed": doc["seed"],
                                "client": client["client"], "threshold": threshold, **recomputed["test"]})
        independent[key] = {endpoint: math.fsum(client[endpoint] for client in reloaded_clients)/3 for endpoint in ENDPOINTS}
        checkpoint_candidates = [relative for relative in doc["artifact_sha256"] if relative.endswith("final_checkpoint.pt")]
        checkpoint = r.torch.load(r.OUT / checkpoint_candidates[0], map_location="cpu")
        dimension = np.load(r.p.OUT / "prepared" / doc["dataset"] / "train_x.npy", mmap_mode="r").shape[1]
        model = r.p.FraudMLP(dimension)
        bn, allowed = r.domains(model)
        r.assert_payload(checkpoint["global_non_bn"], bn, allowed)
        for state in checkpoint["client_bn"]:
            if set(state) != bn:
                raise ValueError("local BN checkpoint domain")
            combined = model.state_dict()
            combined.update(checkpoint["global_non_bn"]); combined.update(state)
            model.load_state_dict(combined)
            r.guard(model, "independent checkpoint audit")
        journal_candidates = [relative for relative in doc["artifact_sha256"] if relative.endswith("rounds.jsonl")]
        rounds = [json.loads(line) for line in (r.OUT / journal_candidates[0]).read_text().splitlines()]
        if [row["round"] for row in rounds] != list(range(1, 51)):
            raise ValueError("round audit mismatch")
        for row in rounds:
            if row["transmitted_keys"] != list(allowed) or not row["no_bn_transmitted"] or row["upload_assertions"] != 3:
                raise ValueError("journal firewall")
            for minima in row["bn_minima_all_steps"]:
                if any(not math.isfinite(value) or value < 0 for value in minima.values()):
                    raise ValueError("journal BN gate")
    if len(records) != 189:
        raise ValueError("expected189jobs")
    results, comparisons, differences = [], [], []
    max_statistics_error = 0.
    for dataset in r.p.DATASETS:
        for method in r.p.METHODS[1:]:
            for endpoint in ENDPOINTS:
                values, independent_values, raw_values, raw_base = [], [], [], []
                for seed in r.p.SEEDS:
                    baseline = records[(dataset, "baseline", seed)]
                    dna = records[(dataset, method, seed)]
                    if baseline["partition_sha256"] != dna["partition_sha256"]:
                        raise ValueError("paired partition mismatch")
                    delta = dna["test"][endpoint]-baseline["test"][endpoint]
                    values.append(delta)
                    independent_values.append(independent[(dataset, method, seed)][endpoint]-independent[(dataset, "baseline", seed)][endpoint])
                    differences.append({"dataset": dataset, "method": method, "seed": seed, "endpoint": endpoint,
                                        "baseline": baseline["test"][endpoint], "dna": dna["test"][endpoint], "delta": delta})
                    previous = json.loads((r.RAW_BN / "rq2_jobs" / dataset / method / f"seed_{seed}.json").read_text())
                    previous_base = json.loads((r.RAW_BN / "rq2_jobs" / dataset / "baseline" / f"seed_{seed}.json").read_text())
                    raw_values.append(previous["test"][endpoint]-previous_base["test"][endpoint])
                    raw_base.append(previous_base["test"][endpoint])
                summary = paired_summary(values)
                check = independent_summary(independent_values)
                for field in check:
                    error = abs(summary[field]-check[field])
                    max_statistics_error = max(max_statistics_error, error)
                    if error > 1e-12:
                        raise ValueError("independent paired statistics mismatch")
                summary.update(dataset=dataset, method=method, endpoint=endpoint,
                               baseline_mean=float(np.mean([records[(dataset, "baseline", seed)]["test"][endpoint] for seed in r.p.SEEDS])),
                               dna_mean=float(np.mean([records[(dataset, method, seed)]["test"][endpoint] for seed in r.p.SEEDS])),
                               verdict=("PASS" if summary["ci_low"] > MARGINS[endpoint] else "NOT_ESTABLISHED") if endpoint in MARGINS else "DESCRIPTIVE")
                results.append(summary)
                old = paired_summary(raw_values)
                comparisons.append({"dataset": dataset, "method": method, "endpoint": endpoint,
                    "raw_bn_mean_delta": old["mean_delta"], "raw_bn_ci": [old["ci_low"], old["ci_high"]],
                    "raw_bn_baseline_mean": float(np.mean(raw_base)),
                    "raw_bn_verdict": ("PASS" if old["ci_low"] > MARGINS[endpoint] else "NOT_ESTABLISHED") if endpoint in MARGINS else "DESCRIPTIVE",
                    "local_bn_mean_delta": summary["mean_delta"]})
    overall = {dataset+"/"+method: "PASS" if all(row["verdict"] == "PASS" for row in results if row["dataset"] == dataset and row["method"] == method and row["endpoint"] in MARGINS) else "NOT_ESTABLISHED" for dataset in r.p.DATASETS for method in r.p.METHODS[1:]}
    r.write(r.OUT / "analysis/paired_summary.json", {"results": results, "overall": overall, "raw_bn_comparison": comparisons})
    r.write(r.OUT / "analysis/independent_recomputation.json", {"status": "PASS", "jobs": 189, "clients": 567,
            "paired_endpoints": 18, "max_metric_error": max_metric_error, "max_statistics_error": max_statistics_error,
            "method": "reload probability arrays; recompute client metrics/means; independent math.fsum sample variance and t.isf CI; checkpoint/payload/round audits"})
    for name, rows in (("client_metrics.csv", client_rows), ("paired_differences.csv", differences)):
        with (r.OUT / "analysis" / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    r.checklist(4, "complete"); r.checklist(5, "in_progress")
    lines = ["# Priority34A — Local-BN utility without BN transmission", "",
        "Status: scientific jobs and independent analysis complete; final supervisor-exit audit pending.", "",
        "New FedBN-style scientific variant, not replacement of original or raw-BN evidence. All189jobs (3datasets ×3methods ×21paired seeds) complete; no failed seed excluded.", "",
        "## Frozen setup", "",
        "Identical P32 prepared data, source rows, partitions and50round Adam(.001), focal(.95,2), batch1024, K3. CPU/thread1; final checkpoint. Seeds321000–321020. No selection or extra replicate. BN affine weights/biases and all buffers remain client-local and persist across rounds. Only non-BN trainable tensors upload/transform/sample-weighted average; payload assertions and per-step BN guards pass in every job.", "",
        "Each client tunes validation-F1 threshold separately and evaluates the common test split with its own BN. Primary unit is the arithmetic mean of3client metrics per seed, not averaged predictions or63independent clients. Individual client metrics/thresholds in analysis/client_metrics.csv; probabilities and checkpoints saved per attempt.", "",
        "## Paired noninferiority", "",
        "Paired mean Student-t two-sided95%CI(df20), strict lower bound>-.02F1 and>-.005ROC-AUC; both required. PR-AUC descriptive. NOT_ESTABLISHED is not proof of inferiority or equivalence. No post-hoc n increase or multiplicity/sign-test family.", "",
        "| Dataset | Method | Endpoint | Baseline mean | DNA mean | Mean delta | SD | Median delta | IQR | 95% CI | NI |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in results:
        lines.append(f"| {row['dataset']} | {row['method']} | {row['endpoint']} | {row['baseline_mean']:.9g} | {row['dna_mean']:.9g} | {row['mean_delta']:+.9g} | {row['sd']:.9g} | {row['median_delta']:+.9g} | {row['iqr']} | [{row['ci_low']:.9g}, {row['ci_high']:.9g}] | {row['verdict']} |")
    lines.extend(["", "Overall: " + json.dumps(overall, sort_keys=True), "", "## P32 raw-BN comparison (descriptive)", "",
        "P32 raw-BN transformed BN affine parameters and transmitted/averaged raw running buffers, with one global-model threshold/test metric. P34A sends no BN and reports a mean of locally thresholded client metrics. These cross-variant differences include training AND evaluation changes and are not isolated transform effects. Each NI claim is against its own baseline.", "",
        "| Dataset | Method | Endpoint | P32 baseline mean | P32 raw-BN delta [95% CI] | P32 NI | P34A delta |",
        "| --- | --- | --- | --- | --- | --- | --- |"])
    for row in comparisons:
        lines.append(f"| {row['dataset']} | {row['method']} | {row['endpoint']} | {row['raw_bn_baseline_mean']:.9g} | {row['raw_bn_mean_delta']:+.9g} {row['raw_bn_ci']} | {row['raw_bn_verdict']} | {row['local_bn_mean_delta']:+.9g} |")
    lines.extend(["", "## Commands, integrity and disclosures", "", "```text",
        f"{sys.executable} -B -m unittest discover -s tests -p test_priority34a_local_bn.py -v",
        f"{sys.executable} -B -u experiments/priority34a_local_bn.py --freeze",
        f"{sys.executable} -B -u experiments/priority34a_local_bn.py --supervise", "```", "",
        "Pre-run protocol: protocols/amendments/2026-10-03_priority34a_local_bn.md. Source/input hashes: execution_freeze.json; all input hashes revalidated. Job/attempt journals, launch receipt, progress logs and failure registry preserve interruptions. Tensor-local transform indices are the non-BN allowlist indices (removing BN changes this domain); P32 seed derivation is unchanged. Local BN is a scientific variant, not numerical clamping. No reconstruction/privacy/DP claim follows merely from no BN transmission. FedBN reference: [Li et al., ICLR2021](https://arxiv.org/abs/2102.07623).", "",
        "Independent probability/metric/paired-CI recomputation, checkpoint BN validity and all150payload assertions per job verified; details in analysis/independent_recomputation.json. Sources, datasets, Latex/, external_defenses/ and all earlier evidence remain unchanged. Final checks/manifest and supervisor-exit receipt are in the new namespace. PROJECT note only after verified completion."])
    REPORT.write_text("\n".join(lines)+"\n")
    compile_dir = r.OUT / "compile"
    compile_dir.mkdir(exist_ok=True)
    code = "import py_compile,sys; from pathlib import Path; [py_compile.compile(p,cfile=str(Path(sys.argv[1])/(Path(p).stem+'.pyc')),doraise=True) for p in sys.argv[2:]]"
    commands = [[sys.executable, "-B", "-m", "unittest", "discover", "-s", "tests", "-p", "test_priority34a_local_bn.py", "-v"],
                [sys.executable, "-B", "-c", code, str(compile_dir), *[str(r.ROOT/path) for path in r.SOURCES if path.endswith(".py")]],
                ["git", "diff", "--check"]]
    checks = []
    for command in commands:
        result = subprocess.run(command, cwd=r.ROOT, capture_output=True, text=True)
        checks.append({"command": command, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr})
        if result.returncode:
            r.write(r.OUT / "final_checks.json", {"status": "FAIL", "checks": checks})
            raise ValueError("final command failed")
    r.write(r.OUT / "final_checks.json", {"status": "PASS", "checks": checks})
    hashes = {**manifest["inputs"], **manifest["sources"]}
    for path in sorted(r.OUT.rglob("*")):
        if path.is_file() and path.name not in {"sha256_manifest.json", "progress.json", "progress.log", "runs.jsonl", "checklist.json", "supervisor.lock", "supervisor.lock.json", "detached_stdout.log", "detached_stderr.log"}:
            hashes[str(path.relative_to(r.ROOT))] = r.p.sha(path)
    hashes[str(REPORT.relative_to(r.ROOT))] = r.p.sha(REPORT)
    r.write(r.OUT / "sha256_manifest.json", hashes)
    r.write(r.OUT / "READY_FOR_FINAL_AUDIT.json", {"at": r.p.now(), "scientific_jobs": 189,
        "independent_statistics": "PASS", "final_commands": "PASS", "requires": "no-live-worker audit after supervisor exit and manifest verification; do not mark complete before that"})
    r.progress("awaiting_final_supervisor_exit_audit", 189)
    r.event("ready_for_final_audit", jobs=189)
