"""Independent P33a exact-binomial/Holm recomputation, report and final integrity checks."""
import argparse
import json
import math
import os
import py_compile
import subprocess
import time
from pathlib import Path

from priority33a_audit import ROOT, OUT, AMENDMENT, jobs, folder_for, sha, now, write, append, CORE

REPORT = ROOT / "reports/priority33a_report.md"
METHODS = ("dna_v1_conservative", "dna_v2_0p95")
DATASETS = ("ieee_cis", "baf")


def exact(wins, losses):
    n = wins+losses
    return sum(math.comb(n, k) for k in range(wins, n+1))/(2**n) if n else 1.


def holm(values):
    order = sorted(range(len(values)), key=lambda i: (values[i], i))
    adjusted = [None]*len(values)
    previous = 0.
    for rank, index in enumerate(order):
        previous = min(1., max(previous, values[index]*(len(values)-rank)))
        adjusted[index] = previous
    return adjusted


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"]*len(headers)) + " |"] +
        ["| " + " | ".join(str(v) for v in row) + " |" for row in rows])


def analyze(final=False):
    import numpy as np
    audit = []
    for job in jobs():
        folder = folder_for(job)
        path = folder / "result.json"
        if not path.exists():
            continue
        doc = json.loads(path.read_text())
        assert doc["job"] == job
        audit.append(doc)
        if doc["status"] == "COMPLETED":
            original = json.loads((ROOT / job["stored"]).read_text())["rounds"]
            replay = json.loads((folder / "metrics.json").read_text())["rounds"]
            errors = []
            mismatches = 0
            assert len(original) == len(replay) == 50
            for a, b in zip(original, replay):
                for key in CORE:
                    if a[key] is None or b[key] is None:
                        mismatches += int(a[key] != b[key])
                    else:
                        delta = abs(a[key]-b[key])
                        errors.append(delta)
                        mismatches += int(delta > 1e-8 or (key in {"tn", "fp", "fn", "tp"} and delta != 0))
            assert mismatches == len(doc["reproduction"]["mismatches"])
            assert max(errors, default=0.) == doc["reproduction"]["max_absolute_error"]
            for device in ("cpu", "mps"):
                values = np.load(folder / f"final_{device}_probabilities.npy")
                assert int(np.isfinite(values).sum()) == doc["backends"][device]["finite_probabilities"]
    summary_path = OUT / "A2_COMPLETE.json"
    a2 = json.loads(summary_path.read_text())["datasets"] if summary_path.exists() else {}
    tests = []
    for dataset in DATASETS:
        for method in METHODS:
            paths = sorted((OUT / "A2" / dataset / "confirmatory" / method).glob("target_*/result.json"))
            docs = [json.loads(f.read_text()) for f in paths]
            if docs:
                expected = json.loads((OUT / "A2" / dataset / "targets.json").read_text())["source_ids"]["n39"]
                assert [d["source_ids"] for d in docs] == expected[:len(docs)]
            for comparator in ("distortion", "utility"):
                valid = len(docs) == 39 and all(d["comparators"][comparator]["status"] == "VALID" for d in docs)
                for direction in ("DNA_better", "DP_better"):
                    row = dict(dataset=dataset, method=method, comparator=comparator, direction=direction, status="VALID" if valid else "NOT_ASSESSABLE", p_raw=1.)
                    if valid:
                        dna = np.asarray([d["defended_mse"] for d in docs])
                        dp = np.asarray([d["comparators"][comparator]["dp_mse"] for d in docs])
                        diff = dna-dp
                        signed = diff if direction == "DNA_better" else -diff
                        wins, losses = int((signed > 0).sum()), int((signed < 0).sum())
                        effects = np.sort(diff)
                        row.update(wins=wins, losses=losses, ties=int((signed == 0).sum()), p_raw=exact(wins, losses),
                            median_dna_mse=float(np.median(dna)), median_dp_mse=float(np.median(dp)), median_paired_difference=float(np.median(diff)),
                            interval_rank13_27=[float(effects[12]), float(effects[26])], n=39)
                        # Independently verify SciPy's result without reusing driver sign implementation.
                        from scipy.stats import binomtest
                        scipy_p = float(binomtest(wins, wins+losses, .5, alternative="greater").pvalue) if wins+losses else 1.
                        assert abs(row["p_raw"]-scipy_p) < 1e-12
                    else:
                        row["reason"] = a2.get(dataset, {}).get("reason", "Required gate/calibration/results unavailable")
                        utility = a2.get(dataset, {}).get("utility", {}).get(method, {})
                        if comparator == "utility" and utility.get("reason"):
                            row["reason"] = utility["reason"]
                    tests.append(row)
    assert len(tests) == 16
    adjusted = holm([r["p_raw"] for r in tests])
    for row, value in zip(tests, adjusted):
        row["p_holm"] = value
        row["significant"] = row["status"] == "VALID" and value < .05
    # Alternative closed implementation verifies all Holm values.
    order = np.argsort([r["p_raw"] for r in tests], kind="stable")
    ordered = np.asarray([tests[i]["p_raw"] for i in order])
    alternate = np.minimum(1., np.maximum.accumulate(ordered*np.arange(16, 0, -1)))
    assert np.allclose(alternate, np.asarray(adjusted)[order], rtol=0, atol=1e-15)
    gates = []
    for dataset in DATASETS:
        for stage in ("n8", "n24"):
            path = OUT / "A2" / dataset / stage / "qualification.json"
            if path.exists():
                doc = json.loads(path.read_text())
                for name in ("prior", "decoy"):
                    differences = np.asarray([r[name+"_mse"]-r["recovered_mse"] for r in doc["rows"]])
                    w, l = int((differences > 0).sum()), int((differences < 0).sum())
                    assert abs(exact(w, l)-doc["tests"][name]["p"]) < 1e-12
                    gates.append([dataset, stage, name, f"{w}/{l}/{len(differences)-w-l}", exact(w,l),
                        float(np.median([r["recovered_mse"] for r in doc["rows"]])), float(np.median([r[name+"_mse"] for r in doc["rows"]])), doc["passed"]])
    if final:
        assert len(audit) == 21 and all(d["status"] == "COMPLETED" for d in audit), "A1 unfinished/error; no final claims"
        assert summary_path.exists(), "A2 not finished"
        for freeze_name in ("A1_attempt2_execution_freeze.json", "A2_execution_freeze.json"):
            freeze = json.loads((OUT / freeze_name).read_text())
            if "source_sha256" in freeze:
                for path, value in {**freeze["source_sha256"], **freeze["stored_sha256"]}.items():
                    assert sha(ROOT / path) == value, f"immutable source/output drift: {path}"
            else:
                assert freeze["runner_sha256"] == sha(ROOT / "experiments/priority33a_bn_mean.py")
                assert freeze["p32_runner_sha256"] == sha(ROOT / "experiments/priority32_multidataset.py")
                for name, value in freeze["dependency_sha256"].items():
                    assert sha(ROOT / name) == value
                for dataset, files in freeze["prepared"].items():
                    for name, value in files.items():
                        assert sha(ROOT / "artifacts/priority32_multidataset/prepared" / dataset / name) == value
    write(OUT / "independent_statistics.json", dict(final=final, at=now(), family_size=16, tests=tests, independent_exact_and_holm_agreement=True))
    rows = []
    for d in audit:
        job = d["job"]
        if d["status"] != "COMPLETED":
            rows.append([job["priority"], job["variant"], job["sigma"], job["seed"], "ERROR", "—", "—", "—", "—", "—"])
            continue
        reproduction = d["reproduction"]
        matches = reproduction["round_count_equal"] and not reproduction["mismatches"]
        cpu = d["backends"]["cpu"]
        mps = d["backends"]["mps"]
        ordinary_valid = not d["negative_rounds"] and cpu["nonfinite_logits"] == 0 and cpu["finite_probabilities"] == cpu["rows"] and matches
        rows.append([job["priority"], job["variant"], job["sigma"], job["seed"],
            d["negative_rounds"][0] if d["negative_rounds"] else "none", d["client_negative_rounds"][0] if d["client_negative_rounds"] else "none",
            sum(v["negative"] for v in d["final_bn"].values()), f"{cpu['finite_probabilities']}/{cpu['rows']}", f"{mps['finite_probabilities']}/{mps['rows']}",
            f"metrics {'match' if matches else 'MISMATCH'}; ordinary BN {'VALID' if ordinary_valid else 'INVALID'}"])
    stats_rows = []
    for r in tests:
        stats_rows.append([r["dataset"], r["method"], r["comparator"], r["direction"], r["status"],
            f"{r.get('wins','—')}/{r.get('losses','—')}/{r.get('ties','—')}", r["p_raw"], r["p_holm"],
            r.get("median_dna_mse", "—"), r.get("median_dp_mse", "—"), r.get("median_paired_difference", "—"), r.get("interval_rank13_27", "—")])
    lines = ["# Priority33a — historical DP BN validity and batch-mean recovery", "", f"Status: {'FINAL' if final else 'IN PROGRESS'}; generated {now()}.", "",
        "Protocol saved before any run: protocols/amendments/2026-10-02_priority33a_bn_audit_mean_recovery.md. Earlier artifacts, Latex/, external_defenses/ and datasets are unchanged.", "",
        "## A1 historical DP utility validity", "", f"Native replay jobs finished: {len(audit)}/21. Exact original entry points/seeds/settings on MPS; torch threads=1. CPU/MPS evaluation uses the actual final test inputs and unchanged final checkpoint, not a substitute model.", "",
        table(["Source", "Variant", "sigma", "Seed", "First aggregate negative round", "First client negative round", "Final negative BN coordinates", "CPU finite probabilities", "MPS finite probabilities", "Validity/reproduction"], rows), "",
        "Ordinary BN validity requires no aggregate negative variance, finite CPU logits/probabilities, and reproduction within 1e-8 (counts exact). Client-state violations are separately reported. A reproduced MPS utility number alone is not a valid-domain guarantee. Intermediate failures are not hidden by a finite final endpoint. Only three seeds per selected cell are audited; no extrapolation to unobserved runs.", "",
        "P25 uses seeds 2501101–2501103. P25b uses the first three NEW seeds 2501201–2501203, not duplicate P25 jobs. P27 fedbn_trainable_only sigma=1e-6 is the historical selected but NOT_BRACKETED fallback control; it does not noise BN buffers and must not be presented as an established utility match.", "",
        "## A2 qualification", "", "All new training is CPU/thread1; actual forward logits and local/transmitted/aggregate/final states fail closed on negative BN variance or non-finite values. No clamps, excluded seeds, post-gate feature reduction or raw-BN substitution.", "",
        "P32 did not save checkpoints, so baseline seed321000 is replayed per dataset and checked against its stored validation/test metrics before saving a fixed final checkpoint. IEEE-CIS retains 476 features and a 128-row first linear layer; BAF retains 58 features. P24 plain least-squares recovery and train-only standardized MSE are unchanged; v2 uses key-known sketch-space least squares, not transpose lift.", "",
        table(["Dataset", "Gate", "Reference", "W/L/T", "p", "Median recovered MSE", "Median reference MSE", "Gate passes"], gates), "",
        "Fresh dataset-qualified source IDs are disjoint across n8/n24/n39 and excluded from earlier IEEE Priority5 reconstruction target manifests. The protected quantity is a four-record batch mean, not individual-record recovery.", "",
        "## A2 paired confirmatory tests", "", "Fixed Holm family: 16 tests (two datasets × two transforms × two DP comparators × two directions). Gated absent cells reserve p=1 and remain NOT_ASSESSABLE. Higher MSE indicates stronger protection. Intervals are paired DNA−DP differences at ranks13/27 of39, not differences of marginal medians. Qualification tests are outside this family.", "",
        table(["Dataset", "DNA", "DP arm", "Direction", "Status", "W/L/T", "Raw p", "Holm p", "Median DNA MSE", "Median DP MSE", "Paired median Δ", "Ranks13/27"], stats_rows), "",
        "## Calibration and gates", ""]
    for dataset in DATASETS:
        for name in ("distortion_calibration.json", "utility_calibration.json"):
            path = OUT / "A2" / dataset / name
            if path.exists():
                lines += [f"### {dataset} {name}", "", "```json", path.read_text().strip(), "```", ""]
        if dataset in a2:
            lines += [f"{dataset}: {a2[dataset]['status']}; {a2[dataset].get('reason', '')}", ""]
    lines += ["Utility grid uses validation F1 only, 16 paired seeds, full-state CPU transforms, frozen eight sigma points and at most one extension. A BN-invalid transform cannot supply a utility target: that arm is NOT_ASSESSABLE, not evidence of privacy. Failed required transform replicates prevent unnecessary baseline/DP calibration jobs; the failure records remain intact.", "",
        "## Commands, disclosures and hashes", "", "Commands: `.venv-phase1/bin/python -B -u experiments/priority33a_audit.py --supervise`; `.venv-phase1/bin/python -B -u experiments/priority33a_bn_mean.py --supervise`; `.venv-phase1/bin/python -B experiments/analyze_priority33a.py --final`. Detached jobs use nohup/start_new_session; exact per-job argv is in artifacts/priority33a/runs.jsonl.", "",
        "Infrastructure disclosures: initial shell-background launch did not persist and started zero jobs. The first detached launch resolved the virtualenv symlink to the base interpreter, so all21 jobs failed to import torch before any scientific round. Those artifacts are preserved under A1/. The corrected unchanged scientific configuration runs under A1_attempt2/ using the virtualenv path without symlink resolution. A2's waiting-only supervisor was stopped before any training to complete pre-run harness validation; it is relaunched with the finalized runner. No scientific results were discarded or tuned.", "",
        f"Amendment SHA256: `{sha(AMENDMENT)}`. Per-source/original-output hashes: A1_attempt2_execution_freeze.json; prepared data/source hashes: A2_execution_freeze.json; final manifest: sha256_manifest.json. Per-target receipts and result.json are stored alongside each confirmatory target. Synthetic pre-run adapter/gate unit tests:6/6 pass.", "",
        "Independent exact sign tests (integer binomial tails) agree with SciPy; independent NumPy Holm construction agrees within1e-15. Final compilation/diff results are in final_checks.json when all stages finish.", ""]
    REPORT.write_text("\n".join(lines))
    if final:
        sources = [ROOT / p for p in ("experiments/priority33a_audit.py", "experiments/priority33a_bn_mean.py", "experiments/analyze_priority33a.py", "tests/test_priority33a.py")]
        for path in sources:
            py_compile.compile(str(path), doraise=True)
        diff = subprocess.run(["git", "diff", "--check"], cwd=ROOT, capture_output=True, text=True)
        assert diff.returncode == 0, diff.stdout+diff.stderr
        write(OUT / "final_checks.json", dict(py_compile="PASS", git_diff_check="PASS", independent_statistics="PASS", at=now()))
        hashes = {str(p.relative_to(ROOT)): sha(p) for p in sorted(OUT.rglob("*")) if p.is_file() and p.name != "sha256_manifest.json"}
        hashes.update({str(p.relative_to(ROOT)): sha(p) for p in sources+[AMENDMENT, REPORT]})
        write(OUT / "sha256_manifest.json", hashes)
        write(OUT / "COMPLETE.json", dict(at=now(), report=str(REPORT.relative_to(ROOT)), report_sha256=sha(REPORT), family_size=16))
    return len(audit), a2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    if args.watch:
        while not (OUT / "A2_COMPLETE.json").exists():
            analyze(False)
            time.sleep(60)
        analyze(True)
    else:
        analyze(args.final)
