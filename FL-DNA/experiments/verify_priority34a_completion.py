"""Administrative post-exit audit; never trains or changes frozen sources."""
from __future__ import annotations
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import priority34a_local_bn as r


def live_workers():
    output = subprocess.check_output(["ps", "-axo", "pid=,ppid=,command="], text=True)
    rows = []
    for line in output.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) != 3 or int(parts[0]) == os.getpid():
            continue
        pid, parent, command = parts
        executable = command.split()[0] if command else ""
        if "python" not in Path(executable).name.lower():
            continue
        if ("experiments/priority34a_local_bn.py" in command and "--supervise" in command) or "spawn_main" in command:
            rows.append({"pid": int(pid), "ppid": int(parent), "command": command})
    return rows


def main():
    if not (r.OUT / "READY_FOR_FINAL_AUDIT.json").exists():
        raise RuntimeError("analysis not ready; no completion audit yet")
    workers = live_workers()
    if workers:
        raise RuntimeError("live workers/supervisor; do not complete: " + json.dumps(workers))
    frozen = r.verify_freeze()
    hashes = json.loads((r.OUT / "sha256_manifest.json").read_text())
    for relative, digest in hashes.items():
        if r.p.sha(ROOT / relative) != digest:
            raise RuntimeError("manifest mismatch " + relative)
    checks = json.loads((r.OUT / "final_checks.json").read_text())
    independent = json.loads((r.OUT / "analysis/independent_recomputation.json").read_text())
    if checks["status"] != "PASS" or independent["status"] != "PASS" or independent["jobs"] != 189 or independent["clients"] != 567 or independent["paired_endpoints"] != 18:
        raise RuntimeError("checks or independent statistics incomplete")
    failures = list((r.OUT / "jobs").rglob("*.failure.json"))
    if failures or (r.OUT / "REQUIRES_DIRECTION.json").exists():
        raise RuntimeError("failed jobs require direction")
    records = {}
    assertions = 0
    probability_replays = 0
    for job in frozen["jobs"]:
        path = r.OUT / job["result"]
        if not r.valid_result(path, job["config"]):
            raise RuntimeError("missing result")
        doc = json.loads(path.read_text())
        records[(doc["dataset"], doc["method"], doc["seed"])] = doc
        assertions += doc["payload_assertions"]
        prepared = r.p.OUT / "prepared" / doc["dataset"]
        dimension = r.np.load(prepared / "train_x.npy", mmap_mode="r").shape[1]
        checkpoint_files = [name for name in doc["artifact_sha256"] if name.endswith("final_checkpoint.pt")]
        if len(checkpoint_files) != 1:
            raise RuntimeError("checkpoint provenance incomplete")
        checkpoint = r.torch.load(r.OUT / checkpoint_files[0], map_location="cpu")
        if len(checkpoint["client_bn"]) != 3:
            raise RuntimeError("checkpoint client count")
        for client in range(3):
            model = r.p.FraudMLP(dimension).cpu()
            bn, allowed = r.domains(model)
            r.assert_payload(checkpoint["global_non_bn"], bn, allowed)
            if set(checkpoint["client_bn"][client]) != bn:
                raise RuntimeError("checkpoint BN domain")
            state = model.state_dict()
            state.update(checkpoint["global_non_bn"])
            state.update(checkpoint["client_bn"][client])
            model.load_state_dict(state)
            r.guard(model, "final checkpoint probability replay")
            for split in ("validation", "test"):
                features = r.np.load(prepared / (split+"_x.npy"), mmap_mode="r")
                probability_files = [name for name in doc["artifact_sha256"] if name.endswith(f"client{client}_{split}_probabilities.npy")]
                if len(probability_files) != 1:
                    raise RuntimeError("probability provenance incomplete")
                expected = r.np.load(r.OUT / probability_files[0], allow_pickle=False)
                actual = r.checked_probabilities(model, features)
                if not r.np.array_equal(actual, expected):
                    raise RuntimeError(f"checkpoint probability mismatch {job['result']} client{client}/{split}")
                probability_replays += 1
        if len(records) % 21 == 0:
            print(json.dumps({"event": "checkpoint_probability_audit", "jobs": len(records),
                              "client_split_replays": probability_replays}), flush=True)
    if len(records) != 189 or assertions != 28350 or probability_replays != 1134:
        raise RuntimeError("job/firewall counts incomplete")
    summary = json.loads((r.OUT / "analysis/paired_summary.json").read_text())
    from scipy.stats import t
    max_error = 0.
    for row in summary["results"]:
        values = []
        for seed in r.p.SEEDS:
            dna = records[(row["dataset"], row["method"], seed)]
            baseline = records[(row["dataset"], "baseline", seed)]
            dna_mean = math.fsum(client["test"][row["endpoint"]] for client in dna["clients"])/3
            base_mean = math.fsum(client["test"][row["endpoint"]] for client in baseline["clients"])/3
            values.append(dna_mean-base_mean)
        mean = math.fsum(values)/21
        sd = math.sqrt(math.fsum((value-mean)**2 for value in values)/20)
        half = float(t.ppf(.975, 20))*sd/math.sqrt(21)
        for key, value in (("mean_delta", mean), ("sd", sd), ("ci_low", mean-half), ("ci_high", mean+half)):
            error = abs(value-row[key]); max_error = max(max_error, error)
            if error > 1e-12:
                raise RuntimeError("final independent CI mismatch")
        if row["endpoint"] in {"f1", "auc_roc"}:
            margin = -.02 if row["endpoint"] == "f1" else -.005
            if row["verdict"] != ("PASS" if mean-half > margin else "NOT_ESTABLISHED"):
                raise RuntimeError("NI verdict mismatch")
    command = [sys.executable, "-B", "-c", "import py_compile,sys; py_compile.compile(sys.argv[1],cfile=sys.argv[2],doraise=True)",
               str(Path(__file__).resolve()), str(r.OUT / "compile/verify_priority34a_completion.pyc")]
    compile_result = subprocess.run(command, capture_output=True, text=True)
    diff = subprocess.run(["git", "diff", "--check"], cwd=ROOT, capture_output=True, text=True)
    if compile_result.returncode or diff.returncode:
        raise RuntimeError("final compile/diff check failed")
    receipt = {"status": "PASS", "verified_at": r.p.now(), "jobs": 189, "paired_replicates": 21,
               "client_evaluations": 567, "upload_bn_absence_assertions": assertions,
               "checkpoint_probability_replays_bit_exact": probability_replays,
               "manifest_files_verified": len(hashes), "frozen_sources_verified": len(frozen["sources"]),
               "independent_statistics": independent, "final_independent_CI_max_error": max_error,
               "live_workers": workers, "py_compile": "PASS", "git_diff_check": "PASS",
               "verifier_sha256": r.p.sha(Path(__file__).resolve())}
    r.write(r.OUT / "final_verified_receipt.json", receipt)
    addendum = ROOT / "reports/priority34a_completion_addendum.md"
    if addendum.exists():
        raise RuntimeError("completion addendum already exists; do not overwrite")
    lines = ["# Priority34A — verified completion addendum", "", "Status: COMPLETE after supervisor exit and final independent audit.", "",
             "The pre-exit scientific report is preserved byte-identically in archive/priority34a_report_pre_final_audit.md; priority34a_report.md receives only completion wording and administrative audit details, with all scientific tables unchanged.", "",
             "189/189 jobs and567client evaluations; zero failures, no extra/replaced seed. All28350upload assertions confirm no BN tensor in any simulated transmission. All1134client/split probability arrays reproduce bit-exactly from final global-non-BN plus client-local-BN checkpoints. Original and raw-BN P32 evidence remain unchanged. Source/data/result/checkpoint/probability/round manifest verified; independent18endpoint NI recomputation and compile/diff checks PASS, no live supervisor/worker.", "",
             "Research audit files contain local BN checkpoints/minima for verification; they are not simulated transmitted payloads. Keeping BN local removes that explicit transmission channel but does not establish a general privacy guarantee.", "",
             "## Per-client test metrics (mean over21seeds; individual seed/client values in client_metrics.csv)", "",
             "| Dataset | Method | Client | Mean F1 | Mean ROC-AUC | Mean PR-AUC |",
             "| --- | --- | --- | --- | --- | --- |"]
    for dataset in r.p.DATASETS:
        for method in r.p.METHODS:
            for client in range(3):
                means = {endpoint: math.fsum(records[(dataset, method, seed)]["clients"][client]["test"][endpoint] for seed in r.p.SEEDS)/21 for endpoint in ("f1", "auc_roc", "pr_auc")}
                lines.append(f"| {dataset} | {method} | {client} | {means['f1']:.9g} | {means['auc_roc']:.9g} | {means['pr_auc']:.9g} |")
    lines.extend(["", "Overall NI: " + json.dumps(summary["overall"], sort_keys=True), "",
                  "Commands, paired endpoint effects/CIs, comparison caveats and input provenance remain in the scientific report. Final verified receipt: artifacts/priority34a/final_verified_receipt.json; final manifest includes this addendum and administrative verifier."])
    addendum.write_text("\n".join(lines)+"\n")
    # Preserve the pre-exit report before making the requested report self-contained.
    report = ROOT / "reports/priority34a_report.md"
    archive = ROOT / "reports/archive/priority34a_report_pre_final_audit.md"
    if archive.exists():
        raise RuntimeError("report archive already exists; never overwrite")
    archive.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(report, archive)
    old_digest = hashes[str(report.relative_to(ROOT))]
    if r.p.sha(archive) != old_digest:
        raise RuntimeError("archived report checksum mismatch")
    pending_status = "Status: scientific jobs and independent analysis complete; final supervisor-exit audit pending."
    old_text = report.read_text()
    if old_text.count(pending_status) != 1:
        raise RuntimeError("unexpected report lifecycle wording")
    final_text = old_text.replace(pending_status,
        "Status: COMPLETE — all189jobs, independent statistics, checkpoint replay, final commands and no-live-worker audit PASS.", 1)
    client_heading = "## Per-client test metrics (mean over21seeds; individual seed/client values in client_metrics.csv)"
    final_text += "\n## Final administrative verification\n\n"
    final_text += "The pre-exit report is preserved byte-identically in archive/priority34a_report_pre_final_audit.md. Completion addendum: priority34a_completion_addendum.md. Original scientific-manifest report checksum resolves to that archived snapshot; report_finalization.json records this administrative relocation. No scientific table, margin, seed or result changed.\n\n"
    final_text += "\n".join(lines[lines.index(client_heading):]) + "\n"
    environment = json.loads((r.OUT / "environment_audit.json").read_text())
    final_text += "\n## Runtime provenance\n\n"
    final_text += f"{environment['cpu']}; {environment['platform']}; CPU/thread1. Package versions: {json.dumps(environment['versions'], sort_keys=True)}. Supplementary runtime capture timestamp: {environment['captured_at']}; this did not change the pre-run execution freeze.\n"
    report.write_text(final_text)
    r.write(r.OUT / "report_finalization.json", {
        "at": r.p.now(), "kind": "administrative_only_no_science_replay",
        "original_scientific_manifest_report_path": str(report.relative_to(ROOT)),
        "original_report_sha256": old_digest, "preserved_at": str(archive.relative_to(ROOT)),
        "archived_report_sha256": r.p.sha(archive), "final_report_sha256": r.p.sha(report),
        "amendment_sha256": r.p.sha(ROOT / "protocols/amendments/2026-10-03_priority34a_report_finalization.md")})
    r.checklist(5, "complete")
    r.progress("complete", 189)
    r.event("final_audit_pass", jobs=189, assertions=assertions)
    r.write(r.OUT / "COMPLETE.json", receipt)
    # Complete manifest snapshot excludes itself; no old manifest/report overwritten.
    final_hashes = dict(hashes)
    for path in r.OUT.rglob("*"):
        if path.is_file() and path.name != "sha256_manifest_final.json":
            final_hashes[str(path.relative_to(ROOT))] = r.p.sha(path)
    final_hashes[str(addendum.relative_to(ROOT))] = r.p.sha(addendum)
    final_hashes[str(report.relative_to(ROOT))] = r.p.sha(report)
    final_hashes[str(archive.relative_to(ROOT))] = r.p.sha(archive)
    administrative_amendment = ROOT / "protocols/amendments/2026-10-03_priority34a_report_finalization.md"
    final_hashes[str(administrative_amendment.relative_to(ROOT))] = r.p.sha(administrative_amendment)
    final_hashes[str(Path(__file__).resolve().relative_to(ROOT))] = r.p.sha(Path(__file__).resolve())
    r.write(r.OUT / "sha256_manifest_final.json", final_hashes)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
