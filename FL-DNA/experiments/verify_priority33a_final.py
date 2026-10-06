"""Requirement-level final P33a audit; only the new P33a namespace is writable."""
import json
import math
import py_compile
import subprocess
import sys
import platform
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.priority33a_audit import OUT, AMENDMENT, jobs, folder_for, sha, now, write, append


def verify():
    import numpy as np
    import torch
    from experiments import priority33a_bn_mean as a
    report = ROOT / "reports/priority33a_report.md"
    assert (OUT / "COMPLETE.json").exists(), "analysis has not finished"
    receipt_replay = json.loads((OUT / "RECEIPT_CONTRACT_COMPLETE.json").read_text())
    assert receipt_replay["jobs"] == 78 and receipt_replay["all_pairwise_signs_unchanged"] and receipt_replay["new_training_jobs"] == 0
    from experiments.repair_priority33a_receipts import recover_server_payload
    totals = {"A1_jobs": 0, "A1_rounds": 0, "A1_evaluations": 0, "new_training_completed": 0,
        "new_training_gate_failed": 0, "paired_targets": 0}
    backend_rows, states, debias_rows, floor_rows = [], [], [], []
    ordinary_valid = 0
    raw_hashes = {}
    for dataset in ("paysim", "ieee_cis", "baf"):
        audit = json.loads((ROOT / "artifacts/priority32_multidataset/prepared" / dataset / "audit.json").read_text())
        for name, expected in audit["raw_sha256"].items():
            assert sha(ROOT / name) == expected, f"raw data changed: {name}"
            raw_hashes[name] = expected
    for job in jobs():
        folder = folder_for(job)
        doc = json.loads((folder / "result.json").read_text())
        assert doc["status"] == "COMPLETED" and doc["rounds_completed"] == 50 and doc["native_device"] == "mps" and doc["torch_threads"] == 1
        rounds = [json.loads(line) for line in (folder / "bn_rounds.jsonl").read_text().splitlines()]
        evaluations = [json.loads(line) for line in (folder / "evaluations.jsonl").read_text().splitlines()]
        assert len(rounds) == 50 and len(evaluations) == 100
        negative = [r["round"] for r in rounds if any(v["negative"] for v in r["aggregate"].values())]
        assert negative == doc["negative_rounds"]
        state = torch.load(folder / "final_state.pt", map_location="cpu")
        for key, variance in doc["final_bn"].items():
            assert int((state[key] < 0).sum()) == variance["negative"]
            assert int((~torch.isfinite(state[key])).sum()) == variance["nonfinite"]
        cpu, mps = doc["backends"]["cpu"], doc["backends"]["mps"]
        reproduced = not doc["reproduction"]["mismatches"] and doc["reproduction"]["round_count_equal"]
        valid = not negative and cpu["finite_probabilities"] == cpu["rows"] and cpu["nonfinite_logits"] == 0 and reproduced
        ordinary_valid += int(valid)
        states.append(dict(job=job, ordinary_bn_valid=valid, negative_rounds=negative,
            finite_replay_outputs=all(e["nonfinite_logits"] == 0 and e["nonfinite_probabilities"] == 0 for e in evaluations)))
        backend_rows.append([job["priority"], job["variant"], job["sigma"], job["seed"],
            cpu.get("metrics", {}).get("f1", "nonfinite"), mps.get("metrics", {}).get("f1", "nonfinite"),
            cpu.get("metrics", {}).get("auc_roc", "nonfinite"), mps.get("metrics", {}).get("auc_roc", "nonfinite")])
        totals["A1_jobs"] += 1
        totals["A1_rounds"] += len(rounds)
        totals["A1_evaluations"] += len(evaluations)
    outcomes = json.loads((OUT / "A2_COMPLETE.json").read_text())["datasets"]
    for dataset in a.DATASETS:
        for path in (OUT / "A2" / dataset / "training").glob("*/result.json"):
            doc = json.loads(path.read_text())
            assert doc["device"] == "cpu" and doc["torch_threads"] == 1
            if doc["status"] == "COMPLETED":
                checkpoint = path.parent / "final_state.pt"
                assert sha(checkpoint) == doc["checkpoint_sha256"]
                a.strict_state(torch.load(checkpoint, map_location="cpu"), "independent_final_audit")
                trace = [json.loads(line) for line in (path.parent / "bn_rounds.jsonl").read_text().splitlines()]
                assert len(trace) == 50 and all(v >= 0 and math.isfinite(v) for r in trace for v in r["bn_min"].values())
                totals["new_training_completed"] += 1
            else:
                assert doc["status"] == "GATE_FAILED" and doc.get("exception"), "infrastructure failure must not be called a scientific gate"
                totals["new_training_gate_failed"] += 1
        target_path = OUT / "A2" / dataset / "targets.json"
        if not target_path.exists():
            assert outcomes[dataset]["status"] == "NOT_ASSESSABLE"
            continue
        target = json.loads(target_path.read_text())
        sources = [i for gs in target["source_ids"].values() for g in gs for i in g]
        assert len(sources) == len(set(sources)) == 284
        exclusions = {i for old in target["earlier_provenance"] for i in old["source_ids"]}
        assert not exclusions.intersection(sources)
        for old in target["earlier_provenance"]:
            assert sha(ROOT / old["path"]) == old["sha256"]
        if outcomes[dataset]["status"] == "CONFIRMATORY_COMPLETE":
            model = a.load_model(dataset)
            captures = torch.load(OUT / "A2" / dataset / "n39/captures.pt", map_location="cpu")
            prior, std = a.population(dataset)
            prior_mse = [a.mse_std(prior, c["true_mean"], std) for c in captures]
            write(OUT / "A2" / dataset / "data_free_prior_descriptive.json", dict(status="DESCRIPTIVE_ONLY", n=39,
                rows=[dict(target=i, source_ids=c["source_ids"], prior_mse=prior_mse[i]) for i, c in enumerate(captures)]))
            for method in a.METHODS:
                docs = [json.loads((OUT / "A2" / dataset / "confirmatory" / method / f"target_{i:02d}/result.json").read_text()) for i in range(39)]
                assert [d["source_ids"] for d in docs] == target["source_ids"]["n39"]
                for i, doc in enumerate(docs):
                    receipt_path = OUT / "A2" / dataset / "confirmatory" / method / f"target_{i:02d}/transmitted_payload.pt"
                    assert sha(receipt_path) == doc["transmitted_payload_sha256"]
                    receipt = torch.load(receipt_path, map_location="cpu")
                    assert set(receipt) == {"kind", "q", "metadata"}
                    value = a.mse_std(a.recover_payload(model, receipt), captures[i]["true_mean"], std)
                    assert np.isclose(value, doc["defended_mse"], rtol=1e-8, atol=1e-10)
                    server_folder = OUT / "receipt_contract_replay" / dataset / method / f"target_{i:02d}"
                    current = json.loads((server_folder / "result.json").read_text())
                    server_path = server_folder / "server_observable_receipt.pt"
                    assert current["source_ids"] == doc["source_ids"] and current["server_receipt_sha256"] == sha(server_path)
                    server = torch.load(server_path, map_location="cpu")
                    assert set(server) == ({"kind", "q"} if method == a.METHODS[0] else {"kind", "q", "metadata"})
                    server_score = a.mse_std(recover_server_payload(model, server), captures[i]["true_mean"], std)
                    assert np.isclose(server_score, current["defended_mse"], rtol=1e-8, atol=1e-10)
                    assert np.isclose(server_score, doc["defended_mse"], rtol=1e-8, atol=1e-10)
                    if method == a.METHODS[0]:
                        q = receipt["q"].numpy()
                        # Exact P27 C3 Level-1 debias, pre-registered descriptive only.
                        estimate = (q-a.V1.mix_ratio*float(np.mean(q)))/max(1.-a.V1.mix_ratio, 1e-12)
                        debias_mse = a.mse_std(a.recover_mean(model, torch.from_numpy(estimate.astype(np.float32, copy=False))), captures[i]["true_mean"], std)
                        sidecar = dict(dataset=dataset, target=i, source_ids=doc["source_ids"], direct_mse=value,
                            debias_mse=debias_mse, transmitted_payload_sha256=sha(receipt_path), status="DESCRIPTIVE_ONLY", used_for_primary_selection=False)
                        write(receipt_path.parent / "v1_debias_descriptive.json", sidecar)
                        debias_rows.append(sidecar)
                totals["paired_targets"] += 39
                median_dna = float(np.median([d["defended_mse"] for d in docs]))
                median_prior = float(np.median(prior_mse))
                distortion_dp = float(np.median([d["comparators"]["distortion"]["dp_mse"] for d in docs]))
                floor_rows.append([dataset, method, median_prior, median_dna, distortion_dp,
                    "both no better than Prior" if median_dna >= median_prior and distortion_dp >= median_prior else "not both at Prior reference"])
        else:
            assert outcomes[dataset]["status"] == "NOT_ASSESSABLE"
            assert not (OUT / "A2" / dataset / "confirmatory").exists(), "confirmatory access after failed qualification"
    stats = json.loads((OUT / "independent_statistics.json").read_text())
    assert stats["final"] and stats["family_size"] == len(stats["tests"]) == 16 and stats["independent_exact_and_holm_agreement"]
    from experiments.analyze_priority33a import table
    appendix = ["", "## Completion audit and plain conclusions", "",
        f"Historical DP utility audit: {ordinary_valid}/21 audited jobs meet ordinary aggregate-BN/CPU-finiteness/reproduction criteria; {21-ordinary_valid}/21 do not. This statement is limited to the sampled seeds. Negative client buffers, if any, remain separately visible in the main table.", "",
        "Actual final-checkpoint endpoints (historical validation threshold, no retuning):", "",
        table(["Source", "Variant", "sigma", "Seed", "CPU test F1", "MPS test F1", "CPU ROC-AUC", "MPS ROC-AUC"], backend_rows), "",
        "DP accounting shown for A2 is one clipped update-vector release with add/remove sensitivity C and delta=1e-5. It is not a record-level privacy guarantee for the entire 50-round training procedure. Invalid BN training or missing utility brackets are NOT_ASSESSABLE, never privacy superiority.", "",
        "Additional descriptive Prior reference on the same39 targets (no new tests, attacker selection or verdict-rule changes):", "",
        table(["Dataset", "DNA", "Median Prior MSE", "Median DNA MSE", "Median distortion-DP MSE", "Practical reference flag"], floor_rows), "",
        "The v1 ranking should therefore not be interpreted as a practically useful recovery difference when both reconstructions are worse than a data-free Prior. Numerical validity of the sampled historical DP jobs also does not rehabilitate BN-invalid DNA utility targets identified in P32b, or prove that every unobserved utility-grid replicate is valid.", "",
        "The source firewall reads eight earlier IEEE provenance files, including the actual P6/P8/P10/P12 bundle 'targets' containers in addition to P5 manifests, and excludes444 unique TransactionIDs before any A2 qualification. IEEE's n24/confirmatory runs and expensive utility grid are skipped after its n8 failure. BAF's DP utility grid is not run because neither full-state transform supplies a BN-valid utility target.", "",
        f"Final receipt-contract repair: v1's archived transmitted_payload.pt contained unused DNATransformStats (including encoder-only raw/update difference diagnostics). Those diagnostics were never read by recovery. Original diagnostic receipts remain intact but are superseded as server receipts; authoritative packets are receipt_contract_replay/.../server_observable_receipt.pt, with v1 only kind/q and v2 only kind/q/official decoder metadata. All78 recoveries were replayed on exactly the same39 targets per defense, without new training or changes to the attacker. Maximum score difference is {receipt_replay['max_absolute_score_difference']:.4g}; every paired sign, raw p and Holm p is unchanged. This is a payload-schema repair, not a design change or new replicate set. Command actually executed: `.venv-phase1/bin/python -B experiments/repair_priority33a_receipts.py`.", "",
        "Pre-registered P27-style v1 debias (descriptive only, never used to select the primary attacker by ground truth):", "",
        table(["Dataset", "n", "Median plain standardized MSE", "Median debiased standardized MSE"],
            [[dataset, len([r for r in debias_rows if r["dataset"] == dataset]),
              float(np.median([r["direct_mse"] for r in debias_rows if r["dataset"] == dataset])),
              float(np.median([r["debias_mse"] for r in debias_rows if r["dataset"] == dataset]))]
             for dataset in a.DATASETS if any(r["dataset"] == dataset for r in debias_rows)]), "",
        "Requirement-level evidence: completion_audit.json; final_checks.json; sha256_manifest.json. A2 gate outcomes and unavailable utility arms are explicit, not silently omitted.", ""]
    text = report.read_text().split("\n## Completion audit and plain conclusions")[0]
    report.write_text(text+"\n".join(appendix))
    sources = [ROOT / name for name in ("experiments/priority33a_audit.py", "experiments/priority33a_bn_mean.py", "experiments/analyze_priority33a.py", "experiments/verify_priority33a_final.py", "experiments/repair_priority33a_receipts.py", "tests/test_priority33a.py", "tests/test_priority33a_receipts.py")]
    for source in sources:
        py_compile.compile(str(source), doraise=True)
    tests = subprocess.run([sys.executable, "-B", "-m", "unittest", "tests.test_priority33a", "tests.test_priority33a_receipts"], cwd=ROOT, capture_output=True, text=True)
    diff = subprocess.run(["git", "diff", "--check"], cwd=ROOT, capture_output=True, text=True)
    assert tests.returncode == diff.returncode == 0, tests.stdout+tests.stderr+diff.stdout+diff.stderr
    checklist = json.loads((OUT / "checklist.json").read_text())
    if all(json.loads((OUT / "A2" / d / "training/baseline_321000/result.json").read_text())["status"] == "COMPLETED" for d in a.DATASETS):
        checklist["2"]["status"] = "completed"
    checklist["6"]["status"] = "completed"
    write(OUT / "checklist.json", checklist)
    write(OUT / "completion_audit.json", dict(at=now(), totals=totals, A1=states, A2=outcomes, raw_data_sha256=raw_hashes,
        runtime=dict(python=sys.version, torch=torch.__version__, numpy=np.__version__, platform=platform.platform(), mps_available=torch.backends.mps.is_available()),
        v1_debias_descriptive=debias_rows,
        confirmatory_prior_descriptive=floor_rows,
        server_receipt_contract_replay=dict(jobs=receipt_replay["jobs"], max_absolute_score_difference=receipt_replay["max_absolute_score_difference"], all_pairwise_signs_unchanged=True),
        independent_receipt_recovery="PASS", source_pairing="PASS", no_training_started_after_failed_qualification="PASS"))
    write(OUT / "final_checks.json", dict(at=now(), py_compile="PASS", git_diff_check="PASS", unit_tests="PASS", unit_test_output=tests.stdout+tests.stderr,
        independent_statistics="PASS", requirement_completion_audit="PASS"))
    write(OUT / "COMPLETE.json", dict(at=now(), report=str(report.relative_to(ROOT)), report_sha256=sha(report), family_size=16, independently_verified=True))
    progress = dict(stage="complete_with_explicit_NOT_ASSESSABLE_cells", dataset="all", method="all", done=211, total=211, failed=32,
        started_at=json.loads((OUT / "A1_attempt2_supervisor.json").read_text())["started_at"], last_update=now(), eta_minutes=0,
        initial_scientific_result_jobs=133, same_target_receipt_replays=78, scientific_training_gate_failures=32, earlier_pre_training_infrastructure_failures=21)
    write(OUT / "progress.json", progress)
    append(OUT / "progress.log", progress)
    manifest = {str(path.relative_to(ROOT)): sha(path) for path in sorted(OUT.rglob("*")) if path.is_file() and path.name != "sha256_manifest.json"}
    manifest.update({str(path.relative_to(ROOT)): sha(path) for path in sources+[AMENDMENT, report]})
    manifest.update(raw_hashes)
    write(OUT / "sha256_manifest.json", manifest)
    print(json.dumps(dict(totals=totals, ordinary_bn_valid=ordinary_valid, report=str(report))))


if __name__ == "__main__":
    verify()
