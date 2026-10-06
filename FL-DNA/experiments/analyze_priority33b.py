"""P33b fail-closed report/statistics; never calls an inversion/training loop."""
from __future__ import annotations
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.priority33b_core import OUT,ROOT,sha,write,now
import numpy as np
from scipy.stats import binomtest


def integer_tail(wins,losses):
    n=wins+losses
    return sum(math.comb(n,k) for k in range(wins,n+1))/2**n if n else 1.


def holm(values):
    order=np.argsort(values,kind="stable")
    corrected=np.minimum(1,np.maximum.accumulate(np.array(values)[order]*np.arange(len(values),0,-1)))
    output=np.empty(len(values))
    output[order]=corrected
    return output.tolist()


def rows(dataset,arm):
    paths=[OUT/dataset/"confirmatory"/arm/f"target_{i:03d}/result.json" for i in range(39)]
    if not all(path.exists() for path in paths):return None
    docs=[json.loads(path.read_text()) for path in paths]
    if not all(doc["status"]=="COMPLETED" for doc in docs):return None
    for path,doc in zip(paths,docs):
        for name,digest in doc["files"].items():assert sha(path.parent/name)==digest
    return docs


def analyze():
    if not (OUT/"GATES_COMPLETE.json").exists() or not (OUT/"CELLS_COMPLETE.json").exists():
        raise RuntimeError("qualification/calibration/cells not resolved; no final analysis with pending jobs")
    gates=json.loads((OUT/"GATES_COMPLETE.json").read_text())["outcomes"]
    checks=[]
    for dataset,gate in gates.items():
        for stage in ("n8","n24"):
            path=OUT/dataset/stage/"qualification.json"
            if not path.exists():continue
            doc=json.loads(path.read_text())
            for control,test in doc["tests"].items():
                target_docs=[json.loads((OUT/dataset/stage/f"target_{i:03d}/result.json").read_text()) for i in range(doc["n"])]
                deltas=[r["accuracy"]["accuracy_percent"]-r["baselines"][control] for r in target_docs if r["status"]=="COMPLETED"]
                w,l=sum(d>0 for d in deltas),sum(d<0 for d in deltas)
                assert w==test["wins"] and l==test["losses"] and len(deltas)-w-l==test["ties"]
                p=integer_tail(w,l)
                scipy_p=float(binomtest(test["wins"],test["wins"]+test["losses"],.5,alternative="greater").pvalue) if test["wins"]+test["losses"] else 1.
                assert abs(p-test["p"])<1e-14 and abs(p-scipy_p)<1e-14
                checks.append(dict(dataset=dataset,stage=stage,comparator=control,p=p,scipy_p=scipy_p))
    tests=[]
    for dataset in ("ieee_cis","baf"):
        for method in ("dna_v1_conservative","dna_v2_0p95"):
            for comparator in ("distortion","utility"):
                defense=rows(dataset,method) if gates[dataset]["passed"] else None
                dp=rows(dataset,f"dp_{comparator}_{method}") if gates[dataset]["passed"] else None
                difference=None
                if defense is not None and dp is not None:
                    assert [r["source_ids"] for r in defense]==[r["source_ids"] for r in dp]
                    assert [r["attack_seed"] for r in defense]==[r["attack_seed"] for r in dp]
                    a=np.array([r["accuracy"]["accuracy_percent"] for r in defense])
                    b=np.array([r["accuracy"]["accuracy_percent"] for r in dp])
                    difference=a-b
                    assert np.isfinite(difference).all()
                for direction in ("DNA_stronger_protection","DP_stronger_protection"):
                    test=dict(dataset=dataset,method=method,comparator=comparator,direction=direction,family_size=16)
                    if difference is None:
                        test.update(status="NOT_ASSESSABLE",reason="qualification failed" if not gates[dataset]["passed"] else "required comparator/attack gate failed",p=1.,wins=None,losses=None,ties=None)
                    else:
                        sign=-1 if direction.startswith("DNA") else 1
                        w,l=int(np.sum(sign*difference>0)),int(np.sum(sign*difference<0))
                        p=integer_tail(w,l)
                        independently=float(binomtest(w,w+l,.5,alternative="greater").pvalue) if w+l else 1.
                        assert abs(p-independently)<1e-14
                        sorted_d=np.sort(difference)
                        test.update(status="ASSESSABLE",wins=w,losses=l,ties=39-w-l,p=p,scipy_p=independently,
                            median_dna_accuracy=float(np.median(a)),median_dp_accuracy=float(np.median(b)),
                            median_paired_dna_minus_dp=float(np.median(difference)),order_interval_ranks13_27=[float(sorted_d[12]),float(sorted_d[26])],paired_differences=difference.tolist())
                    tests.append(test)
    adjusted=holm([t["p"] for t in tests])
    # Independent step-down implementation using plain Python sorting/loops.
    independent=[None]*16
    previous=0.
    for rank,i in enumerate(sorted(range(16),key=lambda i:(tests[i]["p"],i))):
        previous=max(previous,min(1.,(16-rank)*tests[i]["p"]))
        independent[i]=previous
    assert np.allclose(adjusted,independent,atol=1e-14,rtol=0)
    for test,p in zip(tests,adjusted):test["holm_p"]=p
    statistics=dict(at=now(),family_size=16,tests=tests,qualification_independent=checks,agreement=True)
    write(OUT/"independent_statistics.json",statistics)
    return statistics,gates


def report(statistics,gates):
    lines=["# Priority33b: tabular individual-record recovery", "", "Single-gradient FedSGD, batch8, known labels, official TabLeak1500iterations ×30member ensemble. This differs from P32 multi-step local Adam FL. Accuracy is logical mixed-feature accuracy after permutation alignment, not BN batch-mean recovery.", "", "## Qualification", "", "| Dataset | Stage | Control | Attack median (%) | Control median (%) | Paired median difference | Wins/losses/ties | Exact p | Gate |", "|---|---|---|---|---|---|---|---|---|"]
    for dataset,gate in gates.items():
        for stage in ("n8","n24"):
            path=OUT/dataset/stage/"qualification.json"
            if not path.exists():continue
            doc=json.loads(path.read_text())
            valid=[json.loads((path.parent/f"target_{i:03d}/result.json").read_text()) for i in range(doc["n"])]
            valid=[r for r in valid if r["status"]=="COMPLETED"]
            for control,t in doc["tests"].items():
                attack_median=float(np.median([r["accuracy"]["accuracy_percent"] for r in valid])) if valid else None
                control_median=float(np.median([r["baselines"][control] for r in valid])) if valid else None
                lines.append(f'| {dataset} | {stage} | {control} | {attack_median} | {control_median} | {t["median_difference"]} | {t["wins"]}/{t["losses"]}/{t["ties"]} | {t["p"]:.12g} | {"PASS" if doc["passed"] else "NOT_ASSESSABLE"} |')
    lines += ["", "Qualification failure is NOT evidence of privacy protection. No tuning on confirmatory targets, no scope/budget reduction to force a passing instrument.", "", "## Representation and feature selection", ""]
    for dataset in gates:
        doc=json.loads((OUT/dataset/"adapter.json").read_text())
        lines.append(f'{dataset}: {doc["original_encoded_dimension"]} original → {doc["selected_encoded_dimension"]} retained encoded coordinates, {doc["logical_feature_count"]} logical features. Train-only additional standardization/bounds; continuous values never integer-rounded. Complete categorical blocks retained; frequency-encoded categories scored as continuous surrogates, not original categories.')
        lines += ["", "Retained coordinates: "+", ".join(doc["encoded_features"]), ""]
    lines += ["## Paired confirmatory tests", "", "Accuracy in percent; lower means stronger protection. Paired effect = DNA−DP accuracy. Intervals use ranks13/27 of39. Fixed16-test Holm family includes p1 placeholders for gated absent cells.", "", "| Dataset | Transform | Comparator | Direction | DNA median | DP median | Paired median [13/27] | W/L/T | Raw p | Holm p | Status |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for t in statistics["tests"]:
        if t["status"]=="NOT_ASSESSABLE":
            lines.append(f'| {t["dataset"]} | {t["method"]} | {t["comparator"]} | {t["direction"]} | NA | NA | NA | NA | 1 (reserved) | {t["holm_p"]:.12g} | NOT_ASSESSABLE |')
        else:
            interval=t["order_interval_ranks13_27"]
            lines.append(f'| {t["dataset"]} | {t["method"]} | {t["comparator"]} | {t["direction"]} | {t["median_dna_accuracy"]:.6f} | {t["median_dp_accuracy"]:.6f} | {t["median_paired_dna_minus_dp"]:.6f} [{interval[0]:.6f}, {interval[1]:.6f}] | {t["wins"]}/{t["losses"]}/{t["ties"]} | {t["p"]:.12g} | {t["holm_p"]:.12g} | ASSESSABLE |')
    lines += ["", "## Calibration and execution status", ""]
    if not any(g["passed"] for g in gates.values()):
        lines += ["Both datasets failed n8 qualification. n24, distortion calibration, utility training/grid and n39 reconstruction were NOT RUN under the frozen gate rule. The39source-ID batches were reserved for pairing/firewall assertions only; no confirmatory records/gradients were inspected. Later-stage implementations were not empirically exercised, and no claim is made for those paths.", ""]
    timing=json.loads((OUT/"GATES_COMPLETE.json").read_text())["elapsed_seconds"]
    lines += [f"Corrected qualification:16/16 completed, zero numerical failures, wall time {timing:.3f}s ({timing/60:.2f}min). First infrastructure-only launch:16errors before any optimization result, retained separately. n24/calibration/confirmatory runtime:0 when gated off.", ""]
    for dataset in gates:
        for name in ("distortion_calibration.json","utility_calibration.json"):
            path=OUT/dataset/name
            if path.exists():lines += [f"### {dataset}: {name}", "", "```json", path.read_text().strip(), "```", ""]
    lines += ["## Commands, disclosures and integrity", "", "- `.venv-phase1/bin/python -B -m unittest tests.test_priority33b -v` (6 pre-run synthetic tests).", "- `.venv-phase1/bin/python -B experiments/priority33b_qualify.py --prepare`.", "- Detached `nohup .../.venv-phase1/bin/python -B -u experiments/priority33b_qualify.py --supervise`; exact worker commands in runs.jsonl.", "- First preparation failed before any scientific job: trusted historical pickle namespace collision. Reused P33a source-ID extraction after verifying each original bundle SHA.", "- Initial16qualification jobs failed at first native closure with callback keyword TypeError, before optimizer updates or metrics. Technical replay amendment fixes only argument names; all original errors kept, root gates superseded as infrastructure-only. attempt2 uses byte-identical target manifests and unchanged seeds/budgets.", "- All new workloads CPU, one torch thread per process,4workers; nonfinite loss/logits/candidates/outputs fail closed, never clamp numerical failures. Official feature projection/bounds are frozen attack operations.", "- Individual records scored up to permutation; this does not establish per-record DP. DP accounting, when available, is one whole-update add/remove release at delta1e-5.", "- Utility uses disclosed FC/selected-representation model variant; no BN-invalid historical utility or repaired raw-BN variant silently substituted. Validation only,50rounds/16pairedseeds/P32 F1 rule and bracketed grid; test not used for selection.", "- Qualification and confirmatory source IDs are fresh/disjoint; IEEE TransactionID and BAF original-row domains remain explicit.", "- Independent exact integer tails agree with SciPy and two Holm implementations.", "", "SHA256 manifests and final_checks.json are the completion evidence; pending or gate-failed cells never count as privacy successes.", ""]
    lines=[line.replace("Utility uses disclosed", "PLANNED, NOT RUN: utility uses disclosed").replace("Qualification and confirmatory source IDs", "Qualification and reserved confirmatory source IDs") for line in lines]
    lines += ["Actual post-gate commands: `.venv-phase1/bin/python -B -u experiments/priority33b_run_cells.py --supervise` (zero jobs by gating), `.venv-phase1/bin/python -B experiments/analyze_priority33b.py`, and `.venv-phase1/bin/python -B experiments/verify_priority33b.py`. Comparator synthetic tests: `tests.test_priority33b_comparators` (3tests); final verification runs9tests total, py_compile and git diff --check. No confirmatory p-values were observed: p1 values are fixed-family reservations only.", ""]
    lines += ["## SHA-256", "", "Full output/code checksums: `artifacts/priority33b/sha256_manifest.json`. Original and corrected execution freezes remain separate; the corrected freeze is `artifacts/priority33b/attempt2/execution_freeze.json`.", "", "| Input/code | SHA-256 |", "|---|---|"]
    hash_inputs={str(p.relative_to(ROOT)):sha(p) for p in ROOT.glob("protocols/amendments/*priority33b*.md")}
    for dataset in gates:
        adapter=json.loads((OUT/dataset/"adapter.json").read_text())
        hash_inputs.update(adapter["raw_sha256"])
        hash_inputs[str((OUT/dataset/"adapter.json").relative_to(ROOT))]=sha(OUT/dataset/"adapter.json")
        hash_inputs[str((OUT/dataset/"targets.json").relative_to(ROOT))]=sha(OUT/dataset/"targets.json")
    for p in ROOT.glob("experiments/*priority33b*.py"):
        hash_inputs[str(p.relative_to(ROOT))]=sha(p)
    for name,digest in sorted(hash_inputs.items()):lines.append(f"| {name} | `{digest}` |")
    lines.append("")
    path=ROOT/"reports/priority33b_report.md"
    path.write_text("\n".join(lines))
    return path


if __name__=="__main__":
    statistics,gates=analyze()
    path=report(statistics,gates)
    print(json.dumps(dict(report=str(path),sha256=sha(path),family_size=16)))
