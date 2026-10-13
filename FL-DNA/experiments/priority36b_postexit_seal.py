"""Additive read-only completion audit; never changes original evidence."""
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import psutil

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"artifacts/priority36b"
def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""): h.update(block)
    return h.hexdigest()

def main():
    complete=json.loads((OUT/"COMPLETE.json").read_text())
    assert complete["status"]=="VERIFIED_COMPLETE"
    assert digest(ROOT/"reports/priority36b_report.md")==complete["report_sha256"]
    assert digest(OUT/"final_manifest.json")==complete["manifest_sha256"]
    manifest=json.loads((OUT/"final_manifest.json").read_text())
    for name,sha in manifest["files"].items(): assert digest(ROOT/name)==sha,name
    freeze=json.loads((OUT/"freeze.json").read_text())
    for section in ["sources","inputs"]:
        for name,sha in freeze[section].items(): assert digest(ROOT/name)==sha,name
    live=[p.pid for p in psutil.process_iter(["cmdline"]) if any(a=="experiments.priority36b_memory" or a=="experiments.priority36b_finish" for a in p.info["cmdline"] or [])]
    assert not live,live
    observations=list(csv.DictReader((OUT/"observations.csv").open()))
    assert len(observations)==12960
    coverage=Counter(tuple(r[k] for k in ["dataset","profile","method","client_count","phase","sample_kind"]) for r in observations)
    assert len(coverage)==432
    assert all(v==(50 if key[-1]=="measured" else 10) for key,v in coverage.items())
    summaries=list(csv.DictReader((OUT/"memory_summary.csv").open()))
    assert len(summaries)==648
    for row in summaries:
        absolute=float(row["absolute_mb_upper"])<=float(row["absolute_limit_mb"])
        relative=float(row["relative_upper"])<=.25
        assert row["absolute_pass"]==str(absolute)
        assert row["relative_pass"]==str(relative)
        assert row["memory_pass"]==str(absolute and relative)
    comparisons=list(csv.DictReader((OUT/"original_memory_comparison.csv").open()))
    assert len(comparisons)==162
    for row in comparisons:
        keys=["dataset","profile","method","client_count","phase"]
        for measure,field in [("RSS","supplementary_rss_pass"),("tracemalloc","same_run_trace_pass")]:
            source=next(r for r in summaries if all(r[k]==row[k] for k in keys) and r["measure"]==measure)
            assert row[field]==source["memory_pass"]
        original=ROOT/"artifacts/priority32_multidataset/repair_trainable_raw_bn_20261002/rq3"/row["dataset"]/"analysis/acceptance_decision.json"
        if row["dataset"]=="paysim" and row["method"]!="DP_CLIP_NOISE":
            original=ROOT/"results/rq3/optimized_implementation_20260916/acceptance_decision.json"
        cells=json.loads(original.read_text())["cell_decisions"]
        cell=next(c for c in cells if c["profile"]==row["profile"] and c["method"]==row["method"] and c["client_count"]==int(row["client_count"]))
        criteria=[c for c in cell["criteria"] if c["criterion"].startswith("peak_"+row["phase"]+"_memory_")]
        assert len(criteria)==2 and row["original_memory_pass"]==str(all(c["pass"] for c in criteria))
    post=json.loads((OUT/"independent_verification.json").read_text()); assert post["status"]=="PASS"
    assert json.loads((OUT/"final_tests.json").read_text())["returncode"]==0
    assert json.loads((OUT/"final_diff_check.json").read_text())["returncode"]==0
    project=ROOT.parent/"PROJECT.md"; before=OUT/"PROJECT_pre_completion_snapshot.md"
    assert project.read_bytes().startswith(before.read_bytes())
    additions=[OUT/"detached.stdout.log",OUT/"detached.stderr.log",before,project,Path(__file__),ROOT/"protocols/amendments/2026-10-12_priority36b_postexit_administrative_seal.md"]
    seal={"at_utc":datetime.now(timezone.utc).isoformat(),"status":"POSTEXIT_VERIFIED_COMPLETE","original_report_sha256":complete["report_sha256"],"original_manifest_sha256":complete["manifest_sha256"],"original_manifest_entries_verified":len(manifest["files"]),"freeze_entries_verified":sum(len(freeze[s]) for s in ["sources","inputs"]),"observations":len(observations),"summary_rows":len(summaries),"comparison_rows":len(comparisons),"no_live_scientific_or_wrapper_process":True,"measurement_replayed":False,"supplementary_hashes":{str(p):digest(p) for p in additions}}
    with (OUT/"POSTEXIT_ADMINISTRATIVE_SEAL.json").open("x") as f: json.dump(seal,f,indent=2,sort_keys=True)
    print(json.dumps(seal))

if __name__=="__main__": main()
