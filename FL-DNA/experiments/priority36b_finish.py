"""Post-supervisor CSV, independent bootstrap, evidence and report seal."""
import csv
import json
import pickle
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import psutil
from experiments.priority36b_memory import OUT,ROOT,DATASETS,METHODS,sha,write,event,verify_freeze,finite,state_hash,bench
from experiments.priority36b_statistics import cell_summary

def save_csv(path,rows):
    with path.open("x",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

def no_live():
    records=[OUT/"supervisor_launch.json",*sorted((OUT/"cells").rglob("*_process.json"))]
    for path in records:
        rec=json.loads(path.read_text())
        try:
            proc=psutil.Process(rec["pid"])
            if proc.create_time()==rec["created"] and proc.is_running() and proc.status()!=psutil.STATUS_ZOMBIE:
                raise RuntimeError("recorded scientific process still alive: "+str(path))
        except psutil.NoSuchProcess: pass
    for proc in psutil.process_iter(["pid","cmdline"]):
        cmd=" ".join(proc.info["cmdline"] or [])
        if "python" in cmd.lower() and "experiments.priority36b_memory" in cmd and ("--phase" in cmd or "--supervise" in cmd):
            raise RuntimeError("scientific workload alive "+str(proc.pid))
    return {"recorded_processes_checked":len(records),"no_live_scientific_workers":True}

def finish():
    verify_freeze()
    assert json.loads((OUT/"supervisor_exit.json").read_text())["returncode"]==0
    live=no_live(); write(OUT/"postexit_check.json",live)
    event("INDEPENDENT_VERIFICATION")
    cells={}; flattened=[]; sample_maxgap=0
    for path in sorted((OUT/"cells").rglob("cell.json")):
        doc=json.loads(path.read_text()); rows=doc["rows"]
        key=(doc["dataset"],doc["profile"],doc["method"],doc["client_count"])
        assert key not in cells and len(rows)==60
        assert [r["repetition"] for r in rows]==list(range(60))
        for row in rows:
            folder=path.parent/("rep_%02d"%row["repetition"])
            assert row==json.loads((folder/"observation.json").read_text())
            assert sha(folder/"bundle.pkl")==row["bundle_sha256"]
            with (folder/"bundle.pkl").open("rb") as f: bundle=pickle.load(f)
            finite(bundle["updates"]+bundle["expected"])
            expected_sha=state_hash(bench._aggregate(bundle["expected"]))
            transfers=json.loads((folder/"transfer.json").read_text())
            assert len(transfers)==key[3] and all(t["correct"] and not t["timed_out"] for t in transfers)
            for phase in ["client","server"]:
                result=json.loads((folder/(phase+"_result.json")).read_text())
                assert result==row[phase] and result["aggregate_sha256"]==expected_sha
                assert result["returncode"]==0 and result["exited"] and result["torch_threads"]==result["torch_interop_threads"]==1
                samples=json.loads((folder/(phase+"_psutil_samples.json")).read_text())
                assert len(samples)==result["psutil_samples"] and max(x[1] for x in samples)==result["psutil_peak_bytes"]
                gap=max((b[0]-a[0] for a,b in zip(samples,samples[1:])),default=0)
                assert gap==result["psutil_max_gap_seconds"]
                sample_maxgap=max(sample_maxgap,gap)
                assert result["rss_peak_bytes"]>=result["rss_before_peak_bytes"]
                assert result["rss_peak_bytes"]==result["ru_maxrss_raw"]*(1 if result["rss_units"]=="bytes" else 1024)
                flattened.append({"dataset":key[0],"profile":key[1],"method":key[2],"client_count":key[3],
                    "repetition":row["repetition"],"sample_kind":row["phase"],"phase":phase,**result})
        cells[key]=rows[10:]
    assert len(cells)==108 and len(flattened)==12960
    save_csv(OUT/"observations.csv",flattened)
    summary=[]
    for (dataset,profile,method,count),rows in sorted(cells.items()):
        raw=cells[(dataset,profile,"RAW_FLOAT32",count)]
        for phase in ["client","server"]:
            for measure in ["RSS","tracemalloc","psutil"]:
                summary.append({"dataset":dataset,"profile":profile,"method":method,"client_count":count,"phase":phase,"measure":measure,
                                **cell_summary(rows,raw,profile,method,count,phase,measure)})
    save_csv(OUT/"memory_summary.csv",summary)
    # Independent CSV reload: floats must round-trip to JSON; separate interpolation
    # recomputes every bootstrap interval. Float64 tolerance fixed pre-measurement.
    csv_obs=list(csv.DictReader((OUT/"observations.csv").open()))
    assert len(csv_obs)==12960
    csv_cells={}
    for r,source in zip(csv_obs,flattened):
        for name,value in source.items():
            assert r[name]==str(value), (name,r[name],value)
        if r["sample_kind"]=="measured":
            key=(r["dataset"],r["profile"],r["method"],int(r["client_count"]))
            csv_cells.setdefault(key,{}).setdefault(int(r["repetition"]),{})[r["phase"]]={k:float(r[k]) for k in ["rss_peak_bytes","trace_peak_bytes","psutil_peak_bytes"]}
    maxdiff=0.; decisions=0
    for r in csv.DictReader((OUT/"memory_summary.csv").open()):
        key=(r["dataset"],r["profile"],r["method"],int(r["client_count"]))
        rawkey=(key[0],key[1],"RAW_FLOAT32",key[3])
        rows=[csv_cells[key][i] for i in range(10,60)]; raw=[csv_cells[rawkey][i] for i in range(10,60)]
        got=cell_summary(rows,raw,key[1],key[2],key[3],r["phase"],r["measure"],independent=True)
        for name,value in got.items():
            if isinstance(value,bool):
                assert r[name]==str(value); decisions+=1
            else:
                diff=abs(float(r[name])-value); maxdiff=max(maxdiff,diff)
                assert np.isclose(float(r[name]),value,rtol=1e-12,atol=1e-12), (key,name,diff)
    audit={**live,"observations":12960,"cells":108,"measured_per_phase_cell":50,"summary_rows":len(summary),
           "independent_bootstrap_max_difference":maxdiff,"unchanged_decision_checks":decisions,"psutil_max_sample_gap_seconds":sample_maxgap,"status":"PASS"}
    write(OUT/"independent_verification.json",audit)
    check=subprocess.run([sys.executable,"-B","-m","pytest","-q","-p","no:cacheprovider","tests/test_priority36b_memory.py"],cwd=ROOT,capture_output=True,text=True)
    write(OUT/"final_tests.json",{"command":check.args,"returncode":check.returncode,"stdout":check.stdout,"stderr":check.stderr})
    assert check.returncode==0
    diff=subprocess.run(["git","diff","--check"],cwd=ROOT,capture_output=True,text=True)
    write(OUT/"final_diff_check.json",{"returncode":diff.returncode,"stdout":diff.stdout,"stderr":diff.stderr}); assert diff.returncode==0
    compile_sources()
    write_report(summary,audit)
    verify_freeze(); no_live()
    checklist=json.loads((OUT/"checklist.json").read_text())
    checklist={k:"DONE" for k in checklist}; write(OUT/"checklist.json",checklist)
    event("COMPLETE",verified=True,cells=108,phase_observations=12960)
    progress=json.loads((OUT/"progress.json").read_text()); progress["complete"]=True; write(OUT/"progress.json",progress)
    excluded={"detached.stdout.log","detached.stderr.log","final_manifest.json","COMPLETE.json"}
    paths=[p for p in OUT.rglob("*") if p.is_file() and p.name not in excluded]
    report=ROOT/"reports/priority36b_report.md"
    manifest={"files":{str(p.relative_to(ROOT)):sha(p) for p in sorted(paths+[report])},
              "excluded":{name:"administrative wrapper log still open; not scientific evidence" for name in ["detached.stdout.log","detached.stderr.log"]}}
    write(OUT/"final_manifest.json",manifest)
    for name,digest in manifest["files"].items(): assert sha(ROOT/name)==digest
    write(OUT/"COMPLETE.json",{"status":"VERIFIED_COMPLETE","report_sha256":sha(report),"manifest_sha256":sha(OUT/"final_manifest.json"),"audit":audit})

def compile_sources():
    import py_compile
    for path in sorted((ROOT/"experiments").glob("priority36b_*.py")):
        py_compile.compile(str(path),cfile=str(OUT/(path.stem+".pyc")),doraise=True)

def write_report(summary,audit):
    protocol=ROOT/"protocols/amendments/2026-10-12_priority36b_rss_supplement.md"
    lines=["# Priority36B: supplementary process memory", "", "Original Bundle B verdicts remain as recorded; this does not replace them.",
        "", "Command: `.venv-phase1/bin/python -B -m experiments.priority36b_memory --launch`. Post-exit wrapper: `python -B -m experiments.priority36b_finish`.",
        "", "Amendment SHA-256: `"+sha(protocol)+"`.","", "MB below means MiB (bytes / 1024²). Each cell has ten host-cache warm-ups and fifty technical repetitions, with fresh client/server processes. 95% percentile bootstrap bounds use 10,000 resamples of the p95 statistic; UB is the 97.5th percentile. These are not independent machines.",
        "", "RSS is process lifetime ru_maxrss including interpreter/import/setup, captured before evidence serialization. On macOS it is bytes. This can mask incremental transform memory. psutil is an externally sampled cross-check (sampling can miss short peaks); maximum observed sampling gap: %.6fs."%audit["psutil_max_sample_gap_seconds"],
        "", "Client peak includes retention/serialization for all K clients, as in the existing harness, not one isolated client's footprint. Network transfer is unchanged and excluded from phase memory. Trace and RSS are collected on the same numeric phase. DP clip100/multiplier.001 is the original cost-only fixture, not a privacy guarantee.",
        "", "Supplementary RSS memory PASS uses upper bounds ≤25% relative overhead and ≤512 MiB client / ≤1024 MiB server **absolute** peak. The original analyzer applied the numeric MB limits to the **increase** instead. Absolute and increase are both reported; no overall Bundle B upgrade follows."]
    for dataset in DATASETS:
        lines += ["", "## "+dataset,"", "|Profile|K|Method|Phase|RSS p95 [UB] MB|RSS Δ p95 MB|RSS relative p95 [UB] %|RSS memory|Trace p95 [UB] MB|Trace Δ p95 MB|Trace relative p95 [UB] %|", "|---|---:|---|---|---:|---:|---:|---|---:|---:|---:|"]
        for r in summary:
            if r["dataset"]!=dataset or r["measure"]!="RSS": continue
            t=next(x for x in summary if all(x[k]==r[k] for k in ["dataset","profile","method","client_count","phase"]) and x["measure"]=="tracemalloc")
            lines.append("|%s|%d|%s|%s|%.3f [%.3f]|%.3f|%.2f [%.2f]|%s|%.3f [%.3f]|%.3f|%.2f [%.2f]|"%(r["profile"],r["client_count"],r["method"],r["phase"],r["absolute_mb_p95"],r["absolute_mb_upper"],r["increase_mb_p95"],100*r["relative_p95"],100*r["relative_upper"],"PASS" if r["memory_pass"] else "FAIL",t["absolute_mb_p95"],t["absolute_mb_upper"],t["increase_mb_p95"],100*t["relative_p95"],100*t["relative_upper"]))
    pairs=[]
    comparison=[]
    old_root=ROOT/"artifacts/priority32_multidataset/repair_trainable_raw_bn_20261002/rq3"
    old_docs={d:json.loads((old_root/d/"analysis/acceptance_decision.json").read_text()) for d in DATASETS}
    for r in summary:
        if r["measure"]=="RSS" and r["method"]!="RAW_FLOAT32":
            t=next(x for x in summary if all(x[k]==r[k] for k in ["dataset","profile","method","client_count","phase"]) and x["measure"]=="tracemalloc")
            pairs.append((r,t))
            old_cells=old_docs[r["dataset"]]["cell_decisions"]
            if r["dataset"]=="paysim" and r["method"]!="DP_CLIP_NOISE":
                old_cells=json.loads((ROOT/"results/rq3/optimized_implementation_20260916/acceptance_decision.json").read_text())["cell_decisions"]
            old=next(x for x in old_cells if all(x[k]==r[k] for k in ["profile","method","client_count"]))
            criteria=[x for x in old["criteria"] if x["criterion"].startswith("peak_"+r["phase"]+"_memory_")]
            assert len(criteria)==2
            comparison.append({k:r[k] for k in ["dataset","profile","method","client_count","phase"]} | {"original_memory_pass":all(x["pass"] for x in criteria),"supplementary_rss_pass":r["memory_pass"],"same_run_trace_pass":t["memory_pass"]})
    rssfail=sum(not r["memory_pass"] for r,t in pairs)
    changed=sum(r["memory_pass"]!=t["memory_pass"] for r,t in pairs)
    old_changed=sum(x["original_memory_pass"]!=x["supplementary_rss_pass"] for x in comparison)
    save_csv(OUT/"original_memory_comparison.csv",comparison)
    lines += ["", "## Reading and limitations","", "%d/%d supplementary method-phase cells fail RSS memory criteria; %d have different memory-only status under same-run RSS versus tracemalloc."%(rssfail,len(pairs),changed),
        "The measurement basis changes from Python-tracked allocations to total process resident memory; therefore the original tracemalloc conclusion is not a total-RSS claim. Original memory and Bundle B decisions stay unchanged. Full intervals (including lower bounds), sampled-RSS values and per-repeat paired values are in `artifacts/priority36b/memory_summary.csv` and `observations.csv`.",
        "%d/%d phase cells differ from the originally recorded shape-matched Priority32 memory-only conclusion; exact per-cell comparison is in original_memory_comparison.csv. This is a supplementary finding, not a retrospective revision."%(old_changed,len(comparison)),
        "", "This supplementary RSS assessment neither re-tests original timing/bandwidth criteria nor supplies new privacy evidence. Fresh-process isolation changes heap warm-up behavior; process startup/import RSS, OS scheduling, memory allocator behavior and sampling resolution limit portability. All observations are retained; no failure exclusions or replay.",
        "", "Independent CSV/coverage/aggregate/source/hash verification: PASS. Bootstrap cross-check maximum absolute difference %.17g; every decision is checked independently. Tests, compilation and git diff --check passed. Final hashes are in artifacts/priority36b/final_manifest.json and COMPLETE.json."%audit["independent_bootstrap_max_difference"]]
    path=ROOT/"reports/priority36b_report.md"
    with path.open("x") as handle: handle.write("\n".join(lines)+"\n")

if __name__=="__main__":
    try: finish()
    except BaseException as exc:
        write(OUT/("VERIFIER_FAILURE_%d.json"%time.time_ns()),{"error":repr(exc)})
        event("REQUIRES_DIRECTION",error=repr(exc),measurement_replayed=False)
        raise
