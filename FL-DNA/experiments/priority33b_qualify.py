"""Detached resumable P33b qualification; never touches confirmatory records."""
import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.priority33b_core import *


def checklist(i,status):
    path=OUT/"checklist.json"
    doc=json.loads(path.read_text())
    doc[str(i)]["status"]=status
    doc[str(i)]["last_update"]=now()
    write(path,doc)
    write(OUT.parent/"checklist.json",doc)


def exclusions(dataset):
    excluded,provenance=set(),[]
    if dataset=="ieee_cis":
        for path in sorted((ROOT/"artifacts/priority5_ieee_cis").glob("**/target_manifest.json")):
            doc=json.loads(path.read_text())
            ids=[int(i) for group in doc["groups"] for i in group["transaction_ids"]]
            excluded.update(ids)
            provenance.append(dict(path=str(path.relative_to(ROOT)),sha256=sha(path),count=len(ids)))
        # P33a parsed these trusted pickles before official TabLeak claimed the
        # 'attacks' namespace. Reuse its exact source-ID extraction, verifying
        # every original bundle byte hash instead of executing pickle imports.
        audited=json.loads((ROOT/"artifacts/priority33a/A2/ieee_cis/targets.json").read_text())["earlier_provenance"]
        for path in sorted((ROOT/"artifacts").glob("**/ieee_priority6_bundle.pt")):
            receipt=next(p for p in audited if p["path"]==str(path.relative_to(ROOT)))
            assert sha(path)==receipt["sha256"]
            ids=[int(i) for i in receipt["source_ids"]]
            excluded.update(ids)
            provenance.append(dict(path=str(path.relative_to(ROOT)),sha256=sha(path),count=len(ids)))
    path=ROOT/"artifacts/priority33a/A2"/dataset/"targets.json"
    doc=json.loads(path.read_text())
    ids=[int(i) for groups in doc["source_ids"].values() for group in groups for i in group]
    excluded.update(ids)
    provenance.append(dict(path=str(path.relative_to(ROOT)),sha256=sha(path),count=len(ids)))
    return excluded,provenance


def targets(dataset):
    source=np.load(PREPARED/dataset/"train_source_ids.npy")
    old,provenance=exclusions(dataset)
    available=np.flatnonzero(~np.isin(source,list(old)))
    rng=np.random.default_rng(333320 if dataset=="ieee_cis" else 333321)
    indices=available[rng.permutation(len(available))[:568]].reshape(71,8)
    groups=dict(n8=indices[:8].tolist(),n24=indices[8:32].tolist(),n39=indices[32:].tolist())
    ids={stage:[[int(source[i]) for i in batch] for batch in batches] for stage,batches in groups.items()}
    flat=[i for groups_i in ids.values() for batch in groups_i for i in batch]
    assert len(flat)==len(set(flat))==568 and not old.intersection(flat)
    doc=dict(dataset=dataset,split="train",groups=groups,source_ids=ids,earlier_provenance=provenance,
        excluded_count=len(old),source_sha256=sha(PREPARED/dataset/"train_source_ids.npy"))
    path=OUT/dataset/"targets.json"
    if path.exists():
        assert json.loads(path.read_text())==doc
    else:
        write(path,doc)
    return doc


def freeze():
    prerequisite=ROOT/"artifacts/priority33a/COMPLETE.json"
    p=json.loads(prerequisite.read_text())
    assert p["independently_verified"] and sha(ROOT/p["report"])==p["report_sha256"]
    assert json.loads((ROOT/"artifacts/priority33a/final_checks.json").read_text())
    docs={dataset:prepare(dataset) for dataset in ("ieee_cis","baf")}
    for dataset in docs:
        targets(dataset)
        assert sha(OUT/dataset/"targets.json")==sha(OUT.parent/dataset/"targets.json"), "technical replay changed targets"
    source_paths=[PROTOCOL,ROOT/"protocols/amendments/2026-10-02_priority33b_keyword_compatibility_replay.md",Path(__file__).resolve(),ROOT/"experiments/priority33b_core.py",ROOT/"tests/test_priority33b.py",
        ROOT/"experiments/priority29/tabular_native_positive_control.py",ROOT/"experiments/priority30_native_defenses/run_audit.py",
        ROOT/"experiments/priority30_native_defenses/native_adapters.py",ROOT/"dna_encoder/transform_defense.py",ROOT/"dna_encoder/transform_defense_v2.py"]
    source_paths+=list((ROOT/"external_defenses/tableak").glob("**/*.py"))
    hashes={str(path.relative_to(ROOT)):sha(path) for path in source_paths}
    doc=dict(protocol_sha256=sha(PROTOCOL),source_sha256=hashes,official_config=official_config(),
        adapters={dataset:sha(OUT/dataset/"adapter.json") for dataset in docs},
        target_manifests={dataset:sha(OUT/dataset/"targets.json") for dataset in docs},prerequisite_sha256=sha(prerequisite))
    path=OUT/"execution_freeze.json"
    if path.exists():
        assert json.loads(path.read_text())==doc
    else:
        write(path,doc)
    checklist(0,"completed")
    checklist(1,"completed")
    return doc


def job(dataset,stage,index):
    folder=OUT/dataset/stage/f"target_{index:03d}"
    folder.mkdir(parents=True,exist_ok=True)
    start=time.time()
    config=dict(dataset=dataset,stage=stage,index=index,method="unprotected",protocol_sha256=sha(PROTOCOL),
        core_sha256=sha(ROOT/"experiments/priority33b_core.py"),adapter_sha256=sha(OUT/dataset/"adapter.json"))
    try:
        data=PreparedDataset(dataset)
        manifest=json.loads((OUT/dataset/"targets.json").read_text())
        indices=manifest["groups"][stage][index]
        x,y=data.Xtrain[indices].clone(),data.ytrain[indices].clone()
        net=make_model(data.num_features)
        loss=torch.nn.CrossEntropyLoss()(net(x),y)
        payload=[g.detach() for g in torch.autograd.grad(loss,net.parameters())]
        for g in payload:finite(g,"unprotected transmitted gradient")
        torch.save(dict(gradient=payload,labels=y,model=net.state_dict()),folder/"server_receipt.pt")
        torch.save(dict(truth=x,labels=y,indices=indices,source_ids=manifest["source_ids"][stage][index]),folder/"scoring_truth.pt")
        offset=0 if stage=="n8" else 100
        domain_offset=0 if dataset=="ieee_cis" else 1000
        seed=333400+domain_offset+offset+index
        rec,ensemble,losses=native_recover(net,data,payload,y,seed)
        score=measure(data,x,rec)
        baseline,mm,emp=controls(data,x,333500+domain_offset+offset+index)
        torch.save(dict(reconstruction=rec.detach(),ensemble=ensemble,objectives=losses,mean_mode=mm,
            empirical_single=emp),folder/"reconstruction.pt")
        result=dict(status="COMPLETED",config=config,accuracy=score,baselines=baseline,attack_seed=seed,
            indices=indices,source_ids=manifest["source_ids"][stage][index],objective_losses=losses,
            files={p.name:sha(p) for p in folder.glob("*.pt")})
    except Exception:
        result=dict(status="GATE_FAILED",config=config,error=traceback.format_exc())
    result.update(elapsed_seconds=time.time()-start,finished_at=now(),torch_threads=torch.get_num_threads(),device="cpu")
    write(folder/"result.json",result)


def execute(dataset,stage,index):
    folder=OUT/dataset/stage/f"target_{index:03d}"
    path=folder/"result.json"
    if path.exists():
        doc=json.loads(path.read_text())
        assert doc["config"]["protocol_sha256"]==sha(PROTOCOL)
        assert doc["config"]["core_sha256"]==sha(ROOT/"experiments/priority33b_core.py")
        if doc["status"]=="COMPLETED":
            for file,digest in doc["files"].items():assert sha(folder/file)==digest
        return doc
    folder.mkdir(parents=True,exist_ok=True)
    command=[sys.executable,"-B","-u",str(Path(__file__).resolve()),"--job",dataset,stage,str(index)]
    append(OUT/"runs.jsonl",dict(at=now(),event="qualification_job_started",dataset=dataset,stage=stage,index=index,command=command))
    with (folder/"stdout.log").open("a") as so,(folder/"stderr.log").open("a") as se:
        proc=subprocess.run(command,cwd=ROOT,stdout=so,stderr=se)
    if not path.exists():
        raise RuntimeError(f"worker exited {proc.returncode} without result: {dataset}/{stage}/{index}")
    return json.loads(path.read_text())


def supervise():
    lock=OUT/"supervisor.lock.json"
    if lock.exists():
        previous=json.loads(lock.read_text())
        try:os.kill(previous["pid"],0)
        except ProcessLookupError:pass
        else:raise RuntimeError("existing live supervisor; refuse duplicate")
    write(lock,dict(pid=os.getpid(),started_at=now()))
    freeze()
    checklist(2,"in_progress")
    outcomes={}
    start=time.time()
    for stage,n in (("n8",8),("n24",24)):
        jobs=[(dataset,stage,i) for dataset in ("ieee_cis","baf") if stage=="n8" or outcomes[dataset]["passed"] for i in range(n)]
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures={pool.submit(execute,*task):task for task in jobs}
            done=0
            for future in as_completed(futures):
                dataset,_,index=futures[future]
                result=future.result()
                done+=1
                append(OUT/"runs.jsonl",dict(at=now(),event="qualification_job_finished",dataset=dataset,stage=stage,index=index,
                    status=result["status"],result_sha256=sha(OUT/dataset/stage/f"target_{index:03d}/result.json")))
                doc=dict(stage=stage,dataset=dataset,method="unprotected",done=done,total=len(jobs),
                    failed=sum(json.loads(path.read_text())["status"]!="COMPLETED" for path in OUT.glob(f"*/{stage}/target_*/result.json")),
                    last_update=now(),eta_minutes=(time.time()-start)/done*(len(jobs)-done)/60)
                write(OUT/"progress.json",doc)
                append(OUT/"progress.log",doc)
                write(OUT.parent/"progress.json",dict(doc,attempt="attempt2"))
                append(OUT.parent/"progress.log",dict(doc,attempt="attempt2"))
        for dataset in ("ieee_cis","baf"):
            if stage=="n24" and not outcomes[dataset]["passed"]:continue
            rows=[json.loads((OUT/dataset/stage/f"target_{i:03d}/result.json").read_text()) for i in range(n)]
            tests={}
            failed=[i for i,r in enumerate(rows) if r["status"]!="COMPLETED"]
            for comparator in ("mean_mode","empirical_mean"):
                differences=[r["accuracy"]["accuracy_percent"]-r["baselines"][comparator] for r in rows if r["status"]=="COMPLETED"]
                w,l=sum(d>0 for d in differences),sum(d<0 for d in differences)
                tests[comparator]=dict(wins=w,losses=l,ties=len(differences)-w-l,p=exact_p(w,l),median_difference=float(np.median(differences)) if differences else None)
            outcome=dict(dataset=dataset,stage=stage,n=n,failed_jobs=failed,tests=tests,
                passed=not failed and all(t["p"]<.05 for t in tests.values()))
            write(OUT/dataset/stage/"qualification.json",outcome)
            outcomes[dataset]=outcome
    write(OUT/"GATES_COMPLETE.json",dict(at=now(),outcomes=outcomes,next_stage="calibration_for_passing_datasets",elapsed_seconds=time.time()-start))
    checklist(2,"completed" if all(o["passed"] for o in outcomes.values()) else "resolved_with_NOT_ASSESSABLE")
    write(OUT/"progress.json",dict(stage="qualification_resolved",dataset="all",method="unprotected",done=sum(8+(24 if (OUT/d/"n24/qualification.json").exists() else 0) for d in outcomes),
        total=sum(8+(24 if (OUT/d/"n24/qualification.json").exists() else 0) for d in outcomes),failed=sum(len(o["failed_jobs"]) for o in outcomes.values()),last_update=now(),eta_minutes=0,outcomes=outcomes))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--prepare",action="store_true")
    parser.add_argument("--supervise",action="store_true")
    parser.add_argument("--job",nargs=3)
    args=parser.parse_args()
    if args.job:job(args.job[0],args.job[1],int(args.job[2]))
    elif args.prepare:freeze()
    elif args.supervise:supervise()
    else:parser.error("choose --prepare, --supervise or --job")
