"""P33b development utility grid and paired passing-cell reconstructions."""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.priority33b_comparators import *
from experiments.priority33b_qualify import checklist


def job_path(config):
    if config["kind"]=="utility":
        name=f'{config["method"]}_seed{config["seed"]}'
        if config["method"]=="dp":name+=f'_sigma{config["sigma"]:.9g}'
        return OUT/config["dataset"]/"utility"/name
    return OUT/config["dataset"]/"confirmatory"/config["arm"]/f'target_{config["index"]:03d}'


def reconstruction_job(config,folder):
    started=time.time()
    try:
        dataset=config["dataset"]
        data=PreparedDataset(dataset)
        manifest=json.loads((OUT/dataset/"targets.json").read_text())
        indices=manifest["groups"]["n39"][config["index"]]
        x,y=data.Xtrain[indices].clone(),data.ytrain[indices].clone()
        net=make_model(data.num_features)
        raw=[g.detach() for g in torch.autograd.grad(torch.nn.CrossEntropyLoss()(net(x),y),net.parameters())]
        arm=config["arm"]
        if arm.startswith("dp_"):
            payload=dp_payload(raw,config["clip"],config["sigma"],derive_seed(333600,"p33b-dp-target",dataset,arm,config["index"]))
            plans=None
        else:payload,plans=transformed_gradient(raw,arm)
        torch.save(dict(kind="v2_sketch" if plans is not None else "gradient",payload=payload,plans=plans,
            labels=y,model=net.state_dict()),folder/"server_receipt.pt")
        # Private evaluator truth never enters the native loss/true_data argument.
        torch.save(dict(truth=x,labels=y,indices=indices,source_ids=manifest["source_ids"]["n39"][config["index"]]),folder/"scoring_truth.pt")
        seed=333400+(0 if dataset=="ieee_cis" else 1000)+200+config["index"]
        variants=[("plain",payload)]
        if arm=="dna_v1_conservative":variants.append(("structure_debias",debias_blocks(payload)))
        candidates=[]
        for name,observed in variants:
            rec,ensemble,losses=native_recover(net,data,observed,y,seed,mode="v2_sketch" if plans is not None else "plain",plans=plans)
            candidates.append((float(min(losses)),name,rec,ensemble,losses))
        objective,name,rec,ensemble,losses=min(candidates,key=lambda item:item[0])
        score=measure(data,x,rec)
        baselines,mm,emp=controls(data,x,333500+(0 if dataset=="ieee_cis" else 1000)+200+config["index"])
        torch.save(dict(reconstruction=rec,ensemble=ensemble,objective_losses=losses,mean_mode=mm,empirical_single=emp),folder/"reconstruction.pt")
        result=dict(status="COMPLETED",config=config,accuracy=score,baselines=baselines,attack_seed=seed,
            source_ids=manifest["source_ids"]["n39"][config["index"]],selected_variant=name,selected_objective=objective,
            variants=[dict(name=item[1],observable_objective=item[0],accuracy=measure(data,x,item[2])) for item in candidates],
            files={path.name:sha(path) for path in folder.glob("*.pt")})
    except Exception:result=dict(status="GATE_FAILED",config=config,error=traceback.format_exc())
    result.update(elapsed_seconds=time.time()-started,device="cpu",torch_threads=torch.get_num_threads(),finished_at=now())
    write(folder/"result.json",result)


def execute(config):
    folder=job_path(config)
    path=folder/"result.json"
    if path.exists():
        doc=json.loads(path.read_text())
        assert doc["config"]==config
        if doc["status"]=="COMPLETED":
            if config["kind"]=="utility":assert sha(folder/"final_state.pt")==doc["checkpoint_sha256"]
            else:
                for name,digest in doc["files"].items():assert sha(folder/name)==digest
        return doc
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/"rounds.jsonl").exists():raise RuntimeError("partial training: explicit interruption review required")
    command=[sys.executable,"-B","-u",str(Path(__file__).resolve()),"--job",json.dumps(config,sort_keys=True)]
    append(OUT/"runs.jsonl",dict(at=now(),event="cell_job_started",config=config,command=command))
    with (folder/"stdout.log").open("a") as so,(folder/"stderr.log").open("a") as se:
        proc=subprocess.run(command,cwd=ROOT,stdout=so,stderr=se)
    if not path.exists():raise RuntimeError(f"missing result, returncode{proc.returncode}: {config}")
    return json.loads(path.read_text())


def group(configs,stage):
    start=time.time()
    rows=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures={pool.submit(execute,config):config for config in configs}
        for future in as_completed(futures):
            config=futures[future]
            row=future.result()
            rows.append(row)
            append(OUT/"runs.jsonl",dict(at=now(),event="cell_job_finished",config=config,status=row["status"],result_sha256=sha(job_path(config)/"result.json")))
            doc=dict(stage=stage,dataset=config["dataset"],method=config.get("method",config.get("arm")),done=len(rows),total=len(configs),
                failed=sum(r["status"]!="COMPLETED" for r in rows),last_update=now(),eta_minutes=(time.time()-start)/len(rows)*(len(configs)-len(rows))/60)
            for destination in (OUT,OUT.parent):
                write(destination/"progress.json",doc)
                append(destination/"progress.log",doc)
    return rows


def utility(dataset):
    configs=[dict(kind="utility",dataset=dataset,method=method,seed=seed) for method in ("baseline","dna_v1_conservative","dna_v2_0p95") for seed in SEEDS]
    rows=group(configs,"utility_baseline_and_transforms")
    if any(row["status"]!="COMPLETED" for row in rows):
        result=dict(status="NOT_ASSESSABLE",reason="required transform/baseline replicate failed",failed=[r["config"] for r in rows if r["status"]!="COMPLETED"])
        write(OUT/dataset/"utility_calibration.json",result)
        return result
    by_method={m:{r["config"]["seed"]:r for r in rows if r["config"]["method"]==m} for m in ("baseline","dna_v1_conservative","dna_v2_0p95")}
    baseline=np.array([by_method["baseline"][seed]["validation"]["f1"] for seed in SEEDS])
    baseline_auc=np.mean([by_method["baseline"][seed]["validation"]["auc_roc"] for seed in SEEDS])
    prevalence=rows[0]["validation"]["fraud_rate"]
    if baseline_auc<.70 or baseline.mean()<=2*prevalence/(1+prevalence):
        result=dict(status="NOT_ASSESSABLE",reason="validation baseline quality gate",mean_f1=float(baseline.mean()),mean_auc=float(baseline_auc))
        write(OUT/dataset/"utility_calibration.json",result)
        return result
    clip=1.01*float(np.percentile(by_method["baseline"][SEEDS[0]]["update_norms"]+by_method["baseline"][SEEDS[1]]["update_norms"],95))
    thresholds={m:float(np.mean([by_method[m][seed]["validation"]["f1"]-baseline[i] for i,seed in enumerate(SEEDS)]))-.005 for m in ("dna_v1_conservative","dna_v2_0p95")}
    clip_path=OUT/dataset/"utility_clip_freeze.json"
    clip_doc=dict(clip=clip,thresholds=thresholds,sigmas=SIGMAS,baseline=baseline.tolist())
    if clip_path.exists():assert json.loads(clip_path.read_text())==clip_doc
    else:write(clip_path,clip_doc)
    points=[]
    for grid in (SIGMAS,[.01,.03]):
        dpconfigs=[dict(kind="utility",dataset=dataset,method="dp",seed=seed,sigma=sigma,clip=clip) for sigma in grid for seed in SEEDS]
        dp_rows=group(dpconfigs,"utility_dp_grid")
        for sigma in grid:
            selected={r["config"]["seed"]:r for r in dp_rows if r["config"]["sigma"]==sigma}
            valid=all(r["status"]=="COMPLETED" for r in selected.values()) and len(selected)==16
            delta=float(np.mean([selected[seed]["validation"]["f1"]-baseline[i] for i,seed in enumerate(SEEDS)])) if valid else None
            points.append(dict(sigma=sigma,valid=valid,mean_paired_validation_delta=delta))
        if grid==SIGMAS and all(any(p["valid"] and p["mean_paired_validation_delta"]<threshold for p in points) for threshold in thresholds.values()):break
    matches={}
    valid=[p for p in points if p["valid"]]
    for method,threshold in thresholds.items():
        feasible=[p for p in valid if p["mean_paired_validation_delta"]>=threshold]
        chosen=max(feasible,key=lambda p:p["sigma"]) if feasible else None
        higher=[p for p in valid if chosen is not None and p["sigma"]>chosen["sigma"]]
        bracket=min(higher,key=lambda p:p["sigma"]) if higher else None
        matched=chosen is not None and bracket is not None and bracket["mean_paired_validation_delta"]<threshold
        matches[method]=dict(status="MATCHED" if matched else "NOT_ASSESSABLE",reason=None if matched else "NOT_BRACKETED",clip=clip,
            threshold=threshold,selected=chosen,bracket=bracket,accounting=epsilon(chosen["sigma"]) if matched else None)
    result=dict(status="CALIBRATION_RESOLVED",matches=matches,points=points,baseline_mean_f1=float(baseline.mean()),baseline_mean_auc=float(baseline_auc))
    write(OUT/dataset/"utility_calibration.json",result)
    return result


def supervise():
    prerequisite=OUT/"GATES_COMPLETE.json"
    if not prerequisite.exists():raise RuntimeError("wait for qualification supervisor; no duplicate workloads")
    gates=json.loads(prerequisite.read_text())["outcomes"]
    checklist(3,"in_progress")
    sources=[Path(__file__).resolve(),ROOT/"experiments/priority33b_comparators.py",ROOT/"tests/test_priority33b_comparators.py",ROOT/"protocols/amendments/2026-10-02_priority33b_utility_execution_clarification.md"]
    doc=dict(at=now(),sources={str(p.relative_to(ROOT)):sha(p) for p in sources},qualification_sha256=sha(prerequisite),protocol_sha256=sha(PROTOCOL))
    path=OUT/"comparator_execution_freeze.json"
    if path.exists():
        old=json.loads(path.read_text());assert old["sources"]==doc["sources"] and old["qualification_sha256"]==doc["qualification_sha256"]
    else:write(path,doc)
    calibration={}
    for dataset,gate in gates.items():
        if gate["passed"]:
            calibration[dataset]=dict(distortion=distortion_calibration(dataset),utility=utility(dataset))
        else:calibration[dataset]=dict(status="NOT_ASSESSABLE",reason="qualification gate")
    write(OUT/"CALIBRATION_COMPLETE.json",calibration)
    all_calibrated=all(g["passed"] for g in gates.values()) and all(
        all(c["distortion"][m]["status"]=="MATCHED" and c["utility"].get("matches",{}).get(m,{}).get("status")=="MATCHED"
            for m in ("dna_v1_conservative","dna_v2_0p95")) for c in calibration.values())
    checklist(3,"completed" if all_calibrated else "resolved_with_NOT_ASSESSABLE")
    configs=[]
    for dataset,gate in gates.items():
        if not gate["passed"]:continue
        cal=calibration[dataset]
        for method in ("dna_v1_conservative","dna_v2_0p95"):
            configs += [dict(kind="reconstruction",dataset=dataset,arm=method,index=i) for i in range(39)]
            for comparator in ("distortion","utility"):
                match=cal["distortion"][method] if comparator=="distortion" else cal["utility"].get("matches",{}).get(method,{"status":"NOT_ASSESSABLE"})
                if match["status"]!="MATCHED":continue
                sigma=match["sigma"] if comparator=="distortion" else match["selected"]["sigma"]
                configs += [dict(kind="reconstruction",dataset=dataset,arm=f"dp_{comparator}_{method}",index=i,clip=match["clip"],sigma=sigma) for i in range(39)]
    freeze_path=OUT/"confirmatory_execution_freeze.json"
    freeze_doc=dict(configs=configs,calibration_sha256=sha(OUT/"CALIBRATION_COMPLETE.json"))
    if freeze_path.exists():assert json.loads(freeze_path.read_text())==freeze_doc
    else:write(freeze_path,freeze_doc)
    checklist(4,"in_progress")
    rows=group(configs,"n39_confirmatory") if configs else []
    write(OUT/"CELLS_COMPLETE.json",dict(at=now(),jobs=len(rows),failed=[r["config"] for r in rows if r["status"]!="COMPLETED"],qualified_datasets=[d for d,g in gates.items() if g["passed"]]))
    checklist(4,"completed" if all_calibrated and all(r["status"]=="COMPLETED" for r in rows) else "resolved_with_NOT_ASSESSABLE")


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--supervise",action="store_true")
    parser.add_argument("--job")
    args=parser.parse_args()
    if args.job:
        config=json.loads(args.job);folder=job_path(config);folder.mkdir(parents=True,exist_ok=True)
        if config["kind"]=="utility":utility_training_job(config,folder)
        else:reconstruction_job(config,folder)
    elif args.supervise:supervise()
    else:parser.error("choose --supervise or --job")
