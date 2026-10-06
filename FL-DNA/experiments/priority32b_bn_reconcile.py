"""Frozen read-only-hook reconciliation; original code and artifacts untouched."""
import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/priority32b_bn_reconciliation"
AMENDMENT = ROOT / "protocols/amendments/2026-10-02_priority32b_bn_variance_reconciliation.md"
SEEDS = {"v1": [1984441809, 653660776, 650446433], "v2": [94400035, 1530452029, 1646028911]}
CORE = ["round", "train_loss", "f1_score", "auc_roc", "pr_auc", "accuracy", "precision", "recall", "optimal_threshold", "tn", "fp", "fn", "tp"]


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    assert path.resolve().is_relative_to(OUT.resolve())
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    temp.replace(path)


def event(kind, **fields):
    OUT.mkdir(exist_ok=True)
    with (OUT / "runs.jsonl").open("a") as f:
        f.write(json.dumps({"at": now(), "event": kind, **fields})+"\n")


def stored(method, seed):
    branch = "rq2/confirmatory_run_20260913" if method == "v1" else "rq2_v2/confirmatory_20260916"
    name = "dna_transform" if method == "v1" else "dna_transform_v2"
    return ROOT / "artifacts" / branch / f"seed_{seed}" / name / "metrics.json"


def run_job(job, folder):
    # Environment frozen before importing modules with import-time constants.
    seed, method, factor = job["seed"], job["method"], job["factor"]
    os.environ.update({"FL_RUN_SEED": str(seed), "MAX_ROWS": "500000", "NUM_ROUNDS": "50",
        "LOCAL_EPOCHS": "1", "FL_NUM_CLIENTS": "3", "LOSS_TYPE": "focal", "FOCAL_ALPHA": ".95",
        "FOCAL_GAMMA": "2.0", "DNA_TRANSFORM_BLOCK_SIZE": "256", "DNA_TRANSFORM_MIX": ".08",
        "DNA_TRANSFORM_KEEP": ".88", "DNA_TRANSFORM_SHRINK": ".45",
        "DNA_TRANSFORM_V2_COMPRESSION_RATIO": ".95", "DNA_TRANSFORM_V2_QUANTIZATION_ETA": ".01",
        "DATALOADER_NUM_WORKERS": "4", "DNA_TRANSFORM_OUTPUT_PATH": str(folder/"metrics.json"),
        "DNA_TRANSFORM_V2_OUTPUT_PATH": str(folder/"metrics.json"), "PYTHONDONTWRITEBYTECODE": "1"})
    for var in ("DNA_TRANSFORM_RUN_SEED", "DNA_TRANSFORM_V2_RUN_SEED", "QUICK"):
        os.environ.pop(var, None)
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"
    sys.path.insert(0, str(ROOT))
    import numpy as np
    import torch
    from experiments import fraud_fl_common as common
    from experiments import priority32_multidataset as p
    from models.fraud_mlp import FraudMLP
    from data import load_creditcard as data
    torch.set_num_threads(1)
    folder.mkdir(parents=True, exist_ok=True)
    rounds, evaluations = [], []
    final_state = {}
    state_calls = 0
    evaluation_round = 0

    def read_state(states, counts):
        nonlocal state_calls, evaluation_round, final_state
        rng = torch.get_rng_state().clone()
        averaged = common.fed_avg(states, counts)
        state_calls += 1
        evaluation_round = state_calls
        def variance(state):
            return {k: {"min": float(v.detach().cpu().min()), "negative": int((v < 0).sum().cpu()),
                        "nonfinite": int((~torch.isfinite(v)).sum().cpu())}
                    for k, v in state.items() if k.endswith("running_var")}
        row = {"round": state_calls, "aggregate": variance(averaged), "transmitted_clients": [variance(s) for s in states]}
        assert torch.equal(rng, torch.get_rng_state()), "hook changed CPU RNG"
        rounds.append(row)
        with (folder/"bn_rounds.jsonl").open("a") as f:
            f.write(json.dumps(row)+"\n")
        if state_calls == 50:
            final_state = {k: v.detach().cpu().clone() for k,v in averaged.items()}
            torch.save(final_state, folder/"final_state.pt")
        return averaged

    def observe_evaluation(model, loader, threshold=None):
        record = {"round": evaluation_round, "split": "validation" if threshold is None else "test", "finite_logits": True, "rows": 0}
        values = []
        def hook(module, inputs, outputs):
            before = torch.get_rng_state().clone()
            record["finite_logits"] &= bool(torch.isfinite(outputs).all().cpu())
            record["rows"] += len(outputs)
            values.append(torch.sigmoid(outputs.detach()).cpu().numpy().reshape(-1).copy())
            assert torch.equal(before, torch.get_rng_state())
        handle = model.register_forward_hook(hook)
        try:
            metrics = common.evaluate_model(model, loader, threshold)
            record["metrics"] = metrics
            return metrics
        finally:
            handle.remove()
            probabilities = np.concatenate(values) if values else np.empty(0)
            record["finite_probabilities"] = bool(np.isfinite(probabilities).all())
            record["nonfinite_probabilities"] = int((~np.isfinite(probabilities)).sum())
            evaluations.append(record)
            with (folder/"evaluations.jsonl").open("a") as f:
                f.write(json.dumps(record)+"\n")
            if evaluation_round == 50:
                np.save(folder/(record["split"]+"_probabilities.npy"), probabilities, allow_pickle=False)

    native_device = str(common.DEVICE)
    failure = None
    started = time.time()
    try:
        if factor == "registered":
            if method == "v1":
                from experiments import run_fraud_fl_dna_transform as native
            else:
                from experiments import run_fraud_fl_dna_transform_v2 as native
            native.fed_avg = read_state
            native.evaluate_model = observe_evaluation
            native.main()  # exact original entry point, original environment
        else:
            from torch.utils.data import DataLoader, TensorDataset
            import pandas as pd
            from collections import OrderedDict
            from dataclasses import replace
            from privacy.seed_manager import derive_seed
            import random
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            device = common.DEVICE if factor == "native_device" else torch.device("cpu")
            if str(device) == "mps":
                torch.mps.manual_seed(seed)
            if factor in {"registered_data", "cap_before_fixed", "cap_after_per_seed"}:
                ds_seed = 320032 if factor == "cap_before_fixed" else seed
                if factor != "cap_after_per_seed":
                    pooled = data.load_paysim_splits(batch_size=1024, seed=ds_seed, max_rows=500000)
                    arrays = [loader.dataset.tensors[0].numpy() for loader in pooled[:3]]
                    labels = [loader.dataset.tensors[1].numpy().reshape(-1) for loader in pooled[:3]]
                    categories = np.array(pooled[-1].type_categories)[arrays[0][:,8:].argmax(axis=1)]
                else:
                    frame = pd.read_csv(ROOT/"datasets/creditcard.csv", usecols=data.BASE_FEATURE_COLUMNS+["isFraud"])
                    all_y = frame.isFraud.to_numpy()
                    ids = np.arange(len(frame))
                    train,temp = p.train_test_split(ids,test_size=.35,stratify=all_y,random_state=ds_seed)
                    val,test = p.train_test_split(temp,test_size=.20/.35,stratify=all_y[temp],random_state=ds_seed)
                    parts = p.capped_parts([train,val,test],all_y,seed=320032)
                    arrays, metadata = p.preprocess(data._build_features(frame),parts,["type"])
                    labels = [all_y[x].astype(np.float32) for x in parts]
                    categories = frame.iloc[parts[0]].type.to_numpy()
            else:
                prepared = p.OUT/"prepared/paysim"
                arrays = [np.load(prepared/(split+"_x.npy")) for split in ("train","validation","test")]
                labels = [np.load(prepared/(split+"_y.npy")) for split in ("train","validation","test")]
                categories = np.load(prepared/"train_categories.npy")
            partitions = data._mild_non_iid_client_indices(pd.DataFrame({"type":categories}),labels[0],3,seed)
            assert np.unique(np.concatenate(partitions)).size == len(labels[0])
            workers = 4 if factor == "loader_workers" else 0
            def loader(x,y,shuffle,loader_seed):
                return DataLoader(TensorDataset(torch.from_numpy(np.array(x,dtype=np.float32,copy=True)),
                    torch.from_numpy(np.array(y,dtype=np.float32,copy=True)[:,None])),batch_size=1024,shuffle=shuffle,
                    generator=torch.Generator().manual_seed(loader_seed),num_workers=workers,persistent_workers=workers>0)
            loaders = [loader(arrays[0][idx],labels[0][idx],True,seed+i) for i,idx in enumerate(partitions)]
            validation = loader(arrays[1],labels[1],False,seed)
            test = loader(arrays[2],labels[2],False,seed)
            model = FraudMLP(arrays[0].shape[1]).to(device)
            criterion = common.BinaryFocalLoss(alpha=.95,gamma=2.)
            for round_number in range(1,51):
                global_state = copy.deepcopy(model.state_dict())
                states = []
                for client, client_loader in enumerate(loaders):
                    local = copy.deepcopy(model).train()
                    optimizer = torch.optim.Adam(local.parameters(),lr=.001)
                    for features,target in client_loader:
                        features,target = features.to(device),target.to(device)
                        optimizer.zero_grad()
                        loss = criterion(local(features),target)
                        if not torch.isfinite(loss):
                            raise ValueError("nonfinite training loss")
                        loss.backward(); optimizer.step()
                    if method == "v1":
                        client_seed=derive_seed(derive_seed(seed,"rq2-dna-transform"),"dna_transform",round_number,client)
                        state,_=p.dna_transform_state(local.state_dict(),global_state,p.DNATransformConfig(block_size=256,mix_ratio=.08,keep_ratio=.88,shrink_factor=.45,seed=client_seed))
                    else:
                        root_label="dna-transform-v2" if factor=="transform_seed_label" else "rq2-dna-transform-v2"
                        client_seed=derive_seed(derive_seed(seed,root_label),"dna_transform_v2",round_number,client)
                        state,_=p.dna_transform_v2_state(local.state_dict(),global_state,p.DNATransformV2Config(compression_ratio=.95,quantization_eta=.01,seed=client_seed),derive_seed(client_seed,"quantization"))
                    states.append(OrderedDict((k,v.detach().clone()) for k,v in state.items()))
                model.load_state_dict(read_state(states,list(map(len,partitions))))
                if factor == "evaluation_cadence":
                    # Original eval loader/metric semantics; common DEVICE forced
                    # to this job device, not an accidental second factor.
                    common.DEVICE=device
                    val=observe_evaluation(model,validation)
                    observe_evaluation(model,test,float(val["optimal_threshold"]))
            for split,x,y in zip(("validation","test"),arrays[1:],labels[1:]):
                probability=[]
                model.eval()
                with torch.no_grad():
                    for start in range(0,len(x),4096):
                        probability.append(torch.sigmoid(model(torch.from_numpy(np.array(x[start:start+4096],copy=True)).to(device))).reshape(-1).cpu().numpy())
                probability=np.concatenate(probability)
                np.save(folder/(split+"_probabilities.npy"),probability,allow_pickle=False)
                rec={"round":50,"split":split,"rows":len(y),"finite_probabilities":bool(np.isfinite(probability).all()),"nonfinite_probabilities":int((~np.isfinite(probability)).sum())}
                if rec["finite_probabilities"]:
                    if split=="validation":
                        threshold=common.tune_threshold(y,probability)
                    rec["metrics"]=p.metrics(y,probability,threshold)
                evaluations.append(rec)
                with (folder/"evaluations.jsonl").open("a") as f:
                    f.write(json.dumps(rec)+"\n")
    except Exception:
        failure=traceback.format_exc()
    negative_rounds=[r["round"] for r in rounds if any(v["negative"] for v in r["aggregate"].values())]
    result={"job":job,"status":"FAILED" if failure else "COMPLETED","exception":failure,"rounds_completed":len(rounds),
            "first_negative_round":min(negative_rounds) if negative_rounds else None,"negative_rounds":negative_rounds,
            "native_device":native_device,"evaluations":evaluations,"elapsed_seconds":time.time()-started,"torch_threads":torch.get_num_threads()}
    if factor=="registered" and (folder/"metrics.json").exists():
        original=json.loads(stored(method,seed).read_text())
        replay=json.loads((folder/"metrics.json").read_text())
        mismatch=[]; errors=[]
        for a,b in zip(original["rounds"],replay["rounds"]):
            for k in CORE:
                if a[k] is None or b[k] is None:
                    if a[k]!=b[k]: mismatch.append([a["round"],k,a[k],b[k]])
                else:
                    err=abs(float(a[k])-float(b[k])); errors.append(err)
                    if err>1e-8 or (k in {"tn","fp","fn","tp"} and err): mismatch.append([a["round"],k,a[k],b[k]])
        result["reproduction"]={"config_equal":original["config"]==replay["config"],"round_count_equal":len(original["rounds"])==len(replay["rounds"]),"mismatches":mismatch,"max_absolute_error":max(errors,default=0.),"bit_equal":not mismatch and max(errors,default=0.)==0.}
    write(folder/"result.json",result)


def supervise():
    OUT.mkdir(exist_ok=True)
    lock=OUT/"supervisor.json"
    if lock.exists():
        pid=json.loads(lock.read_text())["pid"]
        try: os.kill(pid,0)
        except ProcessLookupError: pass
        else: raise RuntimeError("duplicate supervisor")
    write(lock,{"pid":os.getpid(),"at":now()})
    jobs=[{"method":m,"seed":s,"factor":f} for f in ("registered","p32_full","registered_data","native_device","evaluation_cadence","loader_workers","transform_seed_label") for m,seeds in SEEDS.items() for s in seeds if f!="transform_seed_label" or m=="v2"]
    freeze={"jobs":jobs,"amendment_sha256":sha(AMENDMENT),"runner_sha256":sha(__file__),"stored_hashes":{str(stored(m,s).relative_to(ROOT)):sha(stored(m,s)) for m,seeds in SEEDS.items() for s in seeds}}
    if (OUT/"execution_freeze.json").exists(): assert json.loads((OUT/"execution_freeze.json").read_text())==freeze
    else: write(OUT/"execution_freeze.json",freeze)
    start=time.time(); completed=failed=0
    def one(job):
        folder=OUT/"jobs"/job["factor"]/job["method"]/f"seed_{job['seed']}"
        result=folder/"result.json"
        if result.exists():
            doc=json.loads(result.read_text()); assert doc["job"]==job
            event("validated_skip",job=job,status=doc["status"])
            return doc
        folder.mkdir(parents=True,exist_ok=True)
        command=[sys.executable,"-B","-u",str(Path(__file__).resolve()),"--job",json.dumps(job),"--folder",str(folder)]
        event("job_started",job=job,command=command)
        with (folder/"stdout.log").open("a") as so,(folder/"stderr.log").open("a") as se:
            proc=subprocess.run(command,cwd=ROOT,stdout=so,stderr=se)
        if not result.exists():
            doc={"job":job,"status":"FAILED","returncode":proc.returncode,"unrecoverable_process_error":True}
            write(result,doc)
        doc=json.loads(result.read_text()); event("job_finished",job=job,status=doc["status"],sha256=sha(result))
        return doc
    def stage(batch):
        nonlocal completed,failed
        with ThreadPoolExecutor(max_workers=4) as pool:
            for future in as_completed([pool.submit(one,j) for j in batch]):
                doc=future.result(); completed+=1; failed+=int(doc["status"]=="FAILED")
                progress={"stage":"diagnostic_replays","dataset":"paysim","method":doc["job"]["method"],"done":completed,"total":len(jobs),"failed":failed,"started_at":datetime.fromtimestamp(start,timezone.utc).isoformat(),"last_update":now(),"eta_minutes":(time.time()-start)/completed*(len(jobs)-completed)/60}
                write(OUT/"progress.json",progress)
                with (OUT/"progress.log").open("a") as f:f.write(json.dumps(progress)+"\n")
    stage(jobs)
    decomposition=False
    for m,seeds in SEEDS.items():
        for s in seeds:
            def doc(f):return json.loads((OUT/"jobs"/f/m/f"seed_{s}"/"result.json").read_text())
            if (doc("p32_full").get("first_negative_round") is None)!=(doc("registered_data").get("first_negative_round") is None): decomposition=True
    if decomposition:
        extra=[{"method":m,"seed":s,"factor":f} for f in ("cap_before_fixed","cap_after_per_seed") for m,seeds in SEEDS.items() for s in seeds]
        jobs.extend(extra);event("predeclared_data_decomposition_triggered",jobs=extra);stage(extra)
    write(OUT/"REPLAYS_COMPLETE.json",{"jobs":completed,"failed":failed,"elapsed_seconds":time.time()-start,"data_decomposition":decomposition})
    event("diagnostic_replays_finished",jobs=completed,failed=failed)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--supervise",action="store_true");parser.add_argument("--job");parser.add_argument("--folder",type=Path);args=parser.parse_args()
    if args.supervise:supervise()
    else:run_job(json.loads(args.job),args.folder)
