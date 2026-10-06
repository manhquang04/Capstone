"""P33b development comparator definitions; no qualification/source changes."""
from __future__ import annotations
import copy
import json
import time
import traceback
from collections import OrderedDict

from experiments.priority33b_core import *
from dna_encoder.transform_defense import DNATransformConfig, transform_update_array
from dna_encoder.transform_defense_v2 import DNATransformV2Config, transform_and_reconstruct_array_v2
from experiments.priority30_native_defenses.run_audit import fwht_normalized_torch
from privacy.seed_manager import derive_seed
from experiments.dp_update_accounting import alpha_grid, epsilon_from_rdp
from experiments.fraud_fl_common import fed_avg, tune_threshold
from data.load_creditcard import _mild_non_iid_client_indices
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

V1=DNATransformConfig(block_size=256,mix_ratio=.08,keep_ratio=.88,shrink_factor=.45,seed=681958327)
V2=DNATransformV2Config(compression_ratio=.95,quantization_eta=.01,seed=20260916)
SIGMAS=[1e-6,3e-6,1e-5,3e-5,1e-4,3e-4,.001,.003]
SEEDS=list(range(333100,333116))


def transformed_gradient(raw, method):
    if method=="dna_v1_conservative":
        return [torch.from_numpy(transform_update_array(g.numpy(),V1,tensor_index=i)[0].copy()) for i,g in enumerate(raw)],None
    if method=="dna_v2_0p95":
        return v2_sketch_payload(OrderedDict((str(i),g) for i,g in enumerate(raw)),seed=20260916)
    raise ValueError(f"unknown method {method}")


def debias_blocks(payload):
    values=[]
    for tensor in payload:
        flat=tensor.flatten()
        blocks=[(part-.08*part.mean())/.92 for part in flat.split(256)]
        values.append(torch.cat(blocks).reshape_as(tensor))
    return values


def decode_for_distortion(sketches,plans,raw):
    decoded=[]
    for q,p,reference in zip(sketches,plans,raw):
        padded=q.new_zeros(int(p["padded_size"]))
        padded[p["sampled"]]=q*float(p["scale"])
        lifted=fwht_normalized_torch(padded)*p["signs"]
        decoded.append(lifted[:reference.numel()].reshape_as(reference))
    return decoded


def dp_payload(raw,clip,sigma,seed):
    if clip<=0 or sigma<0:raise ValueError("invalid DP configuration")
    norm=torch.sqrt(sum((g.double()**2).sum() for g in raw))
    scale=min(1.,clip/float(norm)) if norm>0 else 1.
    generator=torch.Generator().manual_seed(seed)
    result=[g*scale+torch.randn(g.shape,generator=generator)*clip*sigma for g in raw]
    for g in result:finite(g,"DP transmitted payload")
    return result


def epsilon(sigma):
    receipt=epsilon_from_rdp(noise_multiplier=sigma,sensitivity_ratio=1.,delta=1e-5,compositions=1,orders=alpha_grid())
    return dict(epsilon=float(receipt["epsilon"]),rdp_order=float(receipt["alpha"]),delta=1e-5,accounting="one add/remove update-vector release")


def distortion_calibration(dataset):
    raws=[torch.load(OUT/dataset/"n24"/f"target_{i:03d}/server_receipt.pt",map_location="cpu",weights_only=False)["gradient"] for i in range(24)]
    norms=[float(torch.sqrt(sum((g.double()**2).sum() for g in raw))) for raw in raws]
    clip=1.01*max(norms)
    results={}
    for method in ("dna_v1_conservative","dna_v2_0p95"):
        distortions,gaussian_norms,seeds=[],[],[]
        for i,raw in enumerate(raws):
            payload,plans=transformed_gradient(raw,method)
            decoded=decode_for_distortion(payload,plans,raw) if plans is not None else payload
            distortions.append(float(torch.sqrt(sum(((g-a).double()**2).sum() for g,a in zip(raw,decoded)))))
            seed=derive_seed(333600,"p33b-distortion-calibration",dataset,method,i)
            seeds.append(seed)
            gen=torch.Generator().manual_seed(seed)
            gaussian_norms.append(float(torch.sqrt(sum((torch.randn(g.shape,generator=gen).double()**2).sum() for g in raw))))
        sigma=float(np.median(distortions)/(clip*np.median(gaussian_norms)))
        errors=[abs(clip*sigma*z-d)/d if d>0 else 0. for d,z in zip(distortions,gaussian_norms)]
        passed=sigma>0 and np.median(errors)<=.05
        results[method]=dict(status="MATCHED" if passed else "NOT_ASSESSABLE",clip=clip,sigma=sigma,
            median_relative_error=float(np.median(errors)),defense_distortions=distortions,gaussian_norms=gaussian_norms,seeds=seeds,
            accounting=epsilon(sigma) if sigma>0 else None)
    write(OUT/dataset/"distortion_calibration.json",results)
    return results


def strict_model(net,context):
    for name,value in net.state_dict().items():
        finite(value,context+":"+name)
        if name.endswith("running_var") and (value<0).any():raise FloatingPointError("negative BN variance")


def load_split(dataset,split,adapter):
    raw=np.load(PREPARED/dataset/f"{split}_x.npy",mmap_mode="r")
    values=np.asarray(raw[:,adapter["columns"]],dtype=np.float32)
    values=(values-np.asarray(adapter["mean"],np.float32))/np.asarray(adapter["std"],np.float32)
    finite(values,"utility inputs")
    return torch.from_numpy(values),np.load(PREPARED/dataset/f"{split}_y.npy")


def evaluate_validation(net,x,y):
    net.eval()
    strict_model(net,"utility evaluation")
    with torch.no_grad():
        probabilities=torch.cat([net(batch).softmax(1)[:,1] for batch in x.split(4096)]).numpy()
    finite(probabilities,"utility validation probabilities")
    threshold=float(tune_threshold(y,probabilities))
    return dict(f1=float(f1_score(y,probabilities>=threshold)),auc_roc=float(roc_auc_score(y,probabilities)),
        pr_auc=float(average_precision_score(y,probabilities)),threshold=threshold,rows=len(y),fraud_rate=float(y.mean()))


def utility_training_job(config,folder):
    start=time.time()
    seed=config["seed"]
    dataset=config["dataset"]
    method=config["method"]
    adapter=json.loads((OUT/dataset/"adapter.json").read_text())
    try:
        train_x,train_y=load_split(dataset,"train",adapter)
        val_x,val_y=load_split(dataset,"validation",adapter)
        categories=np.load(PREPARED/dataset/"train_categories.npy",allow_pickle=True)
        partitions=_mild_non_iid_client_indices(pd.DataFrame({"type":categories}),train_y,3,seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        net=FullyConnected(len(adapter["columns"]),[100,100,2])
        loaders=[DataLoader(TensorDataset(train_x[ids],torch.from_numpy(train_y[ids]).long()),batch_size=1024,shuffle=True,
            num_workers=0,generator=torch.Generator().manual_seed(derive_seed(seed,"p33b-loader",client))) for client,ids in enumerate(partitions)]
        criterion=torch.nn.CrossEntropyLoss()
        counts=[len(ids) for ids in partitions]
        norms=[]
        for round_i in range(50):
            base=OrderedDict((k,v.detach().clone()) for k,v in net.state_dict().items())
            states=[]
            for client,loader in enumerate(loaders):
                local=copy.deepcopy(net).train()
                optimizer=torch.optim.Adam(local.parameters(),lr=.001)
                for inputs,labels in loader:
                    assert inputs.device.type=="cpu"
                    optimizer.zero_grad(set_to_none=True)
                    logits=local(inputs)
                    finite(logits,"utility training logits")
                    loss=criterion(logits,labels)
                    finite(loss,"utility training loss")
                    loss.backward()
                    for parameter in local.parameters():finite(parameter.grad,"utility parameter gradient")
                    optimizer.step()
                    strict_model(local,"utility optimizer step")
                state=local.state_dict()
                updates=[(state[k]-base[k]).detach() for k in base]
                norms.append(float(torch.sqrt(sum((g.double()**2).sum() for g in updates))))
                if method=="dna_v1_conservative":
                    changed,_=transformed_gradient(updates,method)
                elif method=="dna_v2_0p95":
                    changed=[torch.from_numpy(transform_and_reconstruct_array_v2(g.numpy(),V2,tensor_index=i,
                        quantization_seed=derive_seed(seed,"p33b-utility-v2",round_i,client,i))[1].copy()) for i,g in enumerate(updates)]
                elif method=="dp":
                    changed=dp_payload(updates,config["clip"],config["sigma"],derive_seed(seed,"p33b-utility-dp",round_i,client))
                elif method=="baseline":changed=updates
                else:raise ValueError("unknown utility method")
                release=OrderedDict((k,base[k]+g) for k,g in zip(base,changed))
                for value in release.values():finite(value,"utility transmitted state")
                states.append(release)
            net.load_state_dict(fed_avg(states,counts))
            strict_model(net,"utility aggregate")
            append(folder/"rounds.jsonl",dict(round=round_i+1,state_finite=True))
        metrics=evaluate_validation(net,val_x,val_y)
        torch.save(net.state_dict(),folder/"final_state.pt")
        result=dict(status="COMPLETED",config=config,validation=metrics,update_norms=norms,checkpoint_sha256=sha(folder/"final_state.pt"))
    except Exception:result=dict(status="GATE_FAILED",config=config,error=traceback.format_exc())
    result.update(elapsed_seconds=time.time()-start,device="cpu",torch_threads=torch.get_num_threads(),finished_at=now())
    write(folder/"result.json",result)
