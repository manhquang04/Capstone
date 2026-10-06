"""P34B image FedAvg utility and frozen P33c observable recovery."""
from __future__ import annotations
import copy
import math
import random
import time
from collections import OrderedDict
from pathlib import Path
from experiments import priority34b_core as b
from experiments.priority34b_image_imports import p31, recovery
from experiments.priority33c_checkpoint_replay import accuracy, finite, check

np, torch, ROOT, OUT = b.np, b.torch, b.ROOT, b.OUT


def image_model():
    model,_ = p31.dp.inversefed.construct_model('LeNetZhu',seed=42)
    model.cpu()
    if list(model.buffers()):
        raise ValueError('unexpected LeNet BN/buffers')
    check(model)
    return model


def tensors(data, ids):
    x,y = p31.tensors(data,ids)
    return x.contiguous(),y


def utility_job(job):
    path = OUT/job['result']
    if b.valid(path,job):
        return dict(path=str(path),skipped=True)
    started = time.time()
    seed = job['seed']
    private_seed=b.noise_key(job) if job['method']!='baseline' else None
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    data = p31.dp.CIFAR10(str(ROOT/'datasets/cifar10'),train=True,download=False)
    ids = b.read(p31.OUT/'split.json')
    x,y = tensors(data,ids['train_ids']); vx,vy = tensors(data,ids['validation_ids'])
    order = torch.randperm(len(x),generator=torch.Generator().manual_seed(seed))
    partitions = list(order.reshape(3,4000))
    model = image_model()
    clients = [copy.deepcopy(model) for _ in range(3)]
    current = OrderedDict((n,p.detach().clone()) for n,p in model.named_parameters())
    loaders = [torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x[idx],y[idx]),
              batch_size=256,shuffle=True,num_workers=0,
              generator=torch.Generator().manual_seed(seed+i)) for i,idx in enumerate(partitions)]
    attempt = path.parent/(path.stem+'.attempt_'+str(time.time_ns()))
    attempt.mkdir(parents=True,exist_ok=False)
    journal = attempt/'rounds.jsonl'
    audits = []
    for round_number in range(1,51):
        uploads,clips = [],[]
        for client,(local,loader) in enumerate(zip(clients,loaders)):
            local.load_state_dict(current); local.train()
            optimizer = torch.optim.SGD(local.parameters(),lr=.1,momentum=0)
            for features,labels in loader:
                optimizer.zero_grad()
                logits = local(features); finite(logits,'local image logits')
                loss = torch.nn.functional.cross_entropy(logits,labels); finite(loss,'local image loss')
                loss.backward()
                for parameter in local.parameters():
                    finite(parameter.grad,'local image gradient')
                optimizer.step(); check(local)
            delta = OrderedDict((n,p.detach()-current[n]) for n,p in local.named_parameters())
            if job['method']=='baseline':
                changed = delta
            else:
                changed,receipt = b.clip(delta,job['mechanism'])
                clips.append(dict(client=client,no_bn_transmitted=True,**receipt))
            uploads.append(changed)
        if job['method']=='baseline':
            current = OrderedDict((n,current[n]+sum(row[n]/3 for row in uploads)) for n in current)
            audit = dict(round=round_number,weights=[1/3]*3)
        else:
            current,audit = b.aggregate(uploads,[1/3]*3,job['epsilon'],private_seed,round_number,current)
        model.load_state_dict(current); check(model)
        audit.update(no_bn_transmitted=True,payload_keys=list(current),upload_assertions=3,
                     client_clip_audits=clips,validation_accuracy=accuracy(model,vx,vy))
        audits.append(audit)
        with journal.open('a') as stream:
            import json
            stream.write(json.dumps(audit,allow_nan=False)+'\n')
    logits = []
    model.eval()
    with torch.no_grad():
        for start in range(0,len(vx),512):
            value = model(vx[start:start+512]); finite(value,'validation logits')
            logits.append(value.numpy())
    logits_path = attempt/'validation_logits.npy'
    np.save(logits_path,np.concatenate(logits),allow_pickle=False)
    checkpoint = attempt/'final_checkpoint.pt'; torch.save(current,checkpoint)
    files = [journal,logits_path,checkpoint]
    doc = dict(status='COMPLETED',job=job,job_sha256=b.p.canonical_sha(job),
               validation_accuracy=audits[-1]['validation_accuracy'],no_bn_transmitted=True,
               torch_threads=torch.get_num_threads(),payload_assertions=150,
               partition_indices=[idx.tolist() for idx in partitions],round_audit=audits,
               artifact_sha256={str(f.relative_to(OUT)):b.p.sha(f) for f in files},
               completed_at=b.p.now(),elapsed_seconds=time.time()-started)
    b.save(path,doc); b.valid(path,job)
    return dict(path=str(path),skipped=False)


def recovery_job(job):
    path = OUT/job['result']
    if b.valid(path,job):
        return dict(path=str(path),skipped=True)
    start = time.time()
    native = recovery.native
    data = p31.dp.CIFAR10(str(ROOT/'datasets/cifar10'),train=False,download=False)
    normal,labels = tensors(data,[job['source_id']])
    dm = torch.tensor(native.inversefed.consts.cifar10_mean)[:,None,None]
    ds = torch.tensor(native.inversefed.consts.cifar10_std)[:,None,None]
    truth = normal*ds+dm
    model = image_model(); model.eval()
    state = native.gradient_dict(model,normal,labels,torch.nn.CrossEntropyLoss())
    plans,mix = None,0
    arm = job['method']
    if arm=='dna_v1_conservative':
        transmitted = list(recovery.utility.transform(state,arm).values())
        modes,mix = ['plain','v1_debias'],.08
    elif arm=='dna_v2_0p95':
        transmitted,plans = native.v2_sketch_payload(state)
        modes = ['v2_sketch']
    elif arm=='unprotected':
        transmitted,modes = list(state.values()),['plain']
    else:
        clipped,clip_receipt = b.clip(state,job['mechanism'])
        sd = 2*b.account(10)['sigma_sensitivity']*b.C
        generator = torch.Generator().manual_seed(b.noise_key(job))
        transmitted = [(value.double()+torch.randn(value.shape,generator=generator,dtype=torch.float64)*sd).to(value)
                       for value in clipped.values()]
        modes = ['plain']
    for value in transmitted:
        finite(value,'observed payload')
    candidates = []
    for mode in modes:
        torch.manual_seed(job['attack_seed']); np.random.seed(job['attack_seed'])
        evaluator = recovery.ObservableReconstructor(model,(dm,ds),native.IMAGE_CONFIG.copy(),
                     transmitted,mode,mix=mix,plans=plans,count=1)
        result,stats = evaluator.reconstruct(transmitted,labels,img_shape=(3,32,32))
        candidates.append((float(stats['opt']),mode,result.detach(),dict(stats)))
    best = min(enumerate(candidates),key=lambda row:(row[1][0],row[0]))[1]
    estimate = best[2]*ds+dm; finite(estimate,'pre-metric image recovery')
    scores = native.image_metrics(truth,estimate)
    if not all(math.isfinite(v) for v in scores.values()):
        raise FloatingPointError('nonfinite recovery metric')
    references = {name:native.image_metrics(truth,guess) for name,guess in
                  [('gray',torch.full_like(truth,.5)),('cifar_mean',dm.expand_as(truth))]}
    folder = path.parent/(path.stem+'.attempt_'+str(time.time_ns()))
    folder.mkdir(parents=True,exist_ok=False)
    arrays = folder/'arrays.npz'
    np.savez(arrays,truth=truth.numpy(),reconstruction=estimate.numpy(),
             source_id=job['source_id'],label=labels.numpy())
    receipt = folder/'receipt.pt'
    torch.save(dict(payload=transmitted,projection_plans=plans,kind='transmitted_only'),receipt)
    doc = dict(status='COMPLETED',job=job,job_sha256=b.p.canonical_sha(job),
               metrics=scores,references=references,selected_mode=best[1],
               observable_objectives={c[1]:c[0] for c in candidates},
               iterations=4800,restarts=1,no_bn_transmitted=True,torch_threads=torch.get_num_threads(),
               artifact_sha256={str(f.relative_to(OUT)):b.p.sha(f) for f in [arrays,receipt]},
               elapsed_seconds=time.time()-start,completed_at=b.p.now())
    if arm.startswith('dp_'):
        doc.update(clip_audit=clip_receipt,noise_sd=sd,accounting=b.account(10),
                   observation='one conditional aggregate divided by target weight; not final model inversion')
    b.save(path,doc); b.valid(path,job)
    return dict(path=str(path),skipped=False)
