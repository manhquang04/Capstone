"""Seed-redacted metadata recognizer; no raw update/true seed in scoring API."""
from dataclasses import asdict,replace
import json
import os
import time
import numpy as np
from dna_encoder import transform_defense_v2 as v
from experiments.priority34e import OUT,read,write,progress,verify_freeze

PUBLIC_KEYS={'original_shape','original_size','padded_size','sketch_size','compression_ratio',
             'quantization_eta','quantization_delta','tensor_index','sampled_indices'}

def recognize(candidate,public):
    if set(public)!=PUBLIC_KEYS:raise ValueError('public metadata firewall')
    derived=v._derive_seed(candidate,public['tensor_index'])
    indices=v._sampled_indices(public['padded_size'],public['sketch_size'],derived)
    return bool(np.array_equal(indices,np.asarray(public['sampled_indices'],dtype=np.int64)))

def ranks(scores,true_position):
    scores=np.asarray(scores)
    true_score=int(scores[true_position])
    rank=1+int(np.count_nonzero(scores>true_score))
    tied=int(np.count_nonzero(scores==true_score))
    false_positives=int(np.count_nonzero(scores))-true_score
    return dict(true_rank=rank,tied_at_true_score=tied,false_positives=false_positives,
                recognition_validated=bool(true_score==1 and rank==1 and tied==1))

def capture():
    from experiments import priority34a_local_bn as a
    p=a.p
    folder=p.OUT/'prepared/baf'
    p.torch.manual_seed(343100);np.random.seed(343100)
    model=p.FraudMLP(58).cpu();a.guard(model,'capture_initial')
    x=p.torch.from_numpy(np.array(np.load(folder/'train_x.npy',mmap_mode='r')[:1024],copy=True))
    y=p.torch.from_numpy(np.array(np.load(folder/'train_y.npy',mmap_mode='r')[:1024,None],copy=True))
    before=model.network[0].weight.detach().clone()
    optimizer=p.torch.optim.Adam(model.parameters(),lr=.001)
    optimizer.zero_grad();loss=p.BinaryFocalLoss(alpha=.95,gamma=2.)(model(x),y)
    if not p.torch.isfinite(loss):raise ValueError('capture nonfinite')
    loss.backward();optimizer.step();a.guard(model,'capture_final')
    raw=(model.network[0].weight.detach()-before).numpy().copy()
    q,metadata=v.transform_update_array_v2(raw,v.DNATransformV2Config(seed=343101),tensor_index=0,quantization_seed=343102)
    public=asdict(metadata);public.pop('seed')
    path=OUT/'seed_search';path.mkdir(parents=True,exist_ok=True)
    with (path/'private_validator.npz').open('xb') as f:
        if os.name=='posix':os.fchmod(f.fileno(),0o600)
        np.savez(f,raw_update=raw,true_base_seed=np.array(343101),derived_seed=np.array(metadata.seed))
    with (path/'sketch.npy').open('xb') as f:np.save(f,q,allow_pickle=False)
    write(path/'public_metadata.json',public,once=True)
    write(path/'observation_receipt.json',dict(hashes={name:p.sha(path/name) for name in ('private_validator.npz','sketch.npy','public_metadata.json')},
        torch_threads=p.torch.get_num_threads(),interop_threads=p.torch.get_num_interop_threads(),device='cpu',shape=list(raw.shape),
        source='first1024 frozen BAF training rows; one Adam/focal step; no targets or reconstructions'),once=True)

def run_search():
    from experiments.priority32_multidataset import sha
    manifest=verify_freeze();config=manifest['seed_search'];folder=OUT/'seed_search'
    if not (folder/'observation_receipt.json').exists():
        if folder.exists() and list(folder.iterdir()):raise RuntimeError('partial capture retained; requires direction')
        capture()
    receipt=read(folder/'observation_receipt.json')
    for name,digest in receipt['hashes'].items():
        if sha(folder/name)!=digest:raise ValueError('observation mutated')
    public=read(folder/'public_metadata.json');all_scores=[];times=[]
    for begin in range(0,config['total'],config['chunk_size']):
        end=min(begin+config['chunk_size'],config['total']);result=folder/f'chunk_{begin:07d}.json'
        scores_path=folder/f'scores_{begin:07d}.npy'
        if result.exists():
            chunk=read(result)
            if chunk['begin']!=begin or chunk['end']!=end or sha(scores_path)!=chunk['scores_sha256']:raise ValueError('chunk receipt mismatch')
            scores=np.load(scores_path,allow_pickle=False)
        else:
            if scores_path.exists():raise RuntimeError('partial chunk preserved; requires direction')
            start=time.perf_counter()
            scores=np.fromiter((recognize(candidate,public) for candidate in range(begin,end)),dtype=np.uint8,count=end-begin)
            elapsed=time.perf_counter()-start
            with scores_path.open('xb') as f:np.save(f,scores,allow_pickle=False)
            chunk=dict(begin=begin,end=end,count=end-begin,active_seconds=elapsed,scores_sha256=sha(scores_path))
            write(result,chunk,once=True)
        if len(scores)!=end-begin or not np.isin(scores,[0,1]).all():raise ValueError('scores invalid')
        all_scores.append(scores);times.append(chunk['active_seconds'])
        progress('seed_search',candidates_done=end,candidates_total=config['total'])
    scores=np.concatenate(all_scores)
    # Private ground truth is accessed only AFTER complete, immutable scoring.
    private=np.load(folder/'private_validator.npz',allow_pickle=False)
    true_seed=int(private['true_base_seed']);validation=ranks(scores,true_seed)
    matching=np.flatnonzero(scores).tolist();q=np.load(folder/'sketch.npy',allow_pickle=False)
    diagnostics=[]
    for candidate in matching:
        metadata=v.DNATransformV2Metadata(**{**public,'original_shape':tuple(public['original_shape']),
            'sampled_indices':tuple(public['sampled_indices']),'seed':v._derive_seed(candidate,public['tensor_index'])})
        decoded,stats=v.reconstruct_update_array_v2(q,metadata)
        if not np.isfinite(decoded).all():raise ValueError('nonfinite decode')
        diagnostics.append(dict(candidate=candidate,decoded_l2=stats.reconstructed_l2))
    delta=public['quantization_delta']
    grid_error=float(np.max(np.abs(q.astype(np.float64)/delta-np.rint(q.astype(np.float64)/delta))))
    active=sum(times);rates=[config['chunk_size']/t for t in times]
    result=dict(status='SEED_SEARCH_COMPLETE',criterion='exact sampled_indices fingerprint; seed field redacted',
        observed_seed_field=False,projection_indices_public=True,candidates=len(scores),**validation,
        matching_candidates=matching,decoded_diagnostics=diagnostics,grid_error=grid_error,
        active_seconds=active,candidates_per_second=len(scores)/active,
        full32_worst_seconds=2**32/(len(scores)/active),full32_uniform_expected_seconds=2**31/(len(scores)/active),
        full32_chunk_rate_range_seconds=[2**32/max(rates),2**32/min(rates)],full32_executed=False,
        literal_metadata_seed_is_explicit=True,metadata_free_identification_not_tested=True)
    if (folder/'result.json').exists():
        old=read(folder/'result.json')
        if old!=result:raise ValueError('immutable search summary mismatch')
    else:write(folder/'result.json',result,once=True)
    print(json.dumps(result),flush=True)
    return result
