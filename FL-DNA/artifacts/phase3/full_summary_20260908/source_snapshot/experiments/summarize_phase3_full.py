"""Validate and report completed full-client Phase 3 runs without ranking DNA."""
import argparse
import csv
import json
import time
import platform
import shutil
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from dna_encoder.encoder import DNAEncoder
from dna_encoder.transform_defense import DNATransformConfig
from experiments.run_fraud_fl_dna_transform import dna_transform_state
from experiments.run_phase3_full_client import checksum, dump
from privacy.seed_manager import generate_run_seed, derive_seed
from models.fraud_mlp import FraudMLP
from experiments.fraud_fl_common import BinaryFocalLoss


def main():
    torch.set_num_threads(1)
    parser=argparse.ArgumentParser()
    parser.add_argument('runs',nargs='+',type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    records=[]
    captures=[]
    profiles=[]
    signature=None
    for run in args.runs:
        data=json.loads((run/'update_attack_validation.json').read_text())
        if data['failures'] or not data['records']:
            raise ValueError(f'Incomplete run {run}')
        for file,sha in json.loads((run/'manifest.json').read_text()).items():
            if checksum(run/file)!=sha:
                raise ValueError(f'Checksum mismatch: {run/file}')
        config=data['config']
        current=(config['max_rows'],config['data_seed'],config['run_seed'],config['iterations'],config['restarts'],
                 checksum(run/'pre_local.pt'),checksum(run/'preprocessing.json'))
        if signature is not None and current!=signature:
            raise ValueError('Run protocol/checkpoint/scaler mismatch')
        signature=current
        records.extend(data['records'])
        profiles.append(dict(run=str(run),**json.loads((run/'compute_profile.json').read_text())))
        for folder in sorted(run.glob('client_*')):
            cap=torch.load(folder/'attacker_capture.pt',map_location='cpu',weights_only=False)
            targets=torch.load(folder/'evaluator_targets.pt',map_location='cpu',weights_only=False)
            captures.append((folder,cap,targets))
    clients=[int(folder.name.split('_')[-1]) for folder,_,_ in captures]
    if sorted(clients)!=[0,1,2]:
        raise ValueError('Exactly three distinct complete clients required')
    expected=len(clients)*signature[4]*2
    if len(records)!=expected or len({(r['client'],r['restart'],r['method']) for r in records})!=expected:
        raise ValueError('Missing/duplicate attack trials')
    defense_seed=generate_run_seed()
    defense_rows=[]
    for folder,cap,targets in captures:
        client=int(folder.name.split('_')[-1])
        for restart in range(signature[4]):
            saved_b=torch.load(folder/f'baseline_{restart}.pt',weights_only=False)
            saved_z=torch.load(folder/f'zero_update_{restart}.pt',weights_only=False)
            assert torch.equal(saved_b['initial'],saved_z['initial'])
            for saved in (saved_b,saved_z):
                assert len(saved['history'])==signature[3]+1
                assert saved['best']==min(saved['history'])
                assert saved['candidate'].shape==targets['x'].shape
                assert torch.isfinite(saved['candidate']).all()
        encoder=DNAEncoder()
        started=time.perf_counter()
        restored={}
        for k,v in cap['raw'].items():
            if v.is_floating_point():
                array=v.numpy()
                decoded=encoder.decode_array(encoder.encode_array(array),array.shape)
                if not np.array_equal(array.view('uint32'),decoded.view('uint32')):
                    raise AssertionError('Lossless roundtrip changed bits')
                restored[k]=torch.from_numpy(decoded.copy())
            else:
                restored[k]=v.clone()
        lossless_seconds=time.perf_counter()-started
        native=FraudMLP(targets['x'].shape[1]).train()
        native.load_state_dict(cap['pre'])
        optimizer=torch.optim.Adam(native.parameters(),lr=.001)
        offset=0
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(cap['rng'])
            for size in cap['batch_sizes']:
                optimizer.zero_grad()
                BinaryFocalLoss()(native(targets['x'][offset:offset+size]),targets['y'][offset:offset+size]).backward()
                optimizer.step()
                offset+=size
        local=native.state_dict()
        assert all(torch.equal(local[k]-cap['pre'][k],cap['raw'][k]) for k in local)
        cfg=DNATransformConfig(seed=derive_seed(defense_seed,'client',client))
        started=time.perf_counter()
        protected,stats=dna_transform_state(local,cap['pre'],cfg)
        elapsed=time.perf_counter()-started
        replay,_=dna_transform_state(local,cap['pre'],cfg)
        assert all(torch.equal(protected[k],replay[k]) for k in protected)
        torch.save(dict(lossless_delta=restored,transform_state=protected,
            transform_config=asdict(cfg),raw_capture=str(folder/'attacker_capture.pt')),
            args.output/f'defense_capture_client_{client}.pt')
        defense_rows.append(dict(client=client,lossless_bit_exact=True,lossless_seconds=lossless_seconds,
            transform_seconds=elapsed,config=asdict(cfg),stats=stats,
            limitation='capture-only; no adaptive DNA attack or privacy conclusion'))
    selected=[]
    for client in sorted(clients):
        for method in ('baseline','zero_update'):
            group=[r for r in records if r['client']==client and r['method']==method]
            selected.append(min(group,key=lambda r:r['objective']))
    flat=[]
    for row in records:
        flat.append(dict(client=row['client'],restart=row['restart'],method=row['method'],
            mean_mse=row['metrics']['mean_mse'],median_mse=row['metrics']['median_mse'],
            prior_mse=row['prior']['mean_mse'],best_step=row['best_step'],seconds=row['seconds']))
    with (args.output/'metrics_summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]))
        writer.writeheader();writer.writerows(flat)
    comparison=[]
    for client in sorted(clients):
        b=next(r for r in selected if r['client']==client and r['method']=='baseline')
        z=next(r for r in selected if r['client']==client and r['method']=='zero_update')
        paired=[]
        for restart in range(signature[4]):
            rb=next(r for r in records if (r['client'],r['restart'],r['method'])==(client,restart,'baseline'))
            rz=next(r for r in records if (r['client'],r['restart'],r['method'])==(client,restart,'zero_update'))
            paired.append(dict(restart=restart,baseline_minus_zero=rb['metrics']['mean_mse']-rz['metrics']['mean_mse'],
                baseline_minus_prior=rb['metrics']['mean_mse']-rb['prior']['mean_mse']))
        comparison.append(dict(client=client,baseline=b['metrics'],zero=z['metrics'],paired=paired,
            baseline_best_step=b['best_step']))
    restart_summary=[]
    for client in sorted(clients):
        for method in ('baseline','zero_update'):
            group=[r for r in records if r['client']==client and r['method']==method]
            values=np.asarray([r['metrics']['mean_mse'] for r in group])
            restart_summary.append(dict(client=client,method=method,restarts=len(group),
                mse_mean=float(values.mean()),mse_std_population=float(values.std(ddof=0)),
                median_mse_mean=float(np.mean([r['metrics']['median_mse'] for r in group])),
                best_at_budget=sum(r['best_step']==signature[3] for r in group),
                note='restart variability within one client, not independent targets'))
    with (args.output/'restart_summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(restart_summary[0]))
        writer.writeheader()
        writer.writerows(restart_summary)
    dump(args.output/'report.json',dict(runs=[str(r) for r in args.runs],selected=selected,comparison=comparison,
        restart_summary=restart_summary,
        profiles=profiles,defense_capture=defense_rows,defense_run_seed=defense_seed,
        trial_count=len(records),independent_training_runs=1,client_count=3,
        gate='CONDITIONAL: execution validated; effectiveness requires interpretation, no automatic Phase4 pass'))
    root=Path(__file__).resolve().parents[1]
    snapshot=args.output/'source_snapshot'
    snapshot.mkdir()
    for relative in ('attacks/local_update.py','experiments/run_phase3_full_client.py',
                     'experiments/summarize_phase3_full.py','experiments/fraud_fl_common.py',
                     'data/load_creditcard.py','models/fraud_mlp.py','tests/test_phase3_updates.py'):
        destination=snapshot/relative
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(root/relative,destination)
    dump(args.output/'environment.json',dict(python=platform.python_version(),platform=platform.platform(),
        machine=platform.machine(),torch=torch.__version__,numpy=np.__version__,threads=torch.get_num_threads()))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for ax,(folder,cap,_) in zip(axes,captures):
        for method in ('baseline','zero_update'):
            for restart in range(signature[4]):
                saved=torch.load(folder/f'{method}_{restart}.pt',weights_only=False)
                ax.plot(np.minimum.accumulate(saved['history']),label=f'{method} r{restart}')
        ax.set_title(folder.name)
        ax.set_xlabel('Attack iteration')
        ax.set_ylabel('Own objective (not leakage)')
        ax.set_yscale('log')
        ax.legend(fontsize=6)
    fig.tight_layout();fig.savefig(args.output/'convergence.png',dpi=160);plt.close(fig)
    dump(args.output/'manifest.json',{str(f.relative_to(args.output)):checksum(f) for f in args.output.rglob('*') if f.is_file()})
    print(json.dumps(comparison,indent=2))


if __name__=='__main__':
    main()
