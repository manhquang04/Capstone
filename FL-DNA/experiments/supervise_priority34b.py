"""Frozen P34B manifest and one detached, resumable, fail-closed supervisor."""
from __future__ import annotations
import argparse
import json
import multiprocessing as mp
import os
import platform
import random
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments import priority34b_core as b
OUT=b.OUT


def freeze():
    if (OUT/'execution_freeze.json').exists():
        raise ValueError('never overwrite frozen protocol')
    complete=b.read(ROOT/'artifacts/priority34a/COMPLETE.json')
    if complete['status']!='PASS' or complete['jobs']!=189 or complete['live_workers']:
        raise ValueError('P34A not verified complete')
    parent=b.a.verify_freeze()
    inputs=dict(parent['inputs'])
    for relative in ['artifacts/priority34a/execution_freeze.json','artifacts/priority34a/COMPLETE.json',
                     'artifacts/priority34a/sha256_manifest_final.json','reports/priority34a_report.md',
                     'artifacts/priority31_image_utility_dp/split.json','artifacts/priority31_image_utility_dp/selection.json',
                     'artifacts/priority33c/image_targets.json']:
        inputs[relative]=b.p.sha(ROOT/relative)
    for dataset in b.p.DATASETS:
        for method in b.p.METHODS:
            for seed in b.SEEDS:
                path=ROOT/f'artifacts/priority34a/jobs/{dataset}/{method}/{seed}.json'
                doc=b.read(path)
                if doc['status']!='COMPLETED' or not doc['no_bn_transmitted']:
                    raise ValueError('P34A reused job invalid')
                inputs[str(path.relative_to(ROOT))]=b.p.sha(path)
                if method=='baseline':
                    for relative,digest in doc['artifact_sha256'].items():
                        inputs['artifacts/priority34a/'+relative]=digest
    for path in (ROOT/'datasets/cifar10').rglob('*'):
        if path.is_file():
            inputs[str(path.relative_to(ROOT))]=b.p.sha(path)
    from experiments.priority34b_image_imports import recovery as old
    excluded=set()
    original=b.read(ROOT/'artifacts/priority33c/image_targets.json')
    excluded.update(original['excluded'])
    for key in ['development','C1','C2_trained']:
        excluded.update(original[key])
    for batch in original['C2_batch4']:
        excluded.update(batch)
    evidence=[]
    for folder in ['artifacts','results']:
        for path in (ROOT/folder).rglob('*.json'):
            if path.is_relative_to(OUT) or not any(key in str(path).lower() for key in ['cifar','image','priority29','priority30']):
                continue
            before=len(excluded)
            old.collect_indices(b.read(path),excluded)
            if len(excluded)>before:
                evidence.append(dict(path=str(path.relative_to(ROOT)),sha256=b.p.sha(path)))
                inputs[str(path.relative_to(ROOT))]=b.p.sha(path)
    available=b.np.array(sorted(set(range(10000))-excluded))
    selected=b.np.random.default_rng(344000).permutation(available)[:39].tolist()
    if len(selected)!=39 or len(set(selected))!=39 or set(selected)&excluded:
        raise ValueError('fresh target source gate')
    b.save(OUT/'image_targets.json',dict(seed=344000,source='CIFAR10/test',targets=selected,
        excluded=sorted(excluded),historical_overlap=0,evidence=evidence))
    source_names=['experiments/priority34b_core.py','experiments/priority34b_images.py','experiments/priority34b_image_imports.py',
                  'experiments/priority34b_analysis.py','experiments/supervise_priority34b.py',
                  'experiments/verify_priority34b.py','tests/test_priority34b.py',
                  str(b.PROTOCOL.relative_to(ROOT)),'experiments/priority31_image_utility_dp.py',
                  'experiments/priority33c_image_recovery.py','experiments/priority33c_image_utility.py',
                  'experiments/priority33c_checkpoint_replay.py',
                  'experiments/priority30_native_defenses/run_audit.py',
                  'experiments/priority30_native_defenses/run_e3_dp_comparators.py',
                  'experiments/priority30_native_defenses/native_adapters.py']+list(parent['sources'])
    source_names += [str(path.relative_to(ROOT)) for path in (ROOT/'external_defenses/invertinggradients').rglob('*.py')]
    for module in list(sys.modules.values()):
        filename=getattr(module,'__file__',None)
        if filename:
            path=Path(filename).resolve()
            if path.is_file() and path.is_relative_to(ROOT) and path.suffix=='.py':
                relative=path.relative_to(ROOT)
                if relative.parts[0] in ['experiments','models','data','privacy','dna_encoder','external_defenses']:
                    source_names.append(str(relative))
    source_names += ['experiments/priority34b_preflight.py','experiments/priority34b_launch.py','protocols/amendments/2026-10-03_priority34b_preflight_annex.md']
    sources={relative:b.p.sha(ROOT/relative) for relative in sorted(set(source_names))}
    jobs=[]
    for dataset in b.p.DATASETS:
        for mechanism in b.MECHANISMS:
            for epsilon in b.EPSILONS:
                for seed in b.SEEDS:
                    method=f'dp_{mechanism}_eps{epsilon}'
                    jobs.append(dict(stage='tabular',dataset=dataset,method=method,mechanism=mechanism,
                        epsilon=epsilon,seed=seed,result=f'jobs/{dataset}/{method}/{seed}.json'))
    for method,mechanism,epsilon in [('baseline',None,None)]+[(f'dp_{m}_eps{e}',m,e) for m in b.MECHANISMS for e in b.EPSILONS]:
        for seed in b.SEEDS:
            jobs.append(dict(stage='image_utility',dataset='cifar10',method=method,mechanism=mechanism,
                epsilon=epsilon,seed=seed,result=f'jobs/cifar10/{method}/{seed}.json'))
    for tid,index in enumerate(selected):
        for method,mechanism in [('unprotected',None),('dna_v1_conservative',None),('dna_v2_0p95',None),('dp_global_eps10','global'),('dp_per_tensor_eps10','per_tensor')]:
            jobs.append(dict(stage='recovery',dataset='cifar10',method=method,mechanism=mechanism,
                target_id=tid,source_id=index,attack_seed=344100+tid,noise_seed=344200+tid,
                result=f'recovery/{method}/target_{tid:03d}.json'))
    assert len(jobs)==470
    secrets={job['result']:int.from_bytes(os.urandom(8),'big')&((1<<63)-1)
             for job in jobs if job['method'].startswith('dp_')}
    private_path=OUT/'private_noise_seeds.json'
    if private_path.exists():
        raise ValueError('never overwrite privacy noise keys')
    b.save(private_path,secrets); private_path.chmod(0o600)
    inputs[str(private_path.relative_to(ROOT))]=b.p.sha(private_path)
    for relative,digest in inputs.items():
        if b.p.sha(ROOT/relative)!=digest:
            raise ValueError('input hash mismatch '+relative)
    doc=dict(status='FROZEN',frozen_at=b.p.now(),inputs=inputs,sources=sources,jobs=jobs,
             targets_sha256=b.p.sha(OUT/'image_targets.json'),workers=4,accounting=[b.account(e) for e in b.EPSILONS],
             noise_randomness='independent OS-entropy keys per DP job; private file mode0600, never derived from public seeds',
             environment=dict(python=sys.version,torch=b.torch.__version__,numpy=b.np.__version__,
                 platform=platform.platform(),device='cpu',torch_intraop=1,torch_interop=1))
    b.save(OUT/'execution_freeze.json',doc); b.checklist(2,'complete')
    b.event('frozen',jobs=len(jobs),sources=len(sources),inputs=len(inputs))


def run_job(job):
    if job['stage']=='tabular':
        return b.tabular_job(job)
    from experiments import priority34b_images as image
    return image.utility_job(job) if job['stage']=='image_utility' else image.recovery_job(job)


def supervise():
    import fcntl
    with (OUT/'supervisor.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        b.save(OUT/'supervisor.lock.json',dict(pid=os.getpid(),started_at=b.p.now(),executable=sys.executable))
        frozen=b.verified_freeze()
        if b.p.sha(OUT/'image_targets.json')!=frozen['targets_sha256']:
            raise ValueError('target manifest changed')
        for part,stages,item in [('training',{'tabular','image_utility'},3),('recovery',{'recovery'},4)]:
            all_jobs=[job for job in frozen['jobs'] if job['stage'] in stages]
            missing=[]; done=0
            for job in all_jobs:
                if b.valid(OUT/job['result'],job):
                    done+=1
                elif (OUT/job['result']).with_suffix('.failure.json').exists():
                    raise ValueError('scientific failure requires direction, not automatic replay')
                else:
                    missing.append(job)
            random.Random(344034).shuffle(missing)
            start=time.time(); failures=[]
            b.checklist(item,'in_progress'); b.progress(part,done,len(all_jobs),start=start)
            b.event('stage_started',part=part,done=done,remaining=len(missing))
            with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn')) as pool:
                futures={pool.submit(run_job,job):job for job in missing}
                for future in as_completed(futures):
                    job=futures[future]
                    try:
                        receipt=future.result(); done+=1
                        b.event('job_complete',job=job,**receipt)
                    except Exception:
                        failure=dict(job=job,traceback=traceback.format_exc(),at=b.p.now())
                        b.save((OUT/job['result']).with_suffix('.failure.json'),failure)
                        failures.append(failure); b.event('job_failure',job=job,error=failure['traceback'])
                    b.progress(part,done,len(all_jobs),len(failures),job,start)
            if failures:
                b.save(OUT/'REQUIRES_DIRECTION.json',dict(stage=part,done=done,total=len(all_jobs),failed=failures))
                b.checklist(item,'requires_direction'); b.progress('requires_direction',done,len(all_jobs),len(failures))
                return
            b.checklist(item,'complete')
        b.progress('analysis',470,470)
        from experiments.priority34b_analysis import finish
        finish()


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--freeze',action='store_true'); parser.add_argument('--supervise',action='store_true')
    args=parser.parse_args()
    try:
        if args.freeze:
            freeze()
        elif args.supervise:
            supervise()
        else:
            parser.error('choose --freeze/--supervise')
    except Exception:
        b.save(OUT/('FREEZE_FAILURE.json' if args.freeze else 'SUPERVISOR_INTERRUPTION.json'),
               dict(at=b.p.now(),traceback=traceback.format_exc()))
        raise


if __name__=='__main__':
    main()
