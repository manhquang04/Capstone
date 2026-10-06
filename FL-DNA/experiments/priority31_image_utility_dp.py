"""Frozen P31 image utility calibration, reconstruction and extended Holm analysis.

Imports reference code read-only. Completed jobs are resumable; exceptions are
recorded and stop the stage rather than becoming a scientific result.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from experiments.priority30_native_defenses import run_e3_dp_comparators as dp
from experiments.priority30_native_defenses import native_adapters as adapters

torch.set_num_threads(1)
OUT = ROOT / 'artifacts/priority31_image_utility_dp'
OLD = ROOT / 'artifacts/priority30_native_defenses/audit'
SEEDS = list(range(51016, 51032))
SIGMAS = [.0001, .0003, .001, .003, .01, .03, .1, .3]
TRANSFORMS = ['dna_v1_conservative', 'dna_v2_0p95']
REPORT = ROOT / 'reports/priority31_image_utility_dp_report.md'


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def csv_write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def journal(event, **details):
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'runs.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(time=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                                    event=event, **details)) + '\n')


def sigma_id(sigma):
    return f'dp_{sigma:g}'


def split_ids(labels):
    train, other = train_test_split(np.arange(len(labels)), train_size=12000,
                                    stratify=labels, random_state=51001)
    val, unused = train_test_split(other, train_size=3000,
                                  stratify=np.asarray(labels)[other], random_state=51002)
    assert not set(train) & set(val)
    assert len(set(train) | set(val) | set(unused)) == len(labels)
    return train, val


def tensors(data, indices):
    raw = torch.from_numpy(data.data[np.asarray(indices)].copy()).permute(0, 3, 1, 2).float()/255
    dm = torch.tensor(dp.inversefed.consts.cifar10_mean)[None, :, None, None]
    ds = torch.tensor(dp.inversefed.consts.cifar10_std)[None, :, None, None]
    return (raw-dm)/ds, torch.tensor(np.asarray(data.targets)[indices], dtype=torch.long)


def utility_data():
    data = dp.CIFAR10(str(ROOT/'datasets/cifar10'), train=True, download=False)
    ids = read(OUT/'split.json')
    return (*tensors(data, ids['train_ids']), *tensors(data, ids['validation_ids']))


@torch.no_grad()
def accuracy(model, x, y):
    model.eval()
    correct = 0
    for start in range(0, len(x), 512):
        correct += int((model(x[start:start+512]).argmax(1) == y[start:start+512]).sum())
    model.train()
    return correct/len(x)


def train_job(job):
    torch.set_num_threads(1)
    path = Path(job['path'])
    if path.exists():
        prior = read(path)
        assert prior['job'] == job, f'Incompatible resume: {path}'
        journal('job_resume_skip', path=str(path))
        return dict(path=str(path), skipped=True)
    start = time.monotonic()
    journal('job_start', job=job)
    x, y, vx, vy = utility_data()
    model, _ = dp.inversefed.construct_model('LeNetZhu', seed=42)
    assert not list(model.buffers()), 'Unexpected BN or other buffers in LeNet-Zhu'
    optimizer = torch.optim.SGD(model.parameters(), lr=job['lr'], momentum=0)
    generator = torch.Generator().manual_seed(job['seed'])
    loss_fn = torch.nn.CrossEntropyLoss()
    checkpoints, clipped, total = {}, 0, 0
    for epoch in range(1, job['epochs']+1):
        order = torch.randperm(len(x), generator=generator)
        for start_idx in range(0, len(x), 256):
            batch = order[start_idx:start_idx+256]
            optimizer.zero_grad()
            loss_fn(model(x[batch]), y[batch]).backward()
            state = {name: p.grad.detach().clone() for name, p in model.named_parameters()}
            if job['method'].startswith('dp_'):
                clipped += int(float(dp.flatten(list(state.values())).norm()) > job['C'])
                changed = dp.add_clipped_noise(list(state.values()), job['C'], job['sigma'],
                                              job['seed']*100000+total)
                state = dict(zip(state, changed))
            elif job['method'] == TRANSFORMS[0]:
                state = adapters.dna_v1_gradient(state)
            elif job['method'] == TRANSFORMS[1]:
                state, _ = adapters.dna_v2_gradient(state)
            elif job['method'] != 'baseline':
                raise ValueError(job['method'])
            for name, parameter in model.named_parameters():
                parameter.grad = state[name]
            optimizer.step()
            total += 1
        if epoch in job['checkpoints']:
            checkpoints[str(epoch)] = accuracy(model, vx, vy)
            print(f"progress {job['method']} seed={job['seed']} lr={job['lr']} epoch={epoch} validation={checkpoints[str(epoch)]:.6f}", flush=True)
    result = dict(job=job, validation_accuracy=checkpoints[str(job['epochs'])],
                  checkpoints=checkpoints, clipping_rate=clipped/total,
                  steps=total, elapsed_seconds=time.monotonic()-start)
    if job['test']:
        test = dp.CIFAR10(str(ROOT/'datasets/cifar10'), train=False, download=False)
        tx, ty = tensors(test, list(range(10000)))
        result['test_accuracy'] = accuracy(model, tx, ty)
    save(path, result)
    journal('job_complete', path=str(path), elapsed=result['elapsed_seconds'])
    return dict(path=str(path), skipped=False)


def parallel(function, jobs, workers):
    start = time.monotonic()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(function, job): job for job in jobs}
        for done, future in enumerate(as_completed(pending), 1):
            try:
                result = future.result()
            except BaseException as error:
                journal('job_error', job=pending[future], error=repr(error))
                raise
            elapsed = time.monotonic()-start
            print(f'completed {done}/{len(jobs)} elapsed={elapsed:.1f}s ETA={elapsed/done*(len(jobs)-done):.1f}s {result}', flush=True)


def prepare():
    if (OUT/'split.json').exists():
        return
    data = dp.CIFAR10(str(ROOT/'datasets/cifar10'), train=True, download=False)
    train, val = split_ids(data.targets)
    save(OUT/'split.json', dict(train_ids=train.tolist(), validation_ids=val.tolist(),
                              overlap=0, train_source='CIFAR10/train', validation_source='CIFAR10/train'))
    x, y, _, _ = utility_data()
    model, _ = dp.inversefed.construct_model('LeNetZhu', seed=42)
    norms = []
    for seed in [51000, 51003]:
        order = torch.randperm(len(x), generator=torch.Generator().manual_seed(seed))
        for start in range(0, len(x), 256):
            idx = order[start:start+256]
            model.zero_grad()
            torch.nn.functional.cross_entropy(model(x[idx]), y[idx]).backward()
            norms.append(float(dp.flatten([p.grad for p in model.parameters()]).norm()))
    save(OUT/'clip.json', dict(C=float(np.quantile(norms, .95)), norms=norms,
                             convention='p95 minibatch gradient norms at fixed initial checkpoint; no training',
                             model_seed=42, order_seeds=[51000,51003]))


def job_spec(method, seed, lr, epochs, selection=False, sigma=None):
    name = f'lr_{lr:g}' if selection else method
    return dict(method=method, seed=seed, lr=lr, epochs=epochs,
                checkpoints=[30,60,100] if selection else [epochs], test=not selection,
                C=read(OUT/'clip.json')['C'], sigma=sigma,
                path=str(OUT/('selection_runs' if selection else 'utility')/name/f'seed_{seed}.json'))


def utility_table():
    rows = []
    for path in sorted((OUT/'utility').glob('*/seed_*.json')):
        result = read(path)
        seed, method = result['job']['seed'], result['job']['method']
        baseline = read(OUT/'utility/baseline'/f'seed_{seed}.json')
        rows.append(dict(method=method, seed=seed, sigma=result['job']['sigma'],
                         validation_accuracy=result['validation_accuracy'], test_accuracy=result['test_accuracy'],
                         paired_validation_delta=result['validation_accuracy']-baseline['validation_accuracy'],
                         paired_test_delta=result['test_accuracy']-baseline['test_accuracy'],
                         clipping_rate=result['clipping_rate']))
    csv_write(OUT/'utility_per_seed.csv', rows)
    return rows


def match_grid(rows, grid):
    grouped = {method: [r for r in rows if r['method']==method] for method in set(r['method'] for r in rows)}
    assert all(len(grouped[method])==16 for method in ['baseline',*TRANSFORMS,*map(sigma_id,grid)])
    baseline_mean = float(np.mean([r['validation_accuracy'] for r in grouped['baseline']]))
    matches = {}
    for transform in TRANSFORMS:
        target = float(np.mean([r['paired_validation_delta'] for r in grouped[transform]]))
        means = {sigma: float(np.mean([r['paired_validation_delta'] for r in grouped[sigma_id(sigma)]])) for sigma in grid}
        eligible = [sigma for sigma in grid if means[sigma] >= target-.005]
        chosen = max(eligible) if eligible else None
        next_sigma = grid[grid.index(chosen)+1] if chosen is not None and chosen != grid[-1] else None
        bracketed = next_sigma is not None and means[next_sigma] < target-.005
        matches[transform] = dict(sigma=chosen, bracketed=bracketed, transform_delta=target,
                                  threshold=target-.005, next_sigma=next_sigma, grid_means=means)
    return dict(baseline_validation_mean=baseline_mean, baseline_gate=baseline_mean>=.4 and baseline_mean>=.35,
                matches=matches)


def utility_stage(workers):
    prepare()
    selection = OUT/'selection.json'
    if not selection.exists():
        jobs = [job_spec('baseline', seed, lr, 100, True) for seed in [51000,51003] for lr in [.03,.1,.3]]
        parallel(train_job, jobs, workers)
        candidates=[]
        for lr in [.03,.1,.3]:
            results=[read(OUT/'selection_runs'/f'lr_{lr:g}'/f'seed_{seed}.json') for seed in [51000,51003]]
            for epoch in [30,60,100]:
                candidates.append(dict(lr=lr, epochs=epoch, validation_mean=float(np.mean([r['checkpoints'][str(epoch)] for r in results]))))
        best=sorted(candidates, key=lambda r:(-r['validation_mean'],r['epochs'],r['lr']))[0]
        save(selection, dict(selected=best,candidates=candidates, test_used=False))
    selected=read(selection)['selected']
    jobs=[job_spec('baseline',seed,selected['lr'],selected['epochs']) for seed in SEEDS]
    parallel(train_job,jobs,workers)
    baseline_mean=float(np.mean([read(job['path'])['validation_accuracy'] for job in jobs]))
    save(OUT/'baseline_gate.json',dict(mean_validation_accuracy=baseline_mean, threshold=.4, passed=baseline_mean>=.4))
    if baseline_mean<.4:
        save(OUT/'calibration.json',dict(status='NOT_ASSESSABLE',reason='baseline validation gate failed',baseline_validation_mean=baseline_mean))
        utility_table()
        return
    jobs=[job_spec(method,seed,selected['lr'],selected['epochs']) for method in TRANSFORMS for seed in SEEDS]
    jobs += [job_spec(sigma_id(sigma),seed,selected['lr'],selected['epochs'],sigma=sigma) for sigma in SIGMAS for seed in SEEDS]
    parallel(train_job,jobs,workers)
    rows=utility_table()
    matching=match_grid(rows,SIGMAS)
    grid=SIGMAS.copy()
    if any(not m['bracketed'] for m in matching['matches'].values()):
        extension=[1.,3.,10.]
        parallel(train_job,[job_spec(sigma_id(sigma),seed,selected['lr'],selected['epochs'],sigma=sigma) for sigma in extension for seed in SEEDS],workers)
        grid+=extension
        rows=utility_table()
        matching=match_grid(rows,grid)
    matching.update(status='calibrated',grid=grid,C=read(OUT/'clip.json')['C'])
    for m in matching['matches'].values():
        if m['bracketed']:
            sigma=m['sigma']
            orders=dp.alpha_grid()
            accounting=dp.epsilon_from_rdp(noise_multiplier=sigma, sensitivity_ratio=1.,
                                          delta=1e-5, compositions=1, orders=orders)
            m.update(epsilon_one_release=accounting['epsilon'],best_alpha=accounting['alpha'])
    save(OUT/'calibration.json',matching)


def attack_job(job):
    torch.set_num_threads(1)
    path=Path(job['path'])
    if path.exists():
        assert read(path)['job']==job
        journal('attack_resume_skip', path=str(path))
        return dict(path=str(path),skipped=True)
    journal('attack_start',job=job)
    start=time.monotonic()
    raw,label,_,model,gradient=dp.image_base_gradient(job['index'])
    observed=dp.add_clipped_noise(gradient,job['C'],job['sigma'],job['noise_seed'])
    dm=torch.tensor(dp.inversefed.consts.cifar10_mean)[:,None,None]
    ds=torch.tensor(dp.inversefed.consts.cifar10_std)[:,None,None]
    torch.manual_seed(30600+job['target_id'])
    np.random.seed(30600+job['target_id'])
    attack=dp.inversefed.GradientReconstructor(model,(dm,ds),dp.IMAGE_CONFIG.copy(),num_images=1)
    reconstructed,stats=attack.reconstruct(observed,label,img_shape=(3,32,32))
    recon=reconstructed*ds+dm
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez(path.parent/'arrays.npz',truth=raw.numpy(),reconstruction=recon.detach().numpy())
    save(path,dict(status='ran',job=job,target_id=job['target_id'],cifar10_index=job['index'],
                   metrics=dp.image_metrics(raw,recon),objective=float(dict(stats)['opt']),
                   elapsed_seconds=time.monotonic()-start))
    journal('attack_complete',path=str(path))
    return dict(path=str(path),skipped=False)


def reconstruction_stage(workers):
    calibration=read(OUT/'calibration.json')
    if calibration['status']=='NOT_ASSESSABLE':
        return
    jobs=[]
    for defense,m in calibration['matches'].items():
        if not m['bracketed']:
            continue
        for tid,index in enumerate(dp.image_targets('S1',39)):
            for stage in ['S1','S1c']:
                original=read(OLD/stage/'image'/('E1' if stage=='S1' else 'E2')/defense/f'target_{tid:03d}/result.json')
                assert original['target_id']==tid and original['cifar10_index']==index
            control=read(OLD/'S0u/image'/f'target_{tid:03d}/result.json')
            assert control['cifar10_index']==index
            jobs.append(dict(defense=defense,target_id=tid,index=index,C=calibration['C'],sigma=m['sigma'],
                             noise_seed=510000+101*tid+dp.DEFENSE_SEED_OFFSET[defense],
                             path=str(OUT/'reconstruction'/defense/f'target_{tid:03d}/result.json')))
    parallel(attack_job,jobs,workers)


def analysis():
    # SSIM export includes retained invalid/superseded stages, explicitly labeled.
    descriptive=[]
    for stage in ['S1','S1b','S1c','S1d','S3']:
        groups={}
        for path in (OLD/stage/'image').glob('**/result.json'):
            result=read(path)
            if result.get('status')!='ran' or 'ssim' not in result.get('metrics',{}):
                continue
            group=(result.get('eval','unknown'),result.get('defense','unknown'),result.get('comparator','defense'))
            groups.setdefault(group,[]).append(result['metrics'])
        for (evaluation,defense,comparator),metrics in groups.items():
            active=(stage=='S1' and evaluation=='E1' and defense!='count_sketch') or (stage=='S1c' and evaluation=='E2' and defense not in ['count_sketch','ats']) or (stage=='S1d' and defense=='count_sketch') or (stage=='S3' and not (defense=='precode' and comparator=='distortion_matched_clipped'))
            descriptive.append(dict(stage=stage,evaluation=evaluation,defense=defense,arm=comparator,n=len(metrics),
                                    median_ssim=float(np.median([r['ssim'] for r in metrics])),
                                    median_psnr=float(np.median([r['psnr_db'] for r in metrics])),
                                    status='active' if active else 'superseded/not_applicable'))
    csv_write(OUT/'all_image_ssim.csv',descriptive)
    oldrows=list(csv.DictReader((OLD/'e3_repair_20261001/S6/tests_with_effects_and_references.csv').open()))
    hypotheses=[]
    for row in oldrows:
        for direction,field in [('defense_greater','p_defense_greater_error'),('comparator_greater','p_comparator_greater_error')]:
            if row.get(field) and math.isfinite(float(row[field])):
                hypotheses.append(dict(hypothesis_id=row['hypothesis_id']+'::'+direction,p_value=float(row[field]),origin='P30 repaired'))
    assert len(hypotheses)==111,len(hypotheses)
    summaries=[]
    for defense in TRANSFORMS:
        paired=[]
        for tid in range(39):
            path=OUT/'reconstruction'/defense/f'target_{tid:03d}/result.json'
            if not path.exists():
                break
            defended=read(OLD/'S1c/image/E2'/defense/f'target_{tid:03d}/result.json')
            comparator=read(path)
            assert defended['cifar10_index']==comparator['cifar10_index']
            a,b=defended['metrics'],comparator['metrics']
            paired.append(dict(target_id=tid,cifar10_index=defended['cifar10_index'],
                               dna_mse=a['mse'],dp_mse=b['mse'],dna_psnr=a['psnr_db'],dp_psnr=b['psnr_db'],
                               dna_ssim=a['ssim'],dp_ssim=b['ssim'],D_mse=a['mse']-b['mse'],D_psnr=a['psnr_db']-b['psnr_db']))
        if len(paired)!=39:
            continue
        csv_write(OUT/f'paired_{defense}.csv',paired)
        wins=sum(r['D_mse']>0 for r in paired)
        losses=sum(r['D_mse']<0 for r in paired)
        differences=sorted(r['D_psnr'] for r in paired)
        summary=dict(defense=defense,n=39,dna_wins=wins,dp_wins=losses,ties=39-wins-losses,
                     p_dna_greater=dp.exact_p(wins,losses),p_dp_greater=dp.exact_p(losses,wins),
                     median_dna_psnr=float(np.median([r['dna_psnr'] for r in paired])),
                     median_dp_psnr=float(np.median([r['dp_psnr'] for r in paired])),
                     median_dna_ssim=float(np.median([r['dna_ssim'] for r in paired])),
                     median_dp_ssim=float(np.median([r['dp_ssim'] for r in paired])),
                     median_paired_psnr=float(np.median(differences)),rank13=differences[12],rank27=differences[26])
        summaries.append(summary)
        for direction in ['dna_greater','dp_greater']:
            hypotheses.append(dict(hypothesis_id=f'P31::{defense}::{direction}',p_value=summary['p_'+direction],origin='P31'))
    dp.holm(hypotheses)
    csv_write(OUT/'expanded_holm.csv',hypotheses)
    csv_write(OUT/'comparison_summary.csv',summaries)
    save(OUT/'analysis_summary.json',dict(comparisons=summaries,holm_family_size=len(hypotheses),
                                         rank_interval_coverage=1-2*sum(math.comb(39,k) for k in range(13))/2**39))
    report()


def report():
    lines=['# Priority 31 — CIFAR-10 image utility-matched DP','',
           'Status: IN PROGRESS unless explicitly completed below. No earlier results are replaced.', '',
           'Frozen amendment: protocols/amendments/2026-10-01_priority31_image_utility_dp.md.',
           'No Latex/ or external_defenses/ edits. Each process has torch threads=1.', '',
           'Utility is development scale: 12,000 train / 3,000 validation; initialization 42 is shared across paired replicates. Test accuracy is descriptive only.', '']
    for name in ['clip','selection','baseline_gate','calibration','analysis_summary']:
        path=OUT/f'{name}.json'
        if path.exists():
            lines += [f'## {name}','', '```json',json.dumps(read(path),indent=2),'```',''] if name!='clip' else [f'Clip C = {read(path)["C"]:.12g}; fixed initial-gradient p95, 94 minibatches.','']
    lines += ['## Commands and disclosure','', '```sh',
              'PYTHONDONTWRITEBYTECODE=1 external_defenses/.venv/bin/python -u experiments/priority31_image_utility_dp.py --stage all --workers 8',
              '```','', 'All invocations, jobs, errors and resume skips are tracked in artifacts/priority31_image_utility_dp/runs.jsonl.',
              'Utility checkpoints are validation-only. Baseline failure or failed bracketing means NOT_ASSESSABLE, not a privacy result.', '',
              'Exact per-target paired MSE tests and old raw p-values enter expanded_holm.csv; no ground-truth restart selection.',
              'PSNR order-statistic interval uses one-based ranks 13/27 (coverage in analysis_summary.json), not a refitted interval.', '',
              'File SHA-256 manifest: artifacts/priority31_image_utility_dp/sha256_manifest.csv. The report hash is stored separately to avoid self-reference.','']
    REPORT.write_text('\n'.join(lines))


def manifest():
    files=[ROOT/'experiments/priority31_image_utility_dp.py',ROOT/'tests/test_priority31_image_utility_dp.py',ROOT/'protocols/amendments/2026-10-01_priority31_image_utility_dp.md',REPORT]
    files += [p for p in OUT.rglob('*') if p.is_file() and p.name not in ['sha256_manifest.csv'] and not p.name.endswith('.tmp')]
    csv_write(OUT/'sha256_manifest.csv',[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in files if p.exists()])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=['all','utility','reconstruction','analysis'],default='all')
    parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args()
    journal('invocation',argv=sys.argv,workers=args.workers,threads=torch.get_num_threads())
    try:
        if args.stage in ['all','utility']:
            utility_stage(args.workers)
            report()
            manifest()
        if args.stage in ['all','reconstruction']:
            reconstruction_stage(args.workers)
        if args.stage in ['all','analysis']:
            analysis()
        journal('invocation_complete',stage=args.stage)
        manifest()
    except BaseException as error:
        journal('invocation_error',error=repr(error))
        report()
        manifest()
        raise


if __name__=='__main__':
    main()
