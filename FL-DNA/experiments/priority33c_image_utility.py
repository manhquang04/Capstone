"""P33c frozen image utility jobs and per-tensor Gaussian calibration."""
import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import priority31_image_utility_dp as p31
from experiments.priority33c_checkpoint_replay import finite, check, accuracy, sha
from dna_encoder.transform_defense import DNATransformConfig, transform_update_array
import numpy as np
import torch

OUT = ROOT / 'artifacts/priority33c'
ANNEX = ROOT / 'protocols/amendments/2026-10-03_priority33c_execution_annex.md'
SEEDS = list(range(51016, 51032))
GRID = [.0001, .0003, .001, .003, .01, .03, .1, .3]
EXTENSION = [1., 3., 10.]
V1 = {'dna_v1_conservative': (.08, .88, .45),
      'dna_v1_medium': (.10, .85, .40), 'dna_v1_stronger': (.12, .82, .35)}
METHODS = list(V1) + ['dna_v2_0p95']
torch.set_num_threads(1)


def read(path):
    return json.loads(Path(path).read_text())


def save(path, obj):
    path = Path(path)
    assert path.absolute().is_relative_to(OUT)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def log(event, **details):
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'runs.jsonl').open('a') as f:
        f.write(json.dumps(dict(time=time.time(), event=event, **details)) + '\n')


def progress(stage, done, total, started, failed=0, method='all'):
    elapsed = time.time()-started
    row = dict(stage=stage, dataset='CIFAR10', method=method, done=done, total=total,
               failed=failed, started_at=started, last_update=time.time(),
               eta_minutes=elapsed/done*(total-done)/60 if done else None)
    save(OUT / 'progress.json', row)
    with (OUT / 'progress.log').open('a') as f:
        f.write(json.dumps(row) + '\n')


def transform(state, method):
    if method not in V1:
        raise ValueError(method)
    m, k, s = V1[method]
    cfg = DNATransformConfig(block_size=256, mix_ratio=m, keep_ratio=k,
                             shrink_factor=s, seed=30001)
    result = {}
    for i, (name, value) in enumerate(state.items()):
        array, _ = transform_update_array(value.detach().cpu().numpy(), cfg, tensor_index=i)
        result[name] = torch.from_numpy(array.copy()).to(value)
    return result


def per_tensor_noise(tensors, clips, sigma, seed):
    if len(tensors) != len(clips) or sigma < 0:
        raise ValueError('incompatible per-tensor DP specification')
    generator = torch.Generator().manual_seed(seed)
    result = []
    for value, clip in zip(tensors, clips):
        finite(value, 'DP input')
        if clip <= 0 or not np.isfinite(clip):
            raise ValueError('invalid tensor clip')
        norm = float(value.norm())
        factor = min(1., clip/max(norm, 1e-12))
        noisy = value.detach().cpu()*factor + torch.randn(value.shape, generator=generator,
                dtype=value.dtype)*sigma*clip
        finite(noisy, 'DP transmitted tensor')
        result.append(noisy)
    return result


def prepare():
    target = OUT / 'image_clip.json'
    if target.exists():
        return read(target)
    x, y, _, _ = p31.utility_data()
    model, _ = p31.dp.inversefed.construct_model('LeNetZhu', seed=42)
    names = [name for name, _ in model.named_parameters()]
    norms = {name: [] for name in names}
    for seed in [51000, 51003]:
        order = torch.randperm(len(x), generator=torch.Generator().manual_seed(seed))
        for start in range(0, len(x), 256):
            index = order[start:start+256]
            model.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(x[index]), y[index])
            finite(loss, 'clip probe loss'); loss.backward()
            for name, parameter in model.named_parameters():
                finite(parameter.grad, 'clip probe gradient')
                norms[name].append(float(parameter.grad.norm()))
    result = dict(names=names, clips=[float(np.quantile(norms[n], .95)) for n in names],
                  norms=norms, C=read(p31.OUT / 'clip.json')['C'],
                  convention='P31 fixed initial minibatches, per-tensor p95',
                  seed=[51000, 51003], split_sha256=sha(p31.OUT / 'split.json'))
    save(target, result)
    return result


def job_path(job):
    return OUT / 'image_utility' / job['method'] / f"seed_{job['seed']}.json"


def fingerprints():
    paths = [Path(__file__), ANNEX, ROOT / 'experiments/priority31_image_utility_dp.py',
             ROOT / 'experiments/priority33c_checkpoint_replay.py',
             ROOT / 'dna_encoder/transform_defense.py',
             ROOT / 'dna_encoder/transform_defense_v2.py', p31.OUT / 'split.json',
             OUT / 'image_clip.json']
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}


def train(job):
    torch.set_num_threads(1)
    path = job_path(job)
    hashes = fingerprints()
    if path.exists():
        result = read(path)
        assert result['job'] == job and result['source_hashes'] == hashes
        return result
    started = time.time()
    log('image_utility_start', job=job)
    result = dict(job=job, source_hashes=hashes, device='cpu', threads=1)
    try:
        x, y, vx, vy = p31.utility_data()
        finite(x, 'training data'); finite(vx, 'validation data')
        model, _ = p31.dp.inversefed.construct_model('LeNetZhu', seed=42)
        assert not list(model.buffers())
        optimizer = torch.optim.SGD(model.parameters(), lr=.1, momentum=0)
        generator = torch.Generator().manual_seed(job['seed'])
        clip = read(OUT / 'image_clip.json')
        step = 0
        for epoch in range(1, 101):
            order = torch.randperm(len(x), generator=generator)
            for start in range(0, len(x), 256):
                batch = order[start:start+256]
                optimizer.zero_grad()
                logits = model(x[batch]); finite(logits, 'training logits')
                loss = torch.nn.functional.cross_entropy(logits, y[batch]); finite(loss, 'loss')
                loss.backward()
                state = {name: p.grad.detach().clone() for name, p in model.named_parameters()}
                if job['method'] in V1:
                    state = transform(state, job['method'])
                elif job['method'].startswith('per_tensor_dp_'):
                    changed = per_tensor_noise(list(state.values()), clip['clips'], job['sigma'],
                                                job['seed']*100000+step)
                    state = dict(zip(state, changed))
                else:
                    raise ValueError('unknown training method ' + job['method'])
                for name, p in model.named_parameters():
                    finite(state[name], 'transmitted gradient ' + name)
                    p.grad = state[name]
                optimizer.step(); check(model); step += 1
            if epoch % 10 == 0:
                print('utility', job['method'], job['seed'], epoch, flush=True)
        validation = accuracy(model, vx, vy)
        data = p31.dp.CIFAR10(str(ROOT / 'datasets/cifar10'), train=False, download=False)
        tx, ty = p31.tensors(data, list(range(10000)))
        testing = accuracy(model, tx, ty)
        result.update(status='passed', validation_accuracy=validation, test_accuracy=testing, steps=step)
    except FloatingPointError as error:
        result.update(status='GATE_FAILED', error=repr(error))
    result['elapsed_seconds'] = time.time()-started
    save(path, result)
    log('image_utility_finished', job=job, status=result['status'], sha256=sha(path))
    return result


def run_jobs(jobs, workers):
    started = time.time()
    failed = 0
    progress('image_utility', 0, len(jobs), started)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(train, job): job for job in jobs}
        for done, future in enumerate(as_completed(futures), 1):
            try:
                result = future.result()
            except BaseException as error:
                save(OUT / 'REQUIRES_DIRECTION.json', dict(job=futures[future], error=repr(error),
                     reason='infrastructure or harness error; do not silently replace/rerun'))
                log('image_utility_error', job=futures[future], error=repr(error))
                raise
            failed += result['status'] != 'passed'
            progress('image_utility', done, len(jobs), started, failed, result['job']['method'])
            print('finished', done, '/', len(jobs), 'failed', failed, flush=True)


def value(method, seed):
    if method in ['baseline', 'dna_v1_conservative', 'dna_v2_0p95'] or method.startswith('dp_'):
        result = read(p31.OUT / 'utility' / method / f'seed_{seed}.json')
    else:
        result = read(OUT / 'image_utility' / method / f'seed_{seed}.json')
        if result['status'] != 'passed':
            raise FloatingPointError('utility target/grid failed ' + method)
    result_value = float(result['validation_accuracy'])
    if not np.isfinite(result_value):
        raise FloatingPointError('nonfinite validation score')
    return result_value


def select(grid):
    baseline = [value('baseline', seed) for seed in SEEDS]
    matches = {}
    for method in METHODS:
        for variant in ['single', 'per_tensor']:
            key = method + '::' + variant
            try:
                target = float(np.mean([value(method, seed)-baseline[i] for i, seed in enumerate(SEEDS)]))
                means = {}
                for sigma in grid:
                    tag = f'dp_{sigma:g}' if variant == 'single' else f'per_tensor_dp_{sigma:g}'
                    # P31 conditional extension may be absent; single matching remains its frozen grid.
                    if variant == 'single' and not (p31.OUT / 'utility' / tag).exists():
                        continue
                    means[sigma] = float(np.mean([value(tag, seed)-baseline[i] for i, seed in enumerate(SEEDS)]))
                eligible = [sigma for sigma in means if means[sigma] >= target-.005]
                chosen = max(eligible) if eligible else None
                ordered = sorted(means)
                next_sigma = ordered[ordered.index(chosen)+1] if chosen is not None and chosen != ordered[-1] else None
                bracket = next_sigma is not None and means[next_sigma] < target-.005
                result = dict(status='matched' if bracket else 'NOT_BRACKETED', sigma=chosen,
                    bracketed=bracket, target_delta=target, threshold=target-.005,
                    next_sigma=next_sigma, grid_means=means)
                if bracket:
                    ratio = 1. if variant == 'single' else float(np.sqrt(len(read(OUT / 'image_clip.json')['clips'])))
                    account = p31.dp.epsilon_from_rdp(noise_multiplier=chosen, sensitivity_ratio=ratio,
                        delta=1e-5, compositions=1, orders=p31.dp.alpha_grid())
                    result.update(epsilon=account['epsilon'], sensitivity_ratio=ratio)
                matches[key] = result
            except FloatingPointError as error:
                matches[key] = dict(status='NOT_ASSESSABLE', reason=str(error), bracketed=False)
    return dict(matches=matches, grid=grid, baseline_mean=float(np.mean(baseline)),
                baseline_passed=float(np.mean(baseline)) >= .4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.workers != 4:
        raise ValueError('frozen four workers')
    log('image_utility_invocation', argv=sys.argv)
    prepare()
    reused = []
    for method in ['baseline', 'dna_v1_conservative', 'dna_v2_0p95'] + [f'dp_{s:g}' for s in GRID]:
        for seed in SEEDS:
            path = p31.OUT / 'utility' / method / f'seed_{seed}.json'
            doc = read(path)
            assert doc['job']['seed'] == seed and doc['job']['lr'] == .1 and doc['job']['epochs'] == 100
            reused.append(dict(path=str(path.relative_to(ROOT)), sha256=sha(path)))
    save(OUT / 'image_reused_utility.json', reused)
    jobs = [dict(method=m, seed=s, sigma=None) for m in ['dna_v1_medium', 'dna_v1_stronger'] for s in SEEDS]
    jobs += [dict(method=f'per_tensor_dp_{sigma:g}', seed=s, sigma=sigma) for sigma in GRID for s in SEEDS]
    run_jobs(jobs, args.workers)
    matched = select(GRID)
    if any(r['status'] == 'NOT_BRACKETED' for k, r in matched['matches'].items() if k.endswith('::per_tensor')):
        run_jobs([dict(method=f'per_tensor_dp_{sigma:g}', seed=s, sigma=sigma) for sigma in EXTENSION for s in SEEDS], args.workers)
        matched = select(GRID + EXTENSION)
    save(OUT / 'image_calibration.json', matched)
    log('image_utility_stage_complete', calibration_sha256=sha(OUT / 'image_calibration.json'))


if __name__ == '__main__':
    main()
