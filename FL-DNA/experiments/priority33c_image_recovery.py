"""P33c native image recovery: frozen objectives, payload-only, no fallbacks."""
import argparse
import csv
import json
import math
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import priority33c_image_utility as utility
from experiments.priority33c_checkpoint_replay import finite, check, sha
from experiments.priority30_native_defenses import run_audit as native
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

OUT = utility.OUT
torch.set_num_threads(1)


def block_debias(tensors, mix):
    if not 0 <= mix < 1:
        raise ValueError('invalid debias mix')
    result = []
    for value in tensors:
        flat = value.reshape(-1)
        blocks = [(flat[start:start+256]-mix*flat[start:start+256].mean())/(1.-mix)
                  for start in range(0, len(flat), 256)]
        result.append(torch.cat(blocks).reshape_as(value))
    return result


class ObservableReconstructor(native.AdaptiveImageReconstructor):
    MODES = {'plain', 'v1_debias', 'v2_sketch'}

    def __init__(self, model, normalizer, config, receipt, mode, mix=0, plans=None, count=1):
        if mode not in self.MODES:
            raise ValueError('unknown canonical mode ' + mode)
        self.receipt = receipt
        self.mode = mode
        self.mix = mix
        self.plans = plans
        super().__init__(model, normalizer, config, num_images=count,
                         adaptive_mode='plain', adaptive_aux={})

    def reconstruct(self, input_data, labels, *args, **kwargs):
        if input_data is not self.receipt:
            raise ValueError('reconstruct input is not exactly the transmitted payload object')
        for value in input_data:
            finite(value, 'server receipt')
        result, stats = super().reconstruct(input_data, labels, *args, **kwargs)
        finite(result, 'native reconstruction')
        if not math.isfinite(float(stats['opt'])):
            raise FloatingPointError('nonfinite native objective')
        return result, stats

    def _adaptive_loss(self, gradient, input_gradient):
        if input_gradient is not self.receipt:
            raise ValueError('closure lost transmitted payload identity')
        for value in gradient:
            finite(value, 'candidate gradient')
        if self.mode == 'plain':
            loss = native.reconstruction_costs([gradient], input_gradient,
                   cost_fn=self.config['cost_fn'], indices=self.config['indices'],
                   weights=self.config['weights'])
        elif self.mode == 'v1_debias':
            observed = block_debias(input_gradient, self.mix)
            loss = native.reconstruction_costs([gradient], observed,
                   cost_fn=self.config['cost_fn'], indices=self.config['indices'],
                   weights=self.config['weights'])
        elif self.mode == 'v2_sketch':
            if self.plans is None or len(self.plans) != len(gradient):
                raise ValueError('missing sketch projection key')
            projected = [g.reshape(-1) if p.get('identity') else native.v2_project_with_plan(g, p)
                         for g, p in zip(gradient, self.plans)]
            loss = native.reconstruction_costs([projected], input_gradient,
                   cost_fn=self.config['cost_fn'], indices=self.config['indices'],
                   weights=self.config['weights'])
        else:
            raise ValueError(self.mode)
        finite(loss, 'adaptive matching loss')
        return loss


def collect_indices(obj, result):
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in {'cifar10_index', 'decoy_cifar10_index'} and isinstance(value, int) and 0 <= value < 10000:
                result.add(value)
            elif key in {'target_indices', 'image_source_ids'} and isinstance(value, list):
                result.update(v for v in value if isinstance(v, int) and 0 <= v < 10000)
            else:
                collect_indices(value, result)
    elif isinstance(obj, list):
        for item in obj:
            collect_indices(item, result)


def prepare_targets():
    path = OUT / 'image_targets.json'
    if path.exists():
        return utility.read(path)
    excluded = set()
    evidence = []
    for base in ['artifacts', 'results']:
        for file in (ROOT / base).rglob('*.json'):
            if file.is_relative_to(OUT):
                continue
            if not any(word in str(file).lower() for word in ['image', 'cifar', 'priority29', 'priority30']):
                continue
            before = len(excluded)
            collect_indices(utility.read(file), excluded)
            if len(excluded) != before:
                evidence.append(dict(path=str(file.relative_to(ROOT)), sha256=sha(file)))
        for file in (ROOT / base).rglob('*.csv'):
            if file.is_relative_to(OUT) or not any(word in str(file).lower() for word in ['image', 'cifar', 'priority29', 'priority30']):
                continue
            with file.open() as stream:
                rows = csv.DictReader(stream)
                keys = [k for k in (rows.fieldnames or []) if k in ['cifar10_index', 'decoy_cifar10_index']]
                for row in rows:
                    for key in keys:
                        if row[key]:
                            excluded.add(int(float(row[key])))
            if keys:
                evidence.append(dict(path=str(file.relative_to(ROOT)), sha256=sha(file)))
    # Conservative exclusion covers all P29/P30 possible prior stage budgets.
    for seed, count in [(29000801, 1000), (30400, 3100)]:
        population = list(range(10000))
        random.Random(seed).shuffle(population)
        excluded.update(population[:count])
    available = np.array(sorted(set(range(10000))-excluded))
    order = np.random.default_rng(333600).permutation(available)
    assert len(order) >= 24+39+96+24
    dev = order[:24].tolist()
    c1 = order[24:63].tolist()
    batch4 = order[63:159].reshape(24, 4).tolist()
    trained = order[159:183].tolist()
    selected = dev+c1+[j for row in batch4 for j in row]+trained
    assert len(set(selected)) == 183 and not set(selected)&excluded
    result = dict(split='CIFAR10/test', seed=333600, development=dev, C1=c1,
                  C2_batch4=batch4, C2_trained=trained, excluded=sorted(excluded),
                  evidence=evidence, overlap=0)
    utility.save(path, result)
    utility.log('image_targets_frozen', sha256=sha(path), excluded=len(excluded))
    return result


def distortion_calibration(targets):
    path = OUT / 'image_distortion.json'
    if path.exists():
        return utility.read(path)
    samples = []
    for i, index in enumerate(targets['development']):
        _, _, _, model, tensors = utility.p31.dp.image_base_gradient(index)
        check(model)
        state = dict(zip([n for n, _ in model.named_parameters()], tensors))
        gaussian = torch.Generator().manual_seed(333610+i)
        z = [torch.randn(t.shape, generator=gaussian) for t in tensors]
        samples.append(dict(norm=float(utility.p31.dp.flatten(tensors).norm()),
            gaussian_norm=float(utility.p31.dp.flatten(z).norm()),
            distortion={m: float(utility.p31.dp.flatten([utility.transform(state, m)[n]-state[n] for n in state]).norm())
                        for m in ['dna_v1_medium', 'dna_v1_stronger']}))
    clip = 1.01*max(r['norm'] for r in samples)
    result = dict(C=clip, samples=samples, matches={})
    for method in ['dna_v1_medium', 'dna_v1_stronger']:
        target = float(np.median([r['distortion'][method] for r in samples]))
        sigma = target/(clip*float(np.median([r['gaussian_norm'] for r in samples])))
        achieved = float(np.median([sigma*clip*r['gaussian_norm'] for r in samples]))
        distance = abs(achieved-target)/max(target, 1e-18)
        assert distance <= .05
        result['matches'][method] = dict(sigma=sigma, target_median_l2=target,
            achieved_median_l2=achieved, relative_distance=distance,
            epsilon=utility.p31.dp.epsilon_from_rdp(sigma, 1., 1e-5, 1, utility.p31.dp.alpha_grid())['epsilon'])
    utility.save(path, result)
    return result


def attack(job):
    torch.set_num_threads(1)
    folder = OUT / 'image_recovery' / job['setting'] / job['arm'] / f"target_{job['target_id']:03d}"
    path = folder / 'result.json'
    sources = [Path(__file__), ROOT / 'experiments/priority33c_image_utility.py', utility.ANNEX,
               OUT / 'image_targets.json', OUT / 'image_calibration.json', OUT / 'image_distortion.json',
               ROOT / 'experiments/priority30_native_defenses/run_audit.py',
               ROOT / 'external_defenses/invertinggradients/inversefed/reconstruction_algorithms.py']
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in sources}
    if job['checkpoint'] == 'trained':
        hashes['trained_checkpoint'] = sha(OUT / 'checkpoint_replay/final_state.pt')
    if path.exists():
        prior = utility.read(path)
        assert prior['job'] == job and prior['source_hashes'] == hashes and prior['status'] == 'passed'
        assert sha(folder / 'arrays.npz') == prior['arrays_sha256']
        return dict(path=str(path), skipped=True)
    utility.log('image_attack_start', job=job)
    started = time.time()
    data = utility.p31.dp.CIFAR10(str(ROOT / 'datasets/cifar10'), train=False, download=False)
    normal, labels = utility.p31.tensors(data, job['indices'])
    dm = torch.tensor(native.inversefed.consts.cifar10_mean)[:, None, None]
    ds = torch.tensor(native.inversefed.consts.cifar10_std)[:, None, None]
    truth = normal*ds+dm
    model, _ = native.inversefed.construct_model('LeNetZhu', seed=42)
    if job['checkpoint'] == 'trained':
        replay = utility.read(OUT / 'checkpoint_replay/result.json')
        assert sha(OUT / 'checkpoint_replay/final_state.pt') == replay['checkpoint_sha256']
        model.load_state_dict(torch.load(OUT / 'checkpoint_replay/final_state.pt', map_location='cpu'))
    check(model); model.eval()
    state = native.gradient_dict(model, normal, labels, torch.nn.CrossEntropyLoss())
    plans, mix = None, 0
    if job['arm'] in utility.V1:
        transmitted = list(utility.transform(state, job['arm']).values())
        mix = utility.V1[job['arm']][0]
        modes = ['plain', 'v1_debias']
    elif job['arm'] == 'dna_v2_0p95':
        transmitted, plans = native.v2_sketch_payload(state)
        modes = ['v2_sketch']
    elif job['arm'] == 'unprotected':
        transmitted = list(state.values())
        modes = ['plain']
    elif job['arm'].startswith('dp_'):
        if job['variant'] == 'per_tensor':
            transmitted = utility.per_tensor_noise(list(state.values()), utility.read(OUT / 'image_clip.json')['clips'], job['sigma'], job['noise_seed'])
        elif job['variant'] in ['distortion', 'single']:
            transmitted = utility.p31.dp.add_clipped_noise(list(state.values()), job['C'], job['sigma'], job['noise_seed'])
        else:
            raise ValueError('unknown DP variant')
        modes = ['plain']
    else:
        raise ValueError('unknown image arm')
    candidates = []
    for mode in modes:
        torch.manual_seed(job['attack_seed']); np.random.seed(job['attack_seed'])
        evaluator = ObservableReconstructor(model, (dm, ds), native.IMAGE_CONFIG.copy(),
                     transmitted, mode, mix=mix, plans=plans, count=len(labels))
        result, stats = evaluator.reconstruct(transmitted, labels, img_shape=(3, 32, 32))
        candidates.append((float(stats['opt']), mode, result.detach(), dict(stats)))
    # No truth argument appears in optimization or candidate selection.
    best = min(enumerate(candidates), key=lambda item: (item[1][0], item[0]))[1]
    reconstructed = best[2]*ds+dm
    finite(reconstructed, 'raw reconstruction before metric clipping')
    matrix = ((truth[:, None]-reconstructed[None, :])**2).mean(dim=(2, 3, 4)).numpy()
    rows, columns = linear_sum_assignment(matrix)
    aligned = reconstructed[columns]
    metrics = [native.image_metrics(truth[i:i+1], aligned[i:i+1]) for i in range(len(labels))]
    scores = {key: float(np.mean([m[key] for m in metrics])) for key in metrics[0]}
    gray = torch.full_like(truth, .5)
    means = dm.expand_as(truth)
    references = {}
    for key, guess in [('gray', gray), ('cifar_mean', means)]:
        r = [native.image_metrics(truth[i:i+1], guess[i:i+1]) for i in range(len(labels))]
        references[key] = {k: float(np.mean([m[k] for m in r])) for k in r[0]}
    folder.mkdir(parents=True, exist_ok=True)
    torch.save(dict(kind='transmitted_only', payload=transmitted, projection_plans=plans), folder / 'receipt.pt')
    np.savez(folder / 'arrays.npz', truth=truth.numpy(), reconstruction=aligned.numpy(), indices=job['indices'], labels=labels.numpy())
    result = dict(status='passed', job=job, source_hashes=hashes, metrics=scores,
        record_metrics=metrics, references=references, selected_mode=best[1],
        observable_objectives={c[1]: c[0] for c in candidates}, pairing=columns.tolist(),
        arrays_sha256=sha(folder / 'arrays.npz'), receipt_sha256=sha(folder / 'receipt.pt'),
        elapsed_seconds=time.time()-started, iterations=4800, restarts=1)
    utility.save(path, result)
    utility.log('image_attack_complete', path=str(path), sha256=sha(path))
    return dict(path=str(path), skipped=False)


def build_jobs(targets, calibration, distortion):
    jobs, missing = [], []
    def append(setting, arm, ids, checkpoint='untrained', match=None, variant=None):
        for tid, indices in enumerate(ids):
            job = dict(setting=setting, arm=arm, indices=indices if isinstance(indices, list) else [indices],
                target_id=tid, checkpoint=checkpoint, attack_seed=333700+tid,
                noise_seed=333800+101*tid+sum(ord(c) for c in arm),
                sigma=None, C=None, variant=variant)
            if match is not None:
                job.update(sigma=match['sigma'], C=distortion['C'] if variant == 'distortion' else utility.read(OUT / 'image_clip.json')['C'])
            jobs.append(job)
    def dp(setting, method, variant, ids, checkpoint='untrained'):
        key = method+'::'+variant
        match = calibration['matches'][key]
        if not calibration['baseline_passed'] or not match['bracketed']:
            missing.append(dict(setting=setting, method=method, variant=variant, reason=match))
            return
        append(setting, 'dp_'+variant+'_for_'+method, ids, checkpoint, match, variant)
    append('C1', 'unprotected', targets['C1'])
    for method in ['dna_v1_medium', 'dna_v1_stronger']:
        append('C1', method, targets['C1'])
        append('C1', 'dp_distortion_for_'+method, targets['C1'], match=distortion['matches'][method], variant='distortion')
        dp('C1', method, 'single', targets['C1'])
    for method in ['dna_v1_conservative', 'dna_v2_0p95']:
        append('D', method, targets['C1'])
        dp('D', method, 'per_tensor', targets['C1'])
    for setting, checkpoint, ids in [('C2_batch4', 'untrained', targets['C2_batch4']), ('C2_trained', 'trained', targets['C2_trained'])]:
        append(setting, 'unprotected', ids, checkpoint)
        for method in ['dna_v1_conservative', 'dna_v2_0p95']:
            append(setting, method, ids, checkpoint)
            dp(setting, method, 'single', ids, checkpoint)
    return jobs, missing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    targets = prepare_targets()
    distortion = distortion_calibration(targets)
    if args.prepare_only:
        return
    calibration = utility.read(OUT / 'image_calibration.json')
    jobs, missing = build_jobs(targets, calibration, distortion)
    utility.save(OUT / 'image_gated_cells.json', missing)
    utility.save(OUT / 'image_job_manifest.json', jobs)
    started = time.time()
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(attack, job): job for job in jobs}
        for done, future in enumerate(as_completed(futures), 1):
            try:
                future.result()
            except BaseException as error:
                utility.save(OUT / 'REQUIRES_DIRECTION_IMAGE.json', dict(job=futures[future], error=repr(error)))
                utility.log('image_attack_error', job=futures[future], error=repr(error))
                raise
            utility.progress('image_recovery', done, len(jobs), started, method=futures[future]['arm'])
    utility.save(OUT / 'IMAGE_RECOVERY_COMPLETE.json', dict(jobs=len(jobs), missing=missing, seconds=time.time()-started))


if __name__ == '__main__':
    main()
