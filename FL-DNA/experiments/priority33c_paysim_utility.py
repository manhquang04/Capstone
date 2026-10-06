"""P33c full-state PaySim CPU utility qualification; negative BN is never repaired."""
import argparse
import hashlib
import importlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).absolute().parents[1]
OUT = ROOT / 'artifacts/priority33c'
SEEDS = list(range(270201, 270217))
GRID = [1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, .001, .003]
EXTENSION = [.01, .03, .1]
METHODS = ['dna_v1_conservative', 'dna_v2_0p95']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, obj):
    path = Path(path)
    assert path.absolute().is_relative_to(OUT)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def append(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        stream.write(json.dumps(dict(at=time.time(), **row), allow_nan=False)+'\n')


def folder(job):
    return OUT / 'paysim_utility' / job['method'] / f"seed_{job['seed']}"


def fingerprint():
    paths = [Path(__file__), ROOT / 'protocols/amendments/2026-10-03_priority33c_execution_annex.md',
        ROOT / 'protocols/amendments/2026-10-03_priority33c_paysim_execution_details.md',
        ROOT / 'protocols/config/priority27_c2_utility_grid.json',
        ROOT / 'data/load_creditcard.py', ROOT / 'experiments/fraud_fl_common.py',
        ROOT / 'experiments/run_fraud_fl_baseline.py', ROOT / 'experiments/run_fraud_fl_dna_transform.py',
        ROOT / 'experiments/run_fraud_fl_dna_transform_v2.py',
        ROOT / 'dna_encoder/transform_defense.py', ROOT / 'dna_encoder/transform_defense_v2.py']
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}


def run_job(job):
    place = folder(job)
    place.mkdir(parents=True, exist_ok=True)
    os.environ.update(FL_RUN_SEED=str(job['seed']), MAX_ROWS='500000', NUM_ROUNDS='50',
        LOCAL_EPOCHS='1', FL_NUM_CLIENTS='3', LOSS_TYPE='focal', FOCAL_ALPHA='.95',
        FOCAL_GAMMA='2.0', DATALOADER_NUM_WORKERS='0',
        BASELINE_OUTPUT_PATH=str(place / 'metrics.json'),
        DNA_TRANSFORM_OUTPUT_PATH=str(place / 'metrics.json'),
        DNA_TRANSFORM_V2_OUTPUT_PATH=str(place / 'metrics.json'),
        DNA_TRANSFORM_BLOCK_SIZE='256', DNA_TRANSFORM_MIX='.08',
        DNA_TRANSFORM_KEEP='.88', DNA_TRANSFORM_SHRINK='.45',
        DNA_TRANSFORM_V2_COMPRESSION_RATIO='.95', DNA_TRANSFORM_V2_QUANTIZATION_ETA='.01',
        PYTHONDONTWRITEBYTECODE='1')
    os.environ.pop('QUICK', None)
    for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS']:
        os.environ[k] = '1'
    sys.path.insert(0, str(ROOT))
    import numpy as np
    import torch
    from experiments import fraud_fl_common as common
    common.DEVICE = torch.device('cpu')
    torch.set_num_threads(1)
    script = {'baseline': 'run_fraud_fl_baseline',
              'dna_v1_conservative': 'run_fraud_fl_dna_transform',
              'dna_v2_0p95': 'run_fraud_fl_dna_transform_v2'}
    name = script.get(job['method'], 'run_fraud_fl_baseline' if job['method'].startswith('dp_') else None)
    if name is None:
        raise ValueError('unknown PaySim method')
    native = importlib.import_module('experiments.'+name)
    native.DEVICE = torch.device('cpu')
    original_factory = native.FraudMLP
    original_evaluate = common.evaluate_model
    original_average = common.fed_avg
    model_ref = []
    rounds, evaluations = [], []
    started = time.time()

    def strict(state, context):
        for key, value in state.items():
            if not bool(torch.isfinite(value).all()):
                raise FloatingPointError('nonfinite '+context+' '+key)
            if key.endswith('running_var') and bool((value < 0).any()):
                raise FloatingPointError('negative BN running_var '+context+' '+key+' min='+str(float(value.min())))

    def forward(module, inputs, outputs):
        assert outputs.device.type == 'cpu'
        strict(module.state_dict(), 'actual_forward')
        if not bool(torch.isfinite(outputs).all()):
            raise FloatingPointError('nonfinite forward output')

    def factory(dim):
        model = original_factory(dim)
        model.register_forward_hook(forward)
        model_ref.append(model)
        return model

    def average(states, counts):
        rows = {str(i): {k: float(v.min()) for k, v in s.items() if k.endswith('running_var')} for i, s in enumerate(states)}
        append(place / 'bn_rounds.jsonl', dict(round=len(rounds)+1, transmitted_min=rows))
        for i, s in enumerate(states):
            strict(s, 'transmitted_client_'+str(i))
        if job['method'].startswith('dp_'):
            global_state = model_ref[0].state_dict()
            clips = read(ROOT / 'protocols/config/priority27_c2_utility_grid.json')['clip_specs']['per_tensor_clip']
            changed = []
            for client, state in enumerate(states):
                generator = torch.Generator().manual_seed(job['seed']*100000+3*len(rounds)+client)
                result = {}
                for key, value in state.items():
                    if value.is_floating_point():
                        delta = value-global_state[key]
                        factor = min(1., clips[key]/max(float(delta.norm()), 1e-12))
                        result[key] = global_state[key]+delta*factor+torch.randn(value.shape, generator=generator,
                            dtype=value.dtype)*job['sigma']*clips[key]
                    else:
                        result[key] = value.clone()
                strict(result, 'DP transmitted client')
                changed.append(result)
            states = changed
        result = original_average(states, counts)
        strict(result, 'aggregate')
        rounds.append(dict(round=len(rounds)+1, minimum={k: float(v.min()) for k, v in result.items() if k.endswith('running_var')}))
        return result

    def evaluation(model, loader, threshold=None):
        strict(model.state_dict(), 'evaluation')
        result = original_evaluate(model, loader, threshold)
        if any(not np.isfinite(v) for v in result.values() if isinstance(v, (float, int))):
            raise FloatingPointError('nonfinite evaluation metrics')
        evaluations.append(dict(round=len(rounds), split='validation' if threshold is None else 'test', metrics=result))
        return result

    native.FraudMLP, native.fed_avg, native.evaluate_model = factory, average, evaluation
    result = dict(job=job, source_hashes=fingerprint(), device='cpu', threads=1)
    try:
        native.main()
        assert len(rounds) == 50
        val = [r for r in evaluations if r['round'] == 50 and r['split'] == 'validation'][0]['metrics']
        test = [r for r in evaluations if r['round'] == 50 and r['split'] == 'test'][0]['metrics']
        result.update(status='passed', validation=val, test=test, metrics_sha256=sha(place / 'metrics.json'))
    except FloatingPointError as error:
        result.update(status='GATE_FAILED', error=repr(error), completed_rounds=len(rounds))
    result.update(elapsed_seconds=time.time()-started, evaluations=evaluations, rounds=rounds)
    save(place / 'result.json', result)


def execute(job):
    place = folder(job)
    place.mkdir(parents=True, exist_ok=True)
    path = place / 'result.json'
    if path.exists():
        result = read(path)
        assert result['job'] == job and result['source_hashes'] == fingerprint()
        return result
    command = [str(ROOT / '.venv-phase1/bin/python'), '-B', '-u', str(Path(__file__)), '--job', json.dumps(job)]
    append(OUT / 'runs.jsonl', dict(event='paysim_utility_start', command=command, job=job))
    with (place / 'stdout.log').open('a') as so, (place / 'stderr.log').open('a') as se:
        proc = subprocess.run(command, cwd=ROOT, stdout=so, stderr=se)
    if proc.returncode or not path.exists():
        save(OUT / 'REQUIRES_DIRECTION_PAYSIM.json', dict(job=job, returncode=proc.returncode,
             reason='harness/infrastructure failure, not a numerical gate outcome'))
        raise RuntimeError('PaySim job failed: '+str(job))
    result = read(path)
    append(OUT / 'runs.jsonl', dict(event='paysim_utility_finish', job=job, status=result['status'], sha256=sha(path)))
    return result


def parallel(jobs):
    started = time.time()
    failed = 0
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(execute, j): j for j in jobs}
        for done, future in enumerate(as_completed(futures), 1):
            result = future.result()
            failed += result['status'] != 'passed'
            row = dict(stage='paysim_utility', dataset='PaySim', method=result['job']['method'], done=done,
                total=len(jobs), failed=failed, started_at=started, last_update=time.time(),
                eta_minutes=(time.time()-started)/done*(len(jobs)-done)/60)
            save(OUT / 'progress.json', row)
            append(OUT / 'progress.log', row)


def values(method):
    results = [read(folder(dict(method=method, seed=seed)) / 'result.json') for seed in SEEDS]
    if any(r['status'] != 'passed' for r in results):
        return None
    return [r['validation']['f1_score'] for r in results]


def selection(method, grid, base):
    import numpy as np
    target = values(method)
    if target is None or base is None:
        return dict(status='NOT_ASSESSABLE', bracketed=False, reason='full-state CPU utility BN/numerical gate failed')
    threshold = float(np.mean(np.array(target)-base))-.005
    means = {}
    for sigma in grid:
        v = values(f'dp_per_tensor_{sigma:g}')
        if v is None:
            return dict(status='NOT_ASSESSABLE', bracketed=False, reason='DP grid BN/numerical gate failed')
        means[sigma] = float(np.mean(np.array(v)-base))
    eligible = [s for s in grid if means[s] >= threshold]
    sigma = max(eligible) if eligible else None
    next_sigma = grid[grid.index(sigma)+1] if sigma is not None and sigma != grid[-1] else None
    bracket = next_sigma is not None and means[next_sigma] < threshold
    return dict(status='matched' if bracket else 'NOT_BRACKETED', bracketed=bracket,
        sigma=sigma, next_sigma=next_sigma, threshold=threshold, means=means)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job')
    args = parser.parse_args()
    if args.job:
        run_job(json.loads(args.job)); return
    jobs = [dict(method=m, seed=s, sigma=None) for m in ['baseline']+METHODS for s in SEEDS]
    parallel(jobs)
    base = values('baseline')
    valid = [m for m in METHODS if values(m) is not None and base is not None]
    if valid:
        parallel([dict(method=f'dp_per_tensor_{sigma:g}', seed=s, sigma=sigma) for sigma in GRID for s in SEEDS])
    matches = {m: selection(m, GRID, base) for m in METHODS}
    if any(r['status'] == 'NOT_BRACKETED' for r in matches.values()):
        parallel([dict(method=f'dp_per_tensor_{sigma:g}', seed=s, sigma=sigma) for sigma in EXTENSION for s in SEEDS])
        matches = {m: selection(m, GRID+EXTENSION, base) for m in METHODS}
    save(OUT / 'paysim_calibration.json', dict(matches=matches, seeds=SEEDS,
        baseline_valid=base is not None, grid=GRID, own_validation_F1_target=True,
        full_state=True, CPU=True))


if __name__ == '__main__':
    main()
