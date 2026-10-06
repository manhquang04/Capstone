"""P33c per-tensor utility-DP comparison in the existing PaySim BN-mean channel."""
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT))
for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[name] = '1'
from experiments import priority33c_paysim_utility as helper
from experiments.priority24_t1_bn_valid_rq1 import (BN_KEY, V1, V2, load_population,
    recover_mean, mse_std, capture_groups)
from experiments.priority33a_bn_mean import projection
from experiments import fraud_fl_common as common
from dna_encoder.transform_defense import transform_update_array
from dna_encoder.transform_defense_v2 import transform_update_array_v2
from privacy.seed_manager import derive_seed
import numpy as np
import pandas as pd
import torch
from scipy.stats import binomtest

OUT = helper.OUT
torch.set_num_threads(1)
common.DEVICE = torch.device('cpu')


def source_ids(value):
    result = set()
    if isinstance(value, dict):
        for k, v in value.items():
            if k in ['source_ids', 'source_rows'] and isinstance(v, (list, np.ndarray, torch.Tensor)):
                result.update(int(i) for i in np.asarray(v).flatten())
            else:
                result.update(source_ids(v))
    elif isinstance(value, (list, tuple)):
        for v in value:
            result.update(source_ids(v))
    return result


def sample():
    output = OUT / 'paysim_bn'
    path = output / 'targets.pt'
    if path.exists():
        doc = helper.read(output / 'targets.json')
        assert helper.sha(path) == doc['targets_sha256']
        return path
    from data.load_creditcard import BASE_FEATURE_COLUMNS
    from experiments.run_phase3_adam_ladder import excluded_rows, make_groups
    frame = pd.read_csv(ROOT / 'datasets/creditcard.csv', usecols=BASE_FEATURE_COLUMNS+['isFraud'])
    excluded = excluded_rows(frame.isFraud.to_numpy())
    evidence = []
    for file in (ROOT / 'artifacts').rglob('*targets*.pt'):
        if file.is_relative_to(OUT):
            continue
        value = torch.load(file, map_location='cpu', weights_only=False)
        ids = source_ids(value)
        if ids:
            excluded.update(ids)
            evidence.append(dict(path=str(file.relative_to(ROOT)), sha256=helper.sha(file), n=len(ids)))
    for file in (ROOT / 'artifacts').rglob('*.json'):
        if file.is_relative_to(OUT) or not any(s in str(file).lower() for s in ['paysim', 'phase4', 'priority33a', 'priority33b']):
            continue
        ids = source_ids(helper.read(file))
        if ids:
            excluded.update(ids)
            evidence.append(dict(path=str(file.relative_to(ROOT)), sha256=helper.sha(file), n=len(ids)))
    available = np.asarray(sorted(set(range(len(frame)))-excluded))
    groups = make_groups(frame, available, 8+24+39, 4, 1, 333650)
    ids = source_ids(groups)
    assert len(ids) == (8+24+39)*4 and not ids&excluded
    output.mkdir(parents=True, exist_ok=True)
    torch.save(groups, path)
    helper.save(output / 'targets.json', dict(seed=333650, sources=sorted(ids), overlap=0,
        exclusion_evidence=evidence, targets_sha256=helper.sha(path),
        raw_sha256=helper.sha(ROOT / 'datasets/creditcard.csv')))
    return path


def gate(captures, prior, std):
    recs = [recover_mean(c['model'], c['observed'][BN_KEY]) for c in captures]
    rows = []
    for i, cap in enumerate(captures):
        rows.append(dict(source_ids=cap['source_ids'], recovery=mse_std(recs[i], cap['true_mean'], std),
            prior=mse_std(prior, cap['true_mean'], std),
            decoy=mse_std(recs[(i+1)%len(captures)], cap['true_mean'], std)))
    tests = {}
    for name in ['prior', 'decoy']:
        wins = sum(r['recovery'] < r[name] for r in rows)
        losses = sum(r['recovery'] > r[name] for r in rows)
        p = float(binomtest(wins, wins+losses, alternative='greater').pvalue) if wins+losses else 1.
        tests[name] = dict(wins=wins, losses=losses, p=p)
    return dict(rows=rows, tests=tests, passed=all(t['p'] < .05 for t in tests.values()))


def recovery(model, raw, method, target):
    tensor_index = list(model.state_dict()).index(BN_KEY)
    if method == 'dna_v1_conservative':
        q, _ = transform_update_array(raw.numpy(), V1, tensor_index=tensor_index)
        receipt = dict(kind='v1', q=q)
        return recover_mean(model, torch.from_numpy(q)), receipt
    if method != 'dna_v2_0p95':
        raise ValueError('unknown PaySim defense')
    q, meta = transform_update_array_v2(raw.numpy(), V2, tensor_index=tensor_index,
        quantization_seed=derive_seed(V2.seed, 'priority24-v2', tensor_index))
    receipt = dict(kind='v2', q=q, metadata=meta)
    r = projection(meta)
    linear, bn = model.network[0], model.network[1]
    w = linear.weight.detach().double().numpy()
    m = float(bn.momentum)
    offset = m*(linear.bias.detach().double().numpy()-bn.running_mean.detach().double().numpy())
    return np.linalg.lstsq(r@(m*w), np.asarray(q, dtype=np.float64)-r@offset, rcond=None)[0], receipt


def main():
    started = time.time()
    utilities = helper.read(OUT / 'paysim_calibration.json')
    matched = {m: s for m, s in utilities['matches'].items() if s.get('bracketed')}
    if not matched:
        helper.save(OUT / 'PAYSIM_BN_COMPLETE.json', dict(status='NOT_ASSESSABLE',
            reason='no BN-valid full-state CPU utility target and bracketed comparator', utility=utilities))
        return
    target = sample()
    groups = torch.load(target, map_location='cpu', weights_only=False)
    prior, std, _ = load_population()
    stages = [('n8', groups[:8]), ('n24', groups[8:32]), ('n39', groups[32:])]
    clips = helper.read(ROOT / 'protocols/config/priority27_c2_utility_grid.json')['clip_specs']['per_tensor_clip']
    for stage, batch in stages:
        batch_path = OUT / 'paysim_bn' / f'{stage}_targets.pt'
        if not batch_path.exists():
            torch.save(batch, batch_path)
        captures = capture_groups(batch_path, 333660, 'priority33c_'+stage)
        if stage != 'n39':
            test = gate(captures, prior, std)
            helper.save(OUT / 'paysim_bn' / f'{stage}_gate.json', test)
            if not test['passed']:
                helper.save(OUT / 'PAYSIM_BN_COMPLETE.json', dict(status='NOT_ASSESSABLE',
                    reason='unprotected qualification failed '+stage, qualification=test))
                return
            continue
        for method, selected in matched.items():
            for i, cap in enumerate(captures):
                path = OUT / 'paysim_bn' / method / f'target_{i:03d}' / 'result.json'
                if path.exists():
                    stored = helper.read(path)
                    assert stored['source_ids'] == cap['source_ids'] and stored['sigma'] == selected['sigma']
                    continue
                estimate, receipt = recovery(cap['model'], cap['observed'][BN_KEY], method, i)
                generator = torch.Generator().manual_seed(333670+101*i+sum(map(ord, method)))
                dp = {}
                for name, value in cap['observed'].items():
                    if value.is_floating_point():
                        clip = clips[name]
                        factor = min(1., clip/max(float(value.norm()), 1e-12))
                        dp[name] = value*factor+torch.randn(value.shape, dtype=value.dtype, generator=generator)*clip*selected['sigma']
                    else:
                        dp[name] = value.clone()
                error = None
                for name, value in dp.items():
                    received = cap['model'].state_dict()[name]+value
                    if not bool(torch.isfinite(received).all()) or (name.endswith('running_var') and bool((received < 0).any())):
                        error = 'BN or numerical gate failed '+name
                if not np.isfinite(estimate).all():
                    error = 'nonfinite DNA batch-mean estimate'
                if error:
                    helper.save(path, dict(status='GATE_FAILED', method=method, source_ids=cap['source_ids'],
                        sigma=selected['sigma'], error=error))
                else:
                    dp_estimate = recover_mean(cap['model'], dp[BN_KEY])
                    if not np.isfinite(dp_estimate).all():
                        raise FloatingPointError('nonfinite DP batch-mean estimate')
                    path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(receipt, path.parent / 'transmitted_receipt.pt')
                    np.savez(path.parent / 'arrays.npz', truth=cap['true_mean'], dna=estimate, dp=dp_estimate, std=std)
                    helper.save(path, dict(status='passed', method=method, source_ids=cap['source_ids'],
                        sigma=selected['sigma'], dna_mse=mse_std(estimate, cap['true_mean'], std),
                        dp_mse=mse_std(dp_estimate, cap['true_mean'], std),
                        arrays_sha256=helper.sha(path.parent / 'arrays.npz')))
                progress = dict(stage='PaySim_BN_n39', dataset='PaySim', method=method,
                    done=i+1, total=39, failed=0 if not error else 1,
                    started_at=started, last_update=time.time(),
                    eta_minutes=(time.time()-started)/(i+1)*(39-i-1)/60)
                helper.save(OUT / 'progress.json', progress)
                helper.append(OUT / 'progress.log', progress)
    helper.save(OUT / 'PAYSIM_BN_COMPLETE.json', dict(status='resolved', elapsed_seconds=time.time()-started))


if __name__ == '__main__':
    main()
