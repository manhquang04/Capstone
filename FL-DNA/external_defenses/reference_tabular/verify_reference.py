"""Fresh-process verification and metric recomputation of frozen arrays."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name] = '2'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import argparse
import hashlib
import json
from pathlib import Path
import sys
import subprocess
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT/'external_defenses/tableak'))
from utils import match_reconstruction_ground_truth, to_categorical

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-name', default='default_batch8_seed42_v3')
    parser.add_argument('--scenarios', nargs='+', default=['adult_native', 'paysim_official_fc', 'paysim_project_eval'])
    parser.add_argument('--output-name', default='verification.json')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    run = HERE/'runs'/args.run_name
    results = {}
    for name in args.scenarios:
        directory = run/name
        record = json.loads((directory/'result.json').read_text())
        spec = json.loads((directory/'adapter.json').read_text())
        arr = np.load(directory/'arrays.npz')
        inputs = np.load(directory/'inputs.npz')
        true, rec = arr['truth'], arr['reconstruction'].copy()
        assert true.shape == rec.shape
        assert true.shape[0] == 8
        assert arr['ensemble'].shape == (30, *true.shape)
        assert arr['objective_losses'].shape == (30,)
        for key in arr.files:
            assert np.isfinite(arr[key]).all(), (name, key)
        assert np.array_equal(true, inputs['truth'])
        assert record['config']['max_iterations'] == 1500
        assert record['config']['post_selection'] == 30
        assert record['config']['perfect_pooling'] is False
        assert record['threads'] == 2 and record['processes'] == 1
        pointer = 0
        for feature, categories in spec['features'].items():
            if categories is None:
                low, high = spec['standardized_continuous_bounds'][feature]
                rec[:, pointer] = np.clip(rec[:, pointer], low, high)
                pointer += 1
            else:
                pointer += len(categories)
        true_raw = true * inputs['std'] + inputs['mean']
        rec_raw = rec * inputs['std'] + inputs['mean']
        truth = to_categorical(true_raw, spec['features'], nearest_int=spec['round_numeric'])
        guess = to_categorical(rec_raw, spec['features'], nearest_int=spec['round_numeric'])
        aligned, error, cat, cont = match_reconstruction_ground_truth(truth, guess, spec['tolerance_map'])
        metric = {'accuracy_percent': 100*(1-float(np.mean(error))), 'categorical_accuracy_percent': 100*(1-float(np.mean(cat))), 'continuous_accuracy_percent': 100*(1-float(np.mean(cont)))}
        # Independent feature-wise score AND independent assignment cost construction.
        costs = np.zeros((8,8,len(spec['features'])))
        for i in range(8):
            for j in range(8):
                for k, tolerance in enumerate(spec['tolerance_map']):
                    if tolerance == 'cat':
                        costs[i,j,k] = str(truth[i,k]) != str(guess[j,k])
                    else:
                        costs[i,j,k] = not (float(truth[i,k])-tolerance <= float(guess[j,k]) <= float(truth[i,k])+tolerance)
        row, column = linear_sum_assignment(costs.mean(2))
        independent_accuracy = 100*(1-costs[row,column].mean())
        assert np.isclose(metric['accuracy_percent'], independent_accuracy, atol=1e-10)
        for k,v in metric.items():
            assert np.isclose(v, record['metric'][k], atol=1e-10), (name,k,v,record['metric'][k])
        # Per-feature summary independently after the OFFICIAL tie-broken assignment.
        feature_accuracy = {}
        for k,(feature,tolerance) in enumerate(zip(spec['features'],spec['tolerance_map'])):
            if tolerance == 'cat':
                good = truth[:,k] == aligned[:,k]
            else:
                t = truth[:,k].astype(float)
                r = aligned[:,k].astype(float)
                good = (t-tolerance <= r) & (r <= t+tolerance)
            feature_accuracy[feature] = 100*float(good.mean())
            assert np.isclose(feature_accuracy[feature],record['metric']['per_feature_accuracy_percent'][feature])
        results[name] = dict(metric, independently_recomputed_accuracy_percent=float(independent_accuracy), per_feature_accuracy_percent=feature_accuracy, runtime_seconds=record['runtime_seconds'], array_sha256=hashlib.sha256((directory/'arrays.npz').read_bytes()).hexdigest(), sample_positive_labels=spec['sample_positive_labels'])
        if name != 'adult_native':
            errors = np.abs(truth[:,:8].astype(float)-aligned[:,:8].astype(float))
            results[name]['raw_numeric_mae_per_feature'] = {feature:float(errors[:,i].mean()) for i,feature in enumerate(list(spec['features'])[:8])}
            reconstructed = aligned[:,:8].astype(float)
            results[name]['derived_consistency_mae_raw'] = {
                'balance_diff_orig': float(np.abs(reconstructed[:,6] - (reconstructed[:,2]-reconstructed[:,3])).mean()),
                'balance_diff_dest': float(np.abs(reconstructed[:,7] - (reconstructed[:,5]-reconstructed[:,4])).mean())}
        print(name, metric, 'VERIFIED', flush=True)
    for relative, expected in record['source_hashes'].items():
        assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest() == expected
    official_commit = subprocess.check_output(['git','-C',str(ROOT/'external_defenses/tableak'),'rev-parse','HEAD'],text=True).strip()
    assert official_commit == record['official_commit']
    # No upstream modifications; untracked files are not source edits.
    changed = subprocess.check_output(['git','-C',str(ROOT/'external_defenses/tableak'),'diff','--name-only'],text=True).strip()
    assert not changed, changed
    results['source_checks'] = {'official_commit':official_commit,'project_commit':subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(), 'official_tracked_diff':changed}
    (HERE/args.output_name).write_text(json.dumps(results,indent=2,allow_nan=False))

if __name__ == '__main__':
    main()
