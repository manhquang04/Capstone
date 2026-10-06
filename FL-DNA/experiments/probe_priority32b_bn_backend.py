"""Fixed inference-only BN backend diagnosis; no tensor correction or training."""
import hashlib
import json
import os
import sys
from pathlib import Path

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'artifacts/priority32b_bn_reconciliation/backend_probe'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    import numpy as np
    import torch
    from models.fraud_mlp import FraudMLP
    from experiments.priority32b_bn_reconcile import SEEDS
    torch.set_num_threads(1)
    amendment = ROOT / 'protocols/amendments/2026-10-02_priority32b_backend_probe_diagnosis.md'
    assert amendment.is_file()
    assert not OUT.exists(), 'Preserve prior probe; do not rerun'
    states = [(method, seed, ROOT / 'artifacts/priority32b_bn_reconciliation/jobs/registered'
               / method / f'seed_{seed}' / 'final_state.pt') for method, values in SEEDS.items() for seed in values]
    assert all(path.is_file() for _, _, path in states), 'All six frozen registered checkpoints required'
    OUT.mkdir()
    data_path = ROOT / 'artifacts/priority32_multidataset/prepared/paysim/validation_x.npy'
    x = torch.from_numpy(np.array(np.load(data_path)[:1024], copy=True))
    devices = ['cpu'] + (['mps'] if torch.backends.mps.is_available() else [])
    report = {'amendment_sha256': sha(amendment), 'script_sha256': sha(Path(__file__)),
              'input_sha256': sha(data_path), 'rows': len(x), 'threads': torch.get_num_threads(),
              'operator_probe': {}, 'checkpoints': []}
    before = torch.get_rng_state().clone()
    with torch.no_grad():
        for device in devices:
            probe_x = torch.tensor([[1., 1., 1.], [2., 2., 2.]], device=device)
            mean = torch.zeros(3, device=device)
            var = torch.tensor([-.25, 0., .25], device=device)
            y = torch.nn.functional.batch_norm(probe_x, mean, var, torch.ones(3, device=device),
                torch.zeros(3, device=device), training=False, eps=1e-5)
            formula = (probe_x - mean) / torch.sqrt(var + 1e-5)
            np.save(OUT / f'operator_{device}.npy', y.cpu().numpy())
            report['operator_probe'][device] = {'finite_output': torch.isfinite(y).cpu().tolist(),
                'finite_defining_formula': torch.isfinite(formula).cpu().tolist()}
        for method, seed, path in states:
            state = torch.load(path, map_location='cpu')
            row = {'method': method, 'seed': seed, 'checkpoint_sha256': sha(path),
                'negative_bn': {k: int((v < 0).sum()) for k, v in state.items() if k.endswith('running_var')}, 'devices': {}}
            outputs = {}
            for device in devices:
                # Constructor RNG restored; inference itself never draws RNG.
                with torch.random.fork_rng(devices=[]):
                    model = FraudMLP(x.shape[1])
                model.load_state_dict(state)
                model.to(device).eval()
                logits = model(x.to(device)).cpu()
                probability = torch.sigmoid(logits)
                outputs[device] = logits.numpy().reshape(-1)
                np.save(OUT / f'{method}_{seed}_{device}_logits.npy', outputs[device])
                row['devices'][device] = {'finite_logits': int(torch.isfinite(logits).sum()),
                    'finite_probabilities': int(torch.isfinite(probability).sum()), 'rows': len(x)}
            if 'mps' in outputs:
                both = np.isfinite(outputs['cpu']) & np.isfinite(outputs['mps'])
                row['paired_finite_rows'] = int(both.sum())
                row['max_logit_difference_finite_rows'] = float(np.max(np.abs(outputs['cpu'][both]-outputs['mps'][both]))) if both.any() else None
            report['checkpoints'].append(row)
    assert torch.equal(before, torch.get_rng_state()), 'Inference diagnostic changed CPU RNG'
    report['cpu_rng_unchanged'] = True
    (OUT / 'result.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'checkpoints': len(states), 'devices': devices, 'cpu_rng_unchanged': True}))


if __name__ == '__main__':
    main()
