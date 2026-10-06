"""Fixed read-only layer traces for BN-invalid registered checkpoints."""
import hashlib
import json
import os
import sys
from pathlib import Path

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'artifacts/priority32b_bn_reconciliation/activation_probe'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    import numpy as np
    import torch
    from models.fraud_mlp import FraudMLP
    from experiments.priority32b_bn_reconcile import SEEDS
    torch.set_num_threads(1)
    amendment = ROOT / 'protocols/amendments/2026-10-02_priority32b_activation_probe_diagnosis.md'
    assert amendment.is_file() and not OUT.exists()
    OUT.mkdir()
    data_path = ROOT / 'artifacts/priority32_multidataset/prepared/paysim/validation_x.npy'
    x = torch.from_numpy(np.array(np.load(data_path)[:1024], copy=True))
    report = {'amendment_sha256': sha(amendment), 'script_sha256': sha(Path(__file__)),
              'input_sha256': sha(data_path), 'threads': 1, 'relu_probe': {}, 'traces': []}
    before = torch.get_rng_state().clone()
    with torch.no_grad():
        for device in ['cpu'] + (['mps'] if torch.backends.mps.is_available() else []):
            y = torch.relu(torch.tensor([float('nan'), -1., 0., 1.], device=device)).cpu()
            report['relu_probe'][device] = {'finite': torch.isfinite(y).tolist(),
                'values': [float(value) if torch.isfinite(value) else None for value in y]}
            for method, seeds in SEEDS.items():
                for seed in seeds:
                    checkpoint = ROOT / 'artifacts/priority32b_bn_reconciliation/jobs/registered' / method / f'seed_{seed}' / 'final_state.pt'
                    state = torch.load(checkpoint, map_location='cpu')
                    with torch.random.fork_rng(devices=[]):
                        model = FraudMLP(x.shape[1])
                    model.load_state_dict(state)
                    model.to(device).eval()
                    trace = {'method': method, 'seed': seed, 'device': device,
                             'checkpoint_sha256': sha(checkpoint), 'layers': []}
                    handles = []
                    def reader(name):
                        def hook(module, inputs, outputs):
                            trace['layers'].append({'layer': name, 'type': type(module).__name__,
                                'shape': list(outputs.shape), 'nan': int(torch.isnan(outputs).sum().cpu()),
                                'infinity': int(torch.isinf(outputs).sum().cpu()),
                                'finite': int(torch.isfinite(outputs).sum().cpu())})
                        return hook
                    for name, layer in model.network.named_children():
                        handles.append(layer.register_forward_hook(reader('network.' + name)))
                    try:
                        model(x.to(device))
                    finally:
                        for handle in handles:
                            handle.remove()
                    assert all(torch.equal(v.cpu(), model.state_dict()[k].cpu()) for k, v in state.items())
                    report['traces'].append(trace)
    assert torch.equal(before, torch.get_rng_state())
    report['cpu_rng_unchanged'] = True
    (OUT / 'result.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'traces': len(report['traces']), 'relu_probe': report['relu_probe']}))


if __name__ == '__main__':
    main()
