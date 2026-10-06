"""Single-process official TabLeak instrument. All writes stay in external_defenses."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '2'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['DATALOADER_NUM_WORKERS'] = '0'
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import traceback
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OFFICIAL = ROOT / 'external_defenses/tableak'
sys.path.insert(0, str(OFFICIAL))
from attacks import invert_grad
from datasets import ADULT
from datasets.base_dataset import BaseDataset
from models import FullyConnected
from utils import match_reconstruction_ground_truth, post_process_continuous, batch_feature_wise_accuracy_score
from utils.encoder_decoder import to_categorical
import attacks.gradient_inversion_attack as official_attack
_official_sigmoid_bound = official_attack.continuous_sigmoid_bound
def nonleaf_sigmoid_bound(x, *args, **kwargs):
    # PyTorch 2.x rejects the official final leaf-tensor in-place slice update.
    # Clone only; exactly the official transform, autograd preserved, source untouched.
    return _official_sigmoid_bound(x.clone(), *args, **kwargs)
official_attack.continuous_sigmoid_bound = nonleaf_sigmoid_bound

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def official_config():
    tree = ast.parse((OFFICIAL / 'run_inversion_attacks.py').read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'configs' for t in node.targets):
            entry = next(v for k, v in zip(node.value.keys, node.value.values) if ast.literal_eval(k) == 46)
            config = {ast.literal_eval(k): ('cpu' if isinstance(v, ast.Attribute) else ast.literal_eval(v)) for k, v in zip(entry.keys, entry.values)}
            config.pop('invert_labels')
            config['return_all'] = True
            config['return_all_reconstruction_losses'] = True
            return config
    raise RuntimeError('Official experiment 46 absent')

class PaySimAdapter(BaseDataset):
    def __init__(self, train, metadata, numeric_columns):
        super().__init__('PaySim-project-loader', 'cpu', 42)
        self.metadata = metadata
        self.train_features = {name: None for name in numeric_columns}
        self.train_features['type'] = metadata.type_categories
        self.features = dict(self.train_features, isFraud=['0', '1'])
        self.label = 'isFraud'
        self.Xtrain, self.ytrain = train
        self.Xtest, self.ytest = self.Xtrain[:0], self.ytrain[:0]
        self.num_features = self.Xtrain.shape[1]
        self.mean = torch.tensor(metadata.numeric_center + [0.] * len(metadata.type_categories))
        self.std = torch.tensor(metadata.numeric_scale + [1.] * len(metadata.type_categories))
        self.standardized = True
        self._create_index_maps()
        raw = self.de_standardize(self.Xtrain)
        self.metric_std = raw[:, :len(numeric_columns)].std(0)
        self.continuous_bounds = {}
        self.standardized_continuous_bounds = {}
        for i, name in enumerate(numeric_columns):
            self.continuous_bounds[name] = (raw[:, i].min().item(), raw[:, i].max().item())
            self.standardized_continuous_bounds[name] = (self.Xtrain[:, i].min().item(), self.Xtrain[:, i].max().item())
    def decode_batch(self, batch, standardized=True):
        if standardized:
            batch = self.de_standardize(batch)
        # Raw currency is continuous, unlike integer-valued Adult columns.
        return to_categorical(batch.detach().cpu().numpy(), self.train_features, nearest_int=False)
    def create_tolerance_map(self, tol=0.319):
        return [float(tol * x) for x in self.metric_std] + ['cat']

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def measure(dataset, x, reconstruction):
    rec = post_process_continuous(reconstruction.detach().clone(), dataset)
    truth = dataset.decode_batch(x, standardized=True)
    guess = dataset.decode_batch(rec, standardized=True)
    aligned, all_error, cat_error, cont_error = match_reconstruction_ground_truth(truth, guess, dataset.create_tolerance_map())
    return {
        'accuracy_percent': 100 * (1 - float(np.mean(all_error))),
        'categorical_accuracy_percent': 100 * (1 - float(np.mean(cat_error))),
        'continuous_accuracy_percent': 100 * (1 - float(np.mean(cont_error))),
        'per_feature_accuracy_percent': {k: 100 * (1-float(v)) for k, v in batch_feature_wise_accuracy_score(truth, aligned, dataset.create_tolerance_map(), dataset.train_features).items()},
    }

def gradient(net, criterion, x, y):
    return [g.detach() for g in torch.autograd.grad(criterion(net(x), y), net.parameters())]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-name', required=True)
    args = parser.parse_args()
    out = HERE / 'runs' / args.run_name
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    os.chdir(OFFICIAL)
    config = official_config()
    manifest = {'official_commit': '6b8c1c82ffe85bc6fdd96938ac42b6c9b5685300', 'compatibility': 'Driver wraps official continuous_sigmoid_bound with x.clone() to permit leaf slice updates on torch 2.x; official source unchanged', 'config': config, 'torch': torch.__version__, 'numpy': np.__version__, 'seed': 42, 'batch_size': 8, 'threads': torch.get_num_threads(), 'inter_op_threads': torch.get_num_interop_threads(), 'processes': 1, 'source_hashes': {str(p.relative_to(ROOT)): sha(p) for p in [OFFICIAL/'attacks/gradient_inversion_attack.py', ROOT/'data/load_creditcard.py', ROOT/'models/fraud_mlp.py']}}
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
    scenarios = ['adult_native', 'paysim_official_fc', 'paysim_project_eval']
    paysim = None
    for scenario in scenarios:
        directory = out/scenario
        directory.mkdir()
        try:
            np.random.seed(42)
            torch.manual_seed(42)
            if scenario == 'adult_native':
                dataset = ADULT()
                dataset.standardize()
                net = FullyConnected(dataset.num_features, [100, 100, 2])
                criterion = torch.nn.CrossEntropyLoss()
                # Same setting as README's example; no training.
                indices = np.random.randint(0, len(dataset.Xtrain), 8)
                x, y = dataset.Xtrain[indices], dataset.ytrain[indices]
                published = {'table': 'Table 1, true labels, batch 8', 'mean_accuracy_percent': 95.2, 'std_accuracy_percent': 8.8, 'batches': 50, 'source': 'https://arxiv.org/abs/2210.01785', 'single_batch_not_statistical_reproduction': True}
            else:
                if paysim is None:
                    loader = load_module('paysim_readonly_loader', ROOT/'data/load_creditcard.py')
                    clients, val, test, dim, pos_weight, metadata = loader.load_creditcard_data(batch_size=8, max_rows=100000, seed=42)
                    assert all(c.num_workers == 0 for c in clients)
                    train = (torch.cat([c.dataset.tensors[0] for c in clients]), torch.cat([c.dataset.tensors[1] for c in clients]))
                    paysim = (PaySimAdapter(train, metadata, loader.NUMERIC_COLUMNS), clients, dim, pos_weight)
                dataset, clients, dim, pos_weight = paysim
                indices = np.random.randint(0, len(clients[0].dataset), 8)
                x = clients[0].dataset.tensors[0][indices].clone()
                y = clients[0].dataset.tensors[1][indices].clone()
                if scenario == 'paysim_official_fc':
                    net = FullyConnected(dim, [100, 100, 2])
                    criterion = torch.nn.CrossEntropyLoss()
                    y = y.flatten().long()
                else:
                    model_module = load_module('fraud_mlp_readonly', ROOT/'models/fraud_mlp.py')
                    net = model_module.FraudMLP(dim)
                    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                    # Stochastic train-mode gradient replay diagnostic on the real model.
                    net.train()
                    initial_state = {k: v.clone() for k,v in net.state_dict().items()}
                    reference = torch.cat([g.flatten() for g in gradient(net, criterion, x, y)])
                    replay_cosines = []
                    for _ in range(5):
                        replay = torch.cat([g.flatten() for g in gradient(net, criterion, x, y)])
                        replay_cosines.append(torch.nn.functional.cosine_similarity(reference, replay, dim=0).item())
                    net.load_state_dict(initial_state)
                    net.eval()
                    eval_reference = torch.cat([g.flatten() for g in gradient(net, criterion, x, y)])
                    eval_replay = torch.cat([g.flatten() for g in gradient(net, criterion, x, y)])
                    (directory/'replay_diagnostic.json').write_text(json.dumps({'train_mode_same_true_input_gradient_cosines': replay_cosines, 'eval_mode_same_input_gradient_cosine': torch.nn.functional.cosine_similarity(eval_reference, eval_replay, dim=0).item(), 'eval_mode_same_input_gradient_max_abs_diff': (eval_reference-eval_replay).abs().max().item(), 'loss_for_instrument': 'weighted_bce', 'not_claimed_to_match_project_default_focal': True}, indent=2))
                published = None
            true_grad = gradient(net, criterion, x, y)
            torch.save(net.state_dict(), directory/'model.pt')
            np.savez(directory/'inputs.npz', truth=x.numpy(), labels=y.numpy(), indices=indices, mean=dataset.mean.numpy(), std=dataset.std.numpy())
            spec = {'features': dataset.train_features, 'tolerance_map': [v if isinstance(v, str) else float(v) for v in dataset.create_tolerance_map()], 'standardized_continuous_bounds': {k: [float(v) for v in bounds] for k, bounds in dataset.standardized_continuous_bounds.items()}, 'round_numeric': scenario == 'adult_native', 'sample_positive_labels': int((y == 1).sum()), 'sample_client': None if scenario == 'adult_native' else 0}
            (directory/'adapter.json').write_text(json.dumps(spec, indent=2))
            print('START', scenario, config, flush=True)
            start = time.monotonic()
            # Give the attacker only shape/device, not the actual feature values.
            rec, ensemble, losses = invert_grad(net=net, training_criterion=criterion, true_grad=true_grad, true_label=y, true_data=torch.empty_like(x), dataset=dataset, **config)
            elapsed = time.monotonic()-start
            np.savez(directory/'arrays.npz', truth=x.numpy(), labels=y.numpy(), reconstruction=rec.detach().numpy(), ensemble=np.stack([r.numpy() for r in ensemble]), objective_losses=np.array(losses))
            result = dict(manifest, scenario=scenario, runtime_seconds=elapsed, metric=measure(dataset, x, rec), published_reference=published, spec=spec, model=str(net), known_labels=True, checkpoint='initialization', defense='none')
            if scenario != 'adult_native':
                result['paysim_metadata'] = vars(dataset.metadata)
                result['normalized_mse_unaligned_not_primary'] = float(((rec-x)**2).mean())
            (directory/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False))
            print('DONE', scenario, result['metric'], 'seconds', elapsed, flush=True)
        except Exception:
            error = traceback.format_exc()
            (directory/'failure.txt').write_text(error)
            print(error, flush=True)
            if scenario == 'adult_native':
                raise

if __name__ == '__main__':
    main()
