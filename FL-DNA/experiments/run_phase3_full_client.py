"""Capture the real client training path and evaluate bounded full-set inversion.

Every client record participates in the local epoch and dummy optimization.
Budget is explicit; completion is not an attack-effectiveness claim.
"""
import argparse
import copy
import csv
import hashlib
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

try:
    import resource
except ImportError:  # Windows does not expose POSIX process resource counters.
    resource = None

import numpy as np
import torch

from attacks.local_update import simulate, objective
from data.load_creditcard import load_creditcard_data
from data import load_creditcard as data_module
from experiments import fraud_fl_common as common
from models.fraud_mlp import FraudMLP
from privacy.seed_manager import generate_run_seed, derive_seed


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def checksum(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


class Recorder:
    """Observe actual loader order without changing features or training."""
    def __init__(self, loader):
        self.loader = loader
        self.parts = []
        self.rng = None

    def __iter__(self):
        for x, y, ids in self.loader:
            if self.rng is None:
                self.rng = torch.get_rng_state().clone()
            self.parts.append((x.clone(), y.clone(), ids.clone()))
            yield x, y


def score(original, candidate, labels, metadata, path):
    x, z = original.numpy().astype('float64'), candidate.numpy().astype('float64')
    mse = ((x-z)**2).mean(1)
    mae = np.abs(x-z).mean(1)
    cat = [i for i,n in enumerate(metadata.feature_names) if n.startswith('type=')]
    numeric = [i for i in range(x.shape[1]) if i not in cat]
    type_ok = (x[:,cat].argmax(1) == z[:,cat].argmax(1)).astype(float)
    valid = (np.isclose(z[:,cat], 0, atol=1e-6, rtol=0) | np.isclose(z[:,cat], 1, atol=1e-6, rtol=0)).all(1) & (np.isclose(z[:,cat], 1, atol=1e-6, rtol=0).sum(1)==1)
    with path.open('w') as f:
        writer = csv.writer(f)
        writer.writerow(['ordered_record','label','mse','mae','argmax_type_correct','onehot_valid'])
        writer.writerows(zip(range(len(x)), labels.flatten().tolist(), mse, mae, type_ok, valid))
    groups = {}
    for label in (0,1):
        mask = labels.numpy().flatten() == label
        groups[str(label)] = dict(n=int(mask.sum()), mean_mse=float(mse[mask].mean()),
            median_mse=float(np.median(mse[mask])), mean_mae=float(mae[mask].mean()),
            type_accuracy=float(type_ok[mask].mean()), onehot_valid=float(valid[mask].mean()))
    return dict(mean_mse=float(mse.mean()), median_mse=float(np.median(mse)), mean_mae=float(mae.mean()),
        classes=groups, feature_mae_scaled=dict(zip(metadata.feature_names, np.abs(x-z).mean(0).tolist())),
        feature_mae_raw=dict(zip([metadata.feature_names[i] for i in numeric],
            (np.abs(x[:,numeric]-z[:,numeric])*np.asarray(metadata.numeric_scale)).mean(0).tolist())))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--max-rows', type=int, default=500000)
    p.add_argument('--iterations', type=int, default=30)
    p.add_argument('--restarts', type=int, default=2)
    p.add_argument('--clients', type=int, nargs='+', default=[0,1,2])
    p.add_argument('--seed', type=int)
    p.add_argument('--data-seed', type=int, default=20260907)
    args = p.parse_args()
    if args.iterations < 1 or args.restarts < 1 or any(c not in (0,1,2) for c in args.clients):
        p.error('Positive budgets and client ids 0..2 required')
    root = Path(__file__).resolve().parents[1]
    out = root/'artifacts/phase3'/datetime.now(timezone.utc).strftime('full_%Y%m%dT%H%M%S%fZ')
    out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    seed = args.seed if args.seed is not None else generate_run_seed()
    torch.set_num_threads(1)
    common.DEVICE = torch.device('cpu')
    data_module.DEFAULT_NUM_WORKERS = 0
    torch.manual_seed(seed)
    config = dict(vars(args), run_seed=seed, batch_size=1024, local_epochs=1, num_workers=0, optimizer='Adam',
        lr=.001, betas=[.9,.999], eps=1e-8, model_mode='train', known_labels=True,
        known_order=True, known_rng=True, attack_lr=.05, selection='minimum parameter-delta objective',
        observation='individual baseline full local-set delta; buffers available but not optimized',
        checkpoint='one FedAvg warmup round on this exact split/scaler; development', torch_version=torch.__version__)
    dump(out/'protocol_lock.json', config)
    reports, failures = [], []
    try:
        t = time.perf_counter()
        loaders, _, _, dim, weight, metadata = load_creditcard_data(batch_size=1024, num_clients=3,
            max_rows=args.max_rows, seed=args.data_seed)
        setup = time.perf_counter()-t
        dump(out/'preprocessing.json', asdict(metadata))
        base = FraudMLP(dim).train()
        warmup_start = time.perf_counter()
        states = []
        for loader in loaders:
            warmup = copy.deepcopy(base)
            common.train_local_model(warmup, loader, weight, local_epochs=1)
            states.append(copy.deepcopy(warmup.state_dict()))
        base.load_state_dict(common.fed_avg(states,[len(loader.dataset) for loader in loaders]))
        torch.save(base.state_dict(),out/'pre_local.pt')
        dump(out/'setup_profile.json',dict(preprocessing_seconds=setup,
             warmup_seconds=time.perf_counter()-warmup_start,
             dataset_sha256=checksum(root/'datasets/creditcard.csv')))
        criterion = common.build_loss(weight)
        # Loader-local row IDs plus tensor hashes define each exact target set.
        for client in args.clients:
            folder = out/f'client_{client}'
            folder.mkdir()
            loader = loaders[client]
            loader.dataset.tensors = (*loader.dataset.tensors, torch.arange(len(loader.dataset)))
            recorder = Recorder(loader)
            local = copy.deepcopy(base)
            torch.manual_seed(derive_seed(seed, 'local', client))
            t = time.perf_counter()
            training_loss = common.train_local_model(local, recorder, weight, local_epochs=1)
            training_seconds = time.perf_counter()-t
            x = torch.cat([v[0] for v in recorder.parts])
            y = torch.cat([v[1] for v in recorder.parts])
            ids = torch.cat([v[2] for v in recorder.parts])
            assert len(x)==len(loader.dataset) and len(ids.unique())==len(x)
            sizes = [len(v[0]) for v in recorder.parts]
            offset, batches = 0, []
            for size in sizes:
                batches.append(slice(offset, offset+size))
                offset += size
            raw = {k:v.detach().clone()-base.state_dict()[k] for k,v in local.state_dict().items()}
            torch.save(dict(pre=base.state_dict(), raw=raw, transmitted=raw, rng=recorder.rng,
                batch_sizes=sizes, ordered_dataset_ids=ids, config=config), folder/'attacker_capture.pt')
            torch.save(dict(x=x,y=y), folder/'evaluator_targets.pt')
            t = time.perf_counter()
            replay = simulate(base, criterion, x, y, batches, recorder.rng)
            max_error = max(float((replay[k].detach()-raw[k]).abs().max()) for k in raw)
            dump(folder/'replay.json', dict(max_abs_error=max_error, n=len(x), fraud=int(y.sum()),
                steps=len(batches), seconds=time.perf_counter()-t, local_seconds=training_seconds,
                training_loss=training_loss))
            for k in raw:
                torch.testing.assert_close(replay[k], raw[k], atol=2e-7, rtol=2e-4)
            del replay
            print(f'client={client} records={len(x)} steps={len(batches)} replay={max_error}', flush=True)
            names = list(dict(base.named_parameters()))
            for restart in range(args.restarts):
                generator = torch.Generator().manual_seed(derive_seed(seed,'init',client,restart))
                initial = torch.randn(x.shape, generator=generator)
                prior = score(x,initial,y,metadata,folder/f'prior_{restart}.csv')
                for method in ('baseline','zero_update'):
                    observed = raw if method=='baseline' else {k:torch.zeros_like(v) for k,v in raw.items()}
                    dummy = initial.clone().requires_grad_(True)
                    opt = torch.optim.Adam([dummy], lr=.05)
                    best, candidate, best_step = float('inf'), None, None
                    history = []
                    t = time.perf_counter()
                    for iteration in range(args.iterations+1):
                        delta = simulate(base,criterion,dummy,y,batches,recorder.rng)
                        loss = objective(delta,observed,names)
                        if not torch.isfinite(loss):
                            raise RuntimeError('nonfinite objective')
                        value = float(loss.detach())
                        history.append(value)
                        if value < best:
                            best,candidate,best_step=value,dummy.detach().clone(),iteration
                        if iteration < args.iterations:
                            grad, = torch.autograd.grad(loss,dummy)
                            if not torch.isfinite(grad).all():
                                raise RuntimeError('nonfinite input gradient')
                            opt.zero_grad()
                            dummy.grad=grad
                            opt.step()
                        del delta, loss
                        if iteration % 10 == 0:
                            print(f'client={client} restart={restart} method={method} iteration={iteration}',flush=True)
                    elapsed = time.perf_counter()-t
                    path=folder/f'{method}_{restart}.pt'
                    torch.save(dict(initial=initial,candidate=candidate,history=history,best=best,best_step=best_step),path)
                    saved=torch.load(path,weights_only=False)
                    recomputed=float(objective(simulate(base,criterion,saved['candidate'],y,batches,recorder.rng),observed,names).detach())
                    if abs(recomputed-best)>1e-12+abs(best)*1e-5:
                        raise AssertionError('reload objective mismatch')
                    metrics=score(x,candidate,y,metadata,folder/f'{method}_{restart}.csv')
                    reports.append(dict(client=client,restart=restart,method=method,metrics=metrics,prior=prior,
                        seconds=elapsed,best_step=best_step,objective=best,reload_objective=recomputed))
                    dump(out/'progress.json',reports)
    except BaseException as exc:
        failures.append(dict(error=repr(exc)))
        raise
    finally:
        peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss if resource is not None else None
        dump(out/'compute_profile.json',dict(total_seconds=time.perf_counter()-start,
            process_peak_rss_bytes=None if peak is None else (peak if sys.platform=='darwin' else peak*1024),
            includes_restarts=True, includes_setup=True))
        dump(out/'update_attack_validation.json',dict(config=config,records=reports,failures=failures,
            status='FAILED' if failures else 'COMPLETED_EXECUTION_NOT_EFFECTIVENESS'))
        dump(out/'manifest.json',{str(f.relative_to(out)):checksum(f) for f in out.rglob('*') if f.is_file()})
        print(str(out),flush=True)


if __name__=='__main__':
    main()
