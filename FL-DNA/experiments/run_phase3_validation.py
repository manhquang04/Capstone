"""Bounded CPU Adam update capture, replay and inversion validation."""
import copy
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from attacks.local_update import simulate, objective
from experiments.fraud_fl_common import BinaryFocalLoss
from models.fraud_mlp import FraudMLP


def main():
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[1]
    source = root / 'artifacts/phase2/phase2_verified_v2'
    out = root / 'artifacts/phase3' / datetime.now(timezone.utc).strftime('run_%Y%m%dT%H%M%S%fZ')
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    targets = torch.load(source / 'targets.pt', weights_only=False, map_location='cpu')
    x, y = targets['features'][:4].float(), targets['labels'][:4].float().reshape(-1, 1)
    model = FraudMLP(x.shape[1]).train()
    model.load_state_dict(torch.load(source / 'warmup_model.pt', weights_only=False, map_location='cpu'))
    criterion = BinaryFocalLoss()
    iterations = int(os.environ.get('PHASE3_ITERATIONS', '30'))
    seed = int(os.environ['PHASE3_SEED']) if 'PHASE3_SEED' in os.environ else int.from_bytes(os.urandom(4), 'little')
    torch.manual_seed(seed)
    rng = torch.get_rng_state()
    initial = torch.randn_like(x)
    config = dict(seed=seed, iterations=iterations, restarts=1, device='cpu', model_mode='train',
                  optimizer='Adam', lr=.001, betas=[.9,.999], eps=1e-8, attacker_lr=.05,
                  known_labels=True, known_batch_order=True, known_dropout_rng=True,
                  source=str(source), targets='first four Phase2 targets reused as development',
                  scope='bounded local-set update; not official full-client evaluation')
    (out / 'protocol_lock.json').write_text(json.dumps(config, indent=2))
    reports = []
    for steps in (1, 2):
        batches = [list(range(4))] if steps == 1 else [[0,1], [2,3]]
        local = copy.deepcopy(model)
        opt = torch.optim.Adam(local.parameters(), lr=.001)
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(rng)
            for ids in batches:
                opt.zero_grad()
                criterion(local(x[ids]), y[ids]).backward()
                opt.step()
        raw = {k: v.detach() - model.state_dict()[k] for k,v in local.state_dict().items()}
        replay = simulate(model, criterion, x, y, batches, rng)
        errors = {k: float((replay[k].detach()-raw[k]).abs().max()) for k in raw}
        plain = simulate(model, criterion, x, y, batches, rng, create_graph=False)
        plain_errors = {k: float((plain[k].detach()-raw[k]).abs().max()) for k in raw}
        (out / f'replay_diagnostics_{steps}.json').write_text(json.dumps(dict(
            differentiable_errors=errors, ordinary_errors=plain_errors), indent=2))
        torch.save(dict(pre=model.state_dict(), raw=raw, transmitted=raw, x=x, y=y,
                        batches=batches, rng=rng, config=config, source_rows=targets['source_rows'][:4]), out / f'capture_{steps}.pt')
        for k in raw:
            torch.testing.assert_close(replay[k], raw[k], atol=2e-7, rtol=2e-4)
        torch.save(dict(pre=model.state_dict(), raw=raw, transmitted=raw, x=x, y=y,
                        batches=batches, rng=rng, config=config, source_rows=targets['source_rows'][:4]), out / f'capture_{steps}.pt')
        names = list(dict(model.named_parameters()))
        for method in ('baseline', 'zero_update'):
            signal = raw if method == 'baseline' else {k: torch.zeros_like(v) for k,v in raw.items()}
            dummy = initial.clone().requires_grad_(True)
            attack_opt = torch.optim.Adam([dummy], lr=.05)
            best, candidate, history = float('inf'), None, []
            attack_start = time.perf_counter()
            for iteration in range(iterations + 1):
                delta = simulate(model, criterion, dummy, y, batches, rng)
                loss = objective(delta, signal, names)
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite objective')
                value = float(loss.detach())
                history.append(value)
                if value < best:
                    best, candidate = value, dummy.detach().clone()
                if iteration < iterations:
                    attack_opt.zero_grad()
                    grad, = torch.autograd.grad(loss, dummy)
                    if not torch.isfinite(grad).all():
                        raise RuntimeError('Nonfinite input derivative')
                    dummy.grad = grad
                    attack_opt.step()
            path = out / f'{method}_{steps}.pt'
            torch.save(dict(original=x, initial=initial, candidate=candidate, history=history, best=best), path)
            saved = torch.load(path, weights_only=False)
            check = float(objective(simulate(model, criterion, saved['candidate'], y, batches, rng), signal, names).detach())
            if abs(check-best) > 1e-12 + abs(best)*1e-5:
                raise AssertionError('Reloaded candidate objective mismatch')
            reports.append(dict(steps=steps, method=method, mse=float((candidate-x).square().mean()),
                                prior_mse=float((initial-x).square().mean()), best_objective=best,
                                replay_max_error=max(errors.values()), reload_objective=check,
                                seconds=time.perf_counter()-attack_start, target_count=len(x)))
    payload = dict(config=config, records=reports, seconds=time.perf_counter()-started,
                   limitations=['one restart', 'reused development targets', 'parameter-only attack objective; buffers captured',
                                'known RNG is privileged', 'no full-client inversion', 'no DNA ranking'])
    (out/'update_attack_validation.json').write_text(json.dumps(payload, indent=2))
    print(json.dumps(dict(output=str(out), **payload), indent=2))


if __name__ == '__main__':
    main()
