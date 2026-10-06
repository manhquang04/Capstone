"""Frozen P31 baseline replay; creates the missing checkpoint without tuning."""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import priority31_image_utility_dp as p31
import torch

OUT = ROOT / 'artifacts/priority33c'
FOLDER = OUT / 'checkpoint_replay'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            digest.update(chunk)
    return digest.hexdigest()


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def record(event, **details):
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'progress.log').open('a') as stream:
        stream.write(json.dumps(dict(at=time.time(), event=event, **details)) + '\n')


def finite(value, context):
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError('nonfinite: ' + context)


def check(model):
    for name, tensor in model.state_dict().items():
        if tensor.is_floating_point():
            finite(tensor, name)
        if name.endswith('running_var') and bool((tensor < 0).any()):
            raise FloatingPointError('negative BN variance: ' + name)


def accuracy(model, x, y):
    model.eval()
    correct = 0
    with torch.no_grad():
        for start in range(0, len(x), 512):
            output = model(x[start:start + 512])
            finite(output, 'evaluation logits')
            correct += int((output.argmax(1) == y[start:start + 512]).sum())
    model.train()
    return correct / len(x)


def main():
    torch.set_num_threads(1)
    started = time.time()
    original_path = p31.OUT / 'utility/baseline/seed_51016.json'
    sources = [Path(__file__), original_path, p31.OUT / 'split.json',
               ROOT / 'experiments/priority31_image_utility_dp.py',
               ROOT / 'protocols/amendments/2026-10-03_priority33c_image_coverage.md']
    fingerprints = {str(p.relative_to(ROOT)): sha(p) for p in sources}
    result_path = FOLDER / 'result.json'
    if result_path.exists():
        previous = json.loads(result_path.read_text())
        assert previous['status'] == 'passed' and previous['source_hashes'] == fingerprints
        assert sha(FOLDER / 'final_state.pt') == previous['checkpoint_sha256']
        record('validated_resume_skip', path=str(result_path))
        return
    record('checkpoint_replay_start', argv=sys.argv, source_hashes=fingerprints)
    save(OUT / 'progress.json', dict(stage='checkpoint_replay', dataset='CIFAR10',
         method='baseline', done=0, total=1, failed=0, started_at=started,
         last_update=started, eta_minutes=None))
    try:
        original = json.loads(original_path.read_text())
        job = original['job']
        assert (job['seed'], job['lr'], job['epochs']) == (51016, .1, 100)
        x, y, vx, vy = p31.utility_data()
        finite(x, 'train'); finite(vx, 'validation')
        model, _ = p31.dp.inversefed.construct_model('LeNetZhu', seed=42)
        assert not list(model.buffers()), 'unexpected LeNetZhu buffers'
        optimizer = torch.optim.SGD(model.parameters(), lr=.1, momentum=0)
        generator = torch.Generator().manual_seed(51016)
        loss_fn = torch.nn.CrossEntropyLoss()
        steps = 0
        for epoch in range(1, 101):
            order = torch.randperm(len(x), generator=generator)
            for start in range(0, len(x), 256):
                batch = order[start:start + 256]
                optimizer.zero_grad()
                logits = model(x[batch]); finite(logits, 'training logits')
                loss = loss_fn(logits, y[batch]); finite(loss, 'training loss')
                loss.backward()
                state = {name: p.grad.detach().clone() for name, p in model.named_parameters()}
                for name, parameter in model.named_parameters():
                    finite(state[name], 'gradient ' + name)
                    parameter.grad = state[name]
                optimizer.step(); check(model); steps += 1
            print('checkpoint replay epoch', epoch, flush=True)
        validation = accuracy(model, vx, vy)
        test = p31.dp.CIFAR10(str(ROOT / 'datasets/cifar10'), train=False, download=False)
        tx, ty = p31.tensors(test, list(range(10000)))
        testing = accuracy(model, tx, ty)
        assert validation == original['validation_accuracy'], (validation, original['validation_accuracy'])
        assert testing == original['test_accuracy'], (testing, original['test_accuracy'])
        assert validation >= .4
        FOLDER.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), FOLDER / 'final_state.pt')
        save(result_path, dict(status='passed', source_hashes=fingerprints,
             checkpoint_sha256=sha(FOLDER / 'final_state.pt'), steps=steps,
             validation_accuracy=validation, test_accuracy=testing,
             original_job=job, elapsed_seconds=time.time()-started))
        save(OUT / 'progress.json', dict(stage='checkpoint_replay_complete', dataset='CIFAR10',
             method='baseline', done=1, total=1, failed=0, started_at=started,
             last_update=time.time(), eta_minutes=0))
        record('checkpoint_replay_passed', validation=validation, test=testing)
    except BaseException as error:
        save(FOLDER / 'FAILED.json', dict(error=repr(error), source_hashes=fingerprints))
        save(OUT / 'progress.json', dict(stage='requires_direction', dataset='CIFAR10',
             method='baseline', done=0, total=1, failed=1, started_at=started,
             last_update=time.time(), eta_minutes=None))
        record('checkpoint_replay_failed', error=repr(error))
        raise


if __name__ == '__main__':
    main()
