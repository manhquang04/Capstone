"""P34D four-start-state utility calibration; no recovery target access."""
import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import priority33c_image_utility as old
from experiments.priority33c_checkpoint_replay import finite, check, accuracy
import numpy as np
import torch

torch.set_num_threads(1)
if torch.get_num_interop_threads() != 1:
    torch.set_num_interop_threads(1)
assert torch.get_num_threads() == torch.get_num_interop_threads() == 1
OUT = ROOT / 'artifacts/priority34d'
BASE = OUT / 'image_utility'
ANNEX = ROOT / 'protocols/amendments/2026-10-04_priority34d_image_utility_execution.md'
PREP = BASE / 'preparation_freeze.json'
FREEZE = BASE / 'execution_freeze.json'
SEEDS = list(range(51016, 51032))
SETTINGS = {'init342042':342042, 'init342043':342043, 'init342044':342044,
            'trained_batch4':42}
DNA = ['dna_v1_conservative', 'dna_v2_0p95']
GRID = [.0001, .0003, .001, .003, .01, .03, .1, .3]
EXTENSION = [1., 3., 10.]
CHECKPOINT = ROOT / 'artifacts/priority33c/checkpoint_replay/final_state.pt'
CHECKPOINT_SHA = '6d390aaf2bbb52aab0bc5991a5c682bf1cd88cf3d170957edfe8fbbd834d36c5'


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1048576), b''):
            value.update(chunk)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    path = Path(path)
    if not path.resolve().is_relative_to(OUT.resolve()):
        raise ValueError('P34D write firewall')
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False)+'\n')
    temp.replace(path)


def append(path, obj):
    if not Path(path).resolve().is_relative_to(OUT.resolve()):
        raise ValueError('P34D append firewall')
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open('a') as stream:
        stream.write(json.dumps(obj, allow_nan=False)+'\n')


def live():
    rows = subprocess.check_output(['ps', '-axo', 'pid,command'], text=True).splitlines()
    return [r for r in rows if 'priority34d_image_utility.py' in r
            and ('--supervise' in r or '--job' in r)
            and int(r.split()[0]) != os.getpid()]


def verify(path=FREEZE):
    doc = read(path)
    for name, digest in {**doc['sources'], **doc['inputs']}.items():
        if sha(ROOT/name) != digest:
            raise ValueError('utility freeze mismatch: '+name)
    return doc


def starting_model(setting):
    if setting not in SETTINGS:
        raise ValueError('unknown setting')
    model, _ = old.p31.dp.inversefed.construct_model('LeNetZhu', seed=SETTINGS[setting])
    assert not list(model.buffers()), 'unexpected BN/buffers'
    if setting == 'trained_batch4':
        assert sha(CHECKPOINT) == CHECKPOINT_SHA
        model.load_state_dict(torch.load(CHECKPOINT, map_location='cpu', weights_only=False))
    check(model)
    return model


def probe(model, x, y):
    names = [n for n, _ in model.named_parameters()]
    norms = {n:[] for n in names}
    global_norms = []
    model.train()
    for seed in (51000, 51003):
        order = torch.randperm(len(x), generator=torch.Generator().manual_seed(seed))
        for start in range(0, len(x), 256):
            batch = order[start:start+256]
            model.zero_grad()
            logits = model(x[batch]); finite(logits, 'probe logits')
            loss = torch.nn.functional.cross_entropy(logits, y[batch])
            finite(loss, 'probe loss'); loss.backward()
            gradients = []
            for name, param in model.named_parameters():
                finite(param.grad, 'probe gradient '+name)
                gradients.append(param.grad.detach())
                norms[name].append(float(param.grad.norm()))
            global_norms.append(float(old.p31.dp.flatten(gradients).norm()))
    clips = [float(np.quantile(norms[n], .95)) for n in names]
    clip = float(np.quantile(global_norms, .95))
    if not all(np.isfinite(v) and v > 0 for v in clips+[clip]):
        raise FloatingPointError('nonpositive/nonfinite clip')
    return dict(names=names, clips=clips, C=clip, norms=norms,
                global_norms=global_norms, probe_seeds=[51000,51003],
                training_batch=256, quantile=.95)


def prepare_freeze():
    if live() or PREP.exists():
        raise RuntimeError('active/existing preparation; never overwrite')
    sources = {Path(__file__).resolve(), ANNEX,
               ROOT/'tests/test_priority34d_image_utility.py',
               ROOT/'protocols/amendments/2026-10-04_priority34d_checkpoint_robustness.md',
               ROOT/'protocols/amendments/2026-10-04_priority34d_direction_and_preflight.md',
               ROOT/'protocols/amendments/2026-10-04_priority34d_test_process_isolation.md'}
    for module in list(sys.modules.values()):
        raw = getattr(module, '__file__', None)
        if raw:
            path = Path(raw).resolve()
            if path.is_relative_to(ROOT) and '.venv' not in str(path) and path.suffix == '.py':
                sources.add(path)
    inputs = [old.p31.OUT/'split.json', CHECKPOINT,
              OUT/'CHECKPOINTS_COMPLETE.json', OUT/'checkpoint_execution_freeze.json',
              ROOT/'artifacts/priority34c/COMPLETE.json',
              ROOT/'artifacts/priority34b_local_dp/COMPLETE.json']
    inputs += [p for p in (ROOT/'datasets/cifar10').rglob('*') if p.is_file()]
    assert sha(CHECKPOINT) == CHECKPOINT_SHA
    assert len(inputs) > 6, 'CIFAR inputs missing'
    write(PREP, dict(status='FROZEN_BEFORE_PROBES', at=time.time(),
          settings=SETTINGS, seeds=SEEDS, grid=GRID, extension=EXTENSION,
          sources={str(p.relative_to(ROOT)):sha(p) for p in sorted(sources)},
          inputs={str(p.relative_to(ROOT)):sha(p) for p in sorted(inputs)}))
    print(json.dumps(dict(preparation_sha256=sha(PREP), sources=len(sources), inputs=len(inputs))))


def prepare_probes():
    verify(PREP)
    if FREEZE.exists():
        verify(); return
    if (BASE/'PROBE_FAILURE.json').exists():
        raise RuntimeError('preserved probe failure; require direction')
    try:
        x, y, vx, vy = old.p31.utility_data()
        x, vx = x.contiguous(), vx.contiguous()
        finite(x, 'probe data'); finite(vx, 'validation data')
        for setting in SETTINGS:
            folder = BASE/setting
            folder.mkdir(parents=True, exist_ok=True)
            if (folder/'initial_state.pt').exists() or (folder/'clip.json').exists():
                raise RuntimeError('preserved partial probes; require direction')
            model = starting_model(setting)
            torch.save(model.state_dict(), folder/'initial_state.pt')
            clips = probe(model, x, y)
            clips.update(setting=setting, initial_state_sha256=sha(folder/'initial_state.pt'),
                         preparation_sha256=sha(PREP), split_sha256=sha(old.p31.OUT/'split.json'))
            write(folder/'clip.json', clips)
        doc = read(PREP)
        doc['inputs'][str(PREP.relative_to(ROOT))] = sha(PREP)
        for setting in SETTINGS:
            for name in ('initial_state.pt', 'clip.json'):
                path = BASE/setting/name
                doc['inputs'][str(path.relative_to(ROOT))] = sha(path)
        doc.update(status='FROZEN_BEFORE_TRAINING', at=time.time(), jobs=jobs())
        write(FREEZE, doc)
        print(json.dumps(dict(training_freeze_sha256=sha(FREEZE), initial_jobs=len(jobs()))))
    except Exception as error:
        write(BASE/'PROBE_FAILURE.json', dict(error=repr(error), at=time.time()))
        raise


def jobs(grid=GRID, mechanisms=('single','per_tensor'), settings=None, core=True):
    result = []
    for setting in (SETTINGS if settings is None else settings):
        methods = [('baseline', None, None)]+[(name,None,None) for name in DNA] if core else []
        methods += [(mechanism+'_dp_'+format(sigma,'g'), mechanism, sigma)
                    for mechanism in mechanisms for sigma in grid]
        for method, mechanism, sigma in methods:
            for seed in SEEDS:
                result.append(dict(id=setting+'__'+method+'__'+str(seed), setting=setting,
                              method=method, mechanism=mechanism, sigma=sigma, seed=seed))
    return result


def valid(job):
    folder = BASE/'jobs'/job['id']
    receipt = folder/'validated.json'
    if not receipt.exists():
        return False
    doc = read(receipt)
    if doc['job'] != job or doc['freeze_sha256'] != sha(FREEZE) or doc['status'] != 'PASS':
        raise ValueError('invalid utility receipt')
    for name, digest in doc['outputs'].items():
        if sha(folder/name) != digest:
            raise ValueError('utility result hash mismatch')
    return True


def train(job, attempt):
    verify()
    if valid(job):
        return
    folder = BASE/'jobs'/job['id']
    attempt_folder = folder/attempt
    attempt_folder.mkdir(parents=True, exist_ok=True)
    result_path = attempt_folder/'result.json'
    if result_path.exists() or (attempt_folder/'epochs.jsonl').exists():
        raise RuntimeError('attempt already exists; no overwrite')
    started = time.time()
    try:
        x, y, vx, vy = old.p31.utility_data()
        x, vx = x.contiguous(), vx.contiguous()
        finite(x,'train'); finite(vx,'validation')
        model = starting_model(job['setting'])
        model.load_state_dict(torch.load(BASE/job['setting']/'initial_state.pt',
                                        map_location='cpu', weights_only=False))
        clip = read(BASE/job['setting']/'clip.json')
        assert clip['names'] == [n for n,_ in model.named_parameters()]
        optimizer = torch.optim.SGD(model.parameters(), lr=.1, momentum=0)
        generator = torch.Generator().manual_seed(job['seed'])
        step = 0
        for epoch in range(1,101):
            order = torch.randperm(len(x), generator=generator)
            for start in range(0,len(x),256):
                batch = order[start:start+256]
                optimizer.zero_grad()
                logits = model(x[batch]); finite(logits,'training logits')
                loss = torch.nn.functional.cross_entropy(logits,y[batch])
                finite(loss,'training loss'); loss.backward()
                state = {n:p.grad.detach().clone() for n,p in model.named_parameters()}
                for name, value in state.items():
                    finite(value,'raw gradient '+name)
                if job['method'] == DNA[0]:
                    state = old.transform(state,DNA[0])
                elif job['method'] == DNA[1]:
                    state, _ = old.p31.adapters.dna_v2_gradient(state)
                elif job['mechanism'] is not None:
                    noise_seed = job['seed']*100000+step
                    if job['mechanism'] == 'single':
                        values = old.p31.dp.add_clipped_noise(list(state.values()),clip['C'],
                                                             job['sigma'],noise_seed)
                    else:
                        values = old.per_tensor_noise(list(state.values()),clip['clips'],
                                                      job['sigma'],noise_seed)
                    state = dict(zip(state,values))
                elif job['method'] != 'baseline':
                    raise ValueError('unknown method')
                for name, parameter in model.named_parameters():
                    finite(state[name],'transmitted gradient '+name)
                    parameter.grad = state[name]
                optimizer.step(); check(model); step += 1
            if epoch%10 == 0:
                value = accuracy(model,vx,vy)
                append(attempt_folder/'epochs.jsonl',dict(epoch=epoch,validation_accuracy=value,
                                                          at=time.time()))
                print(job['id'],epoch,value,flush=True)
        result = dict(status='PASS',job=job,validation_accuracy=value,steps=step,
                      elapsed_seconds=time.time()-started,device='cpu',
                      intra_threads=torch.get_num_threads(),inter_threads=torch.get_num_interop_threads(),
                      freeze_sha256=sha(FREEZE),test_selection=False)
        write(result_path,result)
        torch.save(model.state_dict(),attempt_folder/'final_state.pt')
        write(folder/'validated.json',dict(status='PASS',job=job,freeze_sha256=sha(FREEZE),
              outputs={str(p.relative_to(folder)):sha(p) for p in attempt_folder.iterdir()
                       if p.is_file() and p.name not in ('stdout.log','stderr.log')}))
    except Exception as error:
        write(attempt_folder/'failure.json',dict(status='FAILED',job=job,error=repr(error),
                                                at=time.time(),freeze_sha256=sha(FREEZE)))
        raise


def execute(job):
    if valid(job):
        return dict(job=job,status='PASS',skipped=True)
    folder = BASE/'jobs'/job['id']
    if list(folder.glob('attempt_*/failure.json')):
        raise RuntimeError('scientific/technical failure preserved; no replay without direction')
    attempt = 'attempt_'+str(time.time_ns())
    attempt_folder = folder/attempt
    attempt_folder.mkdir(parents=True,exist_ok=True)
    append(OUT/'runs.jsonl',dict(event='image_utility_started',job=job,attempt=attempt,at=time.time()))
    with (attempt_folder/'stdout.log').open('a') as so, (attempt_folder/'stderr.log').open('a') as se:
        proc = subprocess.run([sys.executable,'-B','-u',str(Path(__file__).resolve()),
                               '--job',job['id'],'--attempt',attempt],cwd=ROOT,stdout=so,stderr=se)
    status = 'PASS' if proc.returncode == 0 and valid(job) else 'FAILED'
    row = dict(job=job,status=status,returncode=proc.returncode,attempt=attempt,at=time.time())
    append(OUT/'runs.jsonl',dict(event='image_utility_finished',**row))
    return row


def values(setting, method):
    result = []
    for job in jobs(GRID+EXTENSION):
        if job['setting'] != setting or job['method'] != method:
            continue
        if not valid(job):
            raise RuntimeError('missing validated utility job')
        doc = read(BASE/'jobs'/job['id']/'validated.json')
        paths = [name for name in doc['outputs'] if name.endswith('/result.json')]
        assert len(paths) == 1
        result.append(read(BASE/'jobs'/job['id']/paths[0])['validation_accuracy'])
    assert len(result) == 16
    return np.array(result)


def bracket(baseline, target, means):
    if not np.isfinite(baseline) or not np.isfinite(target) or not all(np.isfinite(v) for v in means.values()):
        raise FloatingPointError('nonfinite calibration')
    threshold = target-.005
    eligible = [s for s,v in means.items() if v >= threshold]
    sigma = max(eligible) if eligible else None
    ordered = sorted(means)
    next_sigma = ordered[ordered.index(sigma)+1] if sigma is not None and sigma != ordered[-1] else None
    matched = baseline >= .40 and next_sigma is not None and means[next_sigma] < threshold
    return dict(status='MATCHED' if matched else 'NOT_ASSESSABLE',sigma=sigma,
                next_sigma=next_sigma,target_delta=target,threshold=threshold,grid_means=means,
                baseline_mean=baseline,extension_needed=baseline >= .40 and sigma == ordered[-1])


def selection(extensions=()):
    matches = {}
    for setting in SETTINGS:
        baseline = values(setting,'baseline')
        for mechanism in ('single','per_tensor'):
            grid = GRID+EXTENSION if (setting,mechanism) in extensions else GRID
            means = {s:float(np.mean(values(setting,mechanism+'_dp_'+format(s,'g'))-baseline)) for s in grid}
            for method in DNA:
                target = float(np.mean(values(setting,method)-baseline))
                matches[setting+'::'+method+'::'+mechanism] = bracket(float(baseline.mean()),target,means)
    return matches


def pool_run(scheduled, stage):
    started = time.time(); rows = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(execute,job):job for job in scheduled}
        for future in as_completed(futures):
            try:
                row = future.result()
            except Exception as error:
                row = dict(job=futures[future],status='FAILED',error=repr(error),at=time.time())
            rows.append(row)
            progress = dict(part='image_utility_calibration',stage=stage,done=len(rows),
                total=len(scheduled),failed=sum(r['status']!='PASS' for r in rows),last_update=time.time(),
                elapsed_seconds=time.time()-started)
            write(OUT/'progress.json',progress); append(OUT/'progress.log',progress)
            print(stage,len(rows),'/',len(scheduled),'failed',progress['failed'],flush=True)
    failed = [r for r in rows if r['status'] != 'PASS']
    if failed:
        write(BASE/'REQUIRES_DIRECTION.json',dict(failed=failed,pool_drained=True,stage=stage))
        raise RuntimeError('utility failures: direction required')


def supervise():
    guard = socket.socket(); guard.bind(('127.0.0.1',43467)); guard.listen(1)
    if live():
        raise RuntimeError('utility workers active')
    verify()
    write(BASE/'supervisor.lock.json',dict(pid=os.getpid(),at=time.time()))
    checklist = read(OUT/'checklist.json')
    checklist['image_utility_calibration']['status']='in_progress'
    write(OUT/'checklist.json',checklist)
    pool_run(read(FREEZE)['jobs'],'initial_grid')
    matches = selection()
    extension_pairs = sorted({(key.split('::')[0],key.split('::')[2])
                              for key,row in matches.items() if row['extension_needed']})
    if extension_pairs:
        path = BASE/'conditional_extension_schedule.json'
        schedule = [job for setting, mechanism in extension_pairs
                    for job in jobs(EXTENSION,(mechanism,),[setting],False)]
        doc = dict(freeze_sha256=sha(FREEZE),pairs=extension_pairs,jobs=schedule,
                   trigger_matches=matches,rule='preregistered top-grid passing only')
        if path.exists():
            assert read(path) == doc, 'conditional schedule changed'
        else:
            write(path,doc)
        pool_run(schedule,'conditional_extension')
        matches = selection(extension_pairs)
    write(BASE/'calibration.json',dict(matches=matches,freeze_sha256=sha(FREEZE),
          stage_only=True,at=time.time(),initial_jobs=len(jobs()),conditional_pairs=extension_pairs))
    write(BASE/'UTILITY_COMPLETE.json',dict(stage_only=True,at=time.time(),
          freeze_sha256=sha(FREEZE),calibration_sha256=sha(BASE/'calibration.json')))
    checklist['image_utility_calibration']['status']='completed'
    write(OUT/'checklist.json',checklist)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for flag in ('freeze-preparation','prepare-probes','launch','supervise','resume'):
        parser.add_argument('--'+flag,action='store_true')
    parser.add_argument('--job'); parser.add_argument('--attempt')
    args = parser.parse_args()
    if args.freeze_preparation:
        prepare_freeze()
    elif args.prepare_probes:
        prepare_probes()
    elif args.job:
        scheduled = read(FREEZE)['jobs']
        extension = BASE/'conditional_extension_schedule.json'
        if extension.exists():
            scheduled += read(extension)['jobs']
        selected = [job for job in scheduled if job['id'] == args.job]
        assert len(selected) == 1 and args.attempt
        train(selected[0],args.attempt)
    elif args.supervise:
        try:
            supervise()
        except Exception as error:
            write(BASE/'SUPERVISOR_FAILURE.json',dict(error=repr(error),at=time.time()))
            raise
    elif args.launch:
        assert not live(), 'no duplicate active workloads'
        assert not (BASE/'UTILITY_COMPLETE.json').exists()
        assert not (BASE/'REQUIRES_DIRECTION.json').exists()
        assert not (BASE/'SUPERVISOR_FAILURE.json').exists()
        if list(BASE.glob('launch_*.json')) and not args.resume:
            raise RuntimeError('previous launch; verified infrastructure resume only')
        verify()
        with (BASE/'detached_stdout.log').open('a') as so, (BASE/'detached_stderr.log').open('a') as se:
            proc = subprocess.Popen([sys.executable,'-B','-u',str(Path(__file__).resolve()),
                                     '--supervise'],cwd=ROOT,stdout=so,stderr=se,start_new_session=True)
        receipt = dict(pid=proc.pid,at=time.time(),freeze_sha256=sha(FREEZE),resume=args.resume)
        write(BASE/('launch_'+str(proc.pid)+'.json'),receipt)
        print(json.dumps(receipt))
    else:
        parser.error('choose freeze-preparation/prepare-probes/launch/supervise/job')
