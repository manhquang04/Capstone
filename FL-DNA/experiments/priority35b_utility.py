"""Utility-only P35B execution; frozen P34A training/P34B noise code reused.

No leakage allocation or recovery entry point. Runtime evidence stays in P35B.
"""
from __future__ import annotations
import argparse
import copy
import json
import multiprocessing as mp
import os
import random
import socket
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
from experiments import priority34b_local_dp as local
from experiments import priority35b_design as design
OUT = ROOT / 'artifacts/priority35b/utility'
PROTOCOL = ROOT / 'protocols/amendments/2026-10-10_priority35b_frontier_protocol.md'
PORT = 43572


def read(path): return json.loads(Path(path).read_text())
def sha(path): return local.p.sha(Path(path))


def save(path, doc):
    path = Path(path)
    if not path.resolve().is_relative_to((ROOT/'artifacts/priority35b').resolve()):
        raise ValueError('P35B output boundary')
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(doc, indent=2, sort_keys=True, allow_nan=False)+'\n')
    tmp.replace(path)


def stage(name, **fields):
    doc = dict(at=local.p.now(), stage=name, scope='utility only', leakage_runs=0,
               full_priority35b_complete=False, **fields)
    save(ROOT/'artifacts/priority35b/progress.json', doc)
    with (ROOT/'artifacts/priority35b/progress.log').open('a') as handle:
        handle.write(json.dumps(doc, allow_nan=False)+'\n')
    checklist = read(ROOT/'artifacts/priority35b/checklist.json')
    checklist['scope'] = 'utility authorized; leakage NOT_AUTHORIZED'
    checklist['utility'] = doc
    checklist['items']['measurements'] = 'UTILITY_ONLY'
    save(ROOT/'artifacts/priority35b/checklist.json', checklist)
    print(json.dumps(doc), flush=True)


def account(epsilon):
    result = design.account(epsilon)
    result.update(target_epsilon=epsilon,
        adjacency='per-client replace-one clipped update contribution, conditional on public history',
        release='one client:50 independently noised individual non-BN uploads before transmission',
        unit='one client contribution transcript across training; NOT record-level DP',
        mechanism_location='client before transmission',
        conversion='Gaussian RDP composed at q=1, standard delta conversion')
    return result


def configure():
    local.torch.set_num_threads(1)
    if local.torch.get_num_interop_threads() != 1:
        local.torch.set_num_interop_threads(1)
    local.OUT = OUT
    local.PROTOCOL = PROTOCOL
    local.account = account


def verify_freeze():
    frozen = read(OUT/'execution_freeze.json')
    for name, expected in {**frozen['sources'], **frozen['inputs']}.items():
        if sha(ROOT/name) != expected: raise ValueError('hash drift: '+name)
    return frozen


def train(job, config, key):
    """Same P34B tabular wrapper; only namespace and grid accountant differ."""
    configure()
    a, p = local.a, local.p
    from collections import OrderedDict
    config = copy.deepcopy(config)
    config.update(method=job['method'], protocol_variant='client_side_local_BN_DP',
                  dp_job=job, local_dp_protocol_sha256=sha(PROTOCOL))
    path = OUT/job['result']
    old_out, old_upload, old_average = a.OUT, a.upload, p.fed_avg
    audits, pending, context = [], [], {}
    def upload(model, previous, method, seed, round_number, client, bn, allowed):
        state = OrderedDict((n, model.state_dict()[n].detach().clone()-previous[n]) for n in allowed)
        a.assert_payload(state, bn, allowed)
        noisy, receipt = local.noise_upload(state, job['mechanism'], job['epsilon'], key, round_number, client)
        a.assert_payload(noisy, bn, allowed)
        wire = OrderedDict((n, previous[n]+noisy[n]) for n in allowed)
        a.assert_payload(wire, bn, allowed)
        receipt['wire_absolute_state_sha256'] = local.tensor_digest(wire)
        audits.append(receipt); pending.append(noisy)
        context.update(previous=previous, bn=bn, allowed=allowed)
        return wire
    def average(states, sizes):
        if len(pending) != len(states): raise ValueError('individual noise before aggregation')
        result = local.mean_noisy_uploads(pending, [s/sum(sizes) for s in sizes], context['previous'])
        a.assert_payload(result, context['bn'], context['allowed']); pending.clear()
        return result
    a.OUT, a.upload, p.fed_avg = OUT, upload, average
    try:
        a.train_job(config, str(path)); doc = read(path)
    finally:
        a.OUT, a.upload, p.fed_avg = old_out, old_upload, old_average
    doc.update(job=job, job_sha256=p.canonical_sha(job), local_upload_audits=audits,
               accounting=account(job['epsilon']), server_noise_added=False,
               inter_op_threads=local.torch.get_num_interop_threads())
    save(path, doc)
    return doc


def worker(job):
    configure()
    # The parent verifies all immutable sources/inputs before dispatch.
    frozen = read(OUT/'execution_freeze.json')
    if job not in frozen['jobs']: raise ValueError('job outside frozen registry')
    path = OUT/job['result']
    save(path.with_suffix('.process.json'), dict(pid=os.getpid(), job=job, at=local.p.now()))
    if path.with_suffix('.failure.json').exists(): raise ValueError('preserved failure requires direction')
    try:
        if local.valid(path, job): return dict(skipped=True)
        config = frozen['baseline_configs'][job['dataset']+'/'+str(job['seed'])]
        key = read(OUT/'private_noise_seeds.json')[job['result']]
        train(job, config, key)
        if not local.valid(path, job): raise ValueError('missing output')
        return dict(skipped=False)
    except Exception:
        save(path.with_suffix('.failure.json'), dict(job=job, at=local.p.now(), traceback=traceback.format_exc()))
        raise


def synthetic_identity_fixture(task):
    """Small synthetic two-round execution used ONLY by serial/spawn tests."""
    global OUT
    folder, output, mechanism = task
    previous_out, previous_data = OUT, local.p.OUT
    OUT = Path(output)
    local.p.OUT = Path(folder)
    try:
        job=dict(dataset='synthetic',stage='tabular',mechanism=mechanism,epsilon=30,
                 seed=321000,method=f'local_dp_{mechanism}_eps30',result='result.json')
        config=dict(dataset='synthetic',method='baseline',seed=321000,
                    training=dict(alpha=.95,lr=.001,rounds=2))
        doc=train(job,config,891012)
        checkpoint=next(n for n in doc['artifact_sha256'] if n.endswith('final_checkpoint.pt'))
        state=local.torch.load(OUT/checkpoint,map_location='cpu',weights_only=False)
        predictions={Path(n).name:sha(OUT/n) for n in doc['artifact_sha256'] if n.endswith('.npy')}
        result=dict(global_digest=local.tensor_digest(state['global_non_bn']),
                    bn_digests=[local.tensor_digest(x) for x in state['client_bn']],
                    predictions=predictions,uploads=doc['local_upload_audits'],
                    validation=doc['validation'],test=doc['test'],partition=doc['partition_sha256'],
                    threads=doc['torch_threads'],inter_op=doc['inter_op_threads'])
        return result
    finally:
        OUT, local.p.OUT = previous_out, previous_data


def prepare():
    if (OUT/'execution_freeze.json').exists(): raise ValueError('freeze exists; no overwrite')
    stage('UTILITY_PREPARING', new_jobs=330, reuse_jobs=297)
    # Authorization and implementation are additive execution records, not amendment edits.
    save(OUT/'execution_scope.json', dict(at=local.p.now(), scope='utility only',
        authorization='User requested Config-C utility grid, process pool, tables and matching; stop afterwards',
        scientific_protocol_sha256=sha(PROTOCOL),
        delta_auc='ROC-AUC; PR-AUC additionally descriptive', leakage_authorized=False,
        algorithm='P34A train_job plus same P34B clipping/noise/weighted aggregation wrapper',
        infrastructure='spawn pool max4, thread1; same frozen keys for serial/pool; no scientific changes'))
    from experiments.priority35b_preflight import prior_inputs
    prior = prior_inputs()
    expected = read(ROOT/'artifacts/priority35b/protocol_freeze.json')['sha256']
    if sha(PROTOCOL) != expected: raise ValueError('protocol hash drift')
    save(OUT/'reuse_registry.json', prior)
    checks = []
    for command in ([sys.executable, '-B', '-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_priority35b*.py', '-v'],
                    ['git', 'diff', '--check']):
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        checks.append(dict(command=command, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr))
        save(OUT/'preflight_checks.json', checks)
        if result.returncode: raise ValueError('preflight failed; logs retained')
    import py_compile
    for path in list((ROOT/'experiments').glob('priority35b_*.py'))+list((ROOT/'tests').glob('test_priority35b*.py')):
        py_compile.compile(str(path), cfile=str(OUT/'compile'/ (path.stem+'.pyc')), doraise=True)
    jobs = [dict(dataset=d, stage='tabular', mechanism=c, epsilon=e, seed=s,
                 method=f'local_dp_{c}_eps{e}', result=f'jobs/{d}/local_dp_{c}_eps{e}/{s}.json')
            for d in design.DATASETS for c in design.CLIPS for e in design.EPSILONS[3:] for s in design.SEEDS]
    assert len(jobs) == 330
    configs = {d+'/'+str(s):read(ROOT/f'artifacts/priority34a/jobs/{d}/baseline/{s}.json')['config']
               for d in design.DATASETS for s in design.SEEDS}
    keys = {j['result']:int.from_bytes(os.urandom(8),'big')&((1<<63)-1) for j in jobs}
    if (OUT/'private_noise_seeds.json').exists(): raise ValueError('keys already exist')
    save(OUT/'private_noise_seeds.json', keys); (OUT/'private_noise_seeds.json').chmod(0o600)
    sources = dict(read(ROOT/'artifacts/priority34b_local_dp/execution_freeze.json')['sources'])
    for path in list((ROOT/'experiments').glob('priority35b_*.py'))+list((ROOT/'tests').glob('test_priority35b*.py'))+[PROTOCOL]:
        sources[str(path.relative_to(ROOT))] = sha(path)
    inputs = dict(prior['hashes'])
    for path in (OUT/'private_noise_seeds.json',OUT/'execution_scope.json',OUT/'reuse_registry.json',OUT/'preflight_checks.json'):
        inputs[str(path.relative_to(ROOT))] = sha(path)
    save(OUT/'execution_freeze.json', dict(at=local.p.now(), jobs=jobs, workers=4, sources=sources,
        inputs=inputs, baseline_configs=configs, device='cpu', threads=1, inter_op_threads=1,
        protocol_sha256=expected, scope='UTILITY_ONLY'))
    verify_freeze(); stage('UTILITY_FROZEN', new_jobs=330, reused=297, done=0, failed=0)


def ownership():
    sock = socket.socket()
    try: sock.bind(('127.0.0.1', PORT)); sock.listen(1)
    except Exception: sock.close(); raise RuntimeError('supervisor active/ownership unavailable')
    return sock


def supervise():
    configure()
    with ownership():
        save(OUT/'supervisor.lock.json', dict(pid=os.getpid(), at=local.p.now()))
        frozen = verify_freeze(); done = 0; missing = []
        if (OUT/'REQUIRES_DIRECTION.json').exists(): raise ValueError('unresolved failure')
        for job in frozen['jobs']:
            if (OUT/job['result']).with_suffix('.failure.json').exists(): raise ValueError('failure retained')
            if local.valid(OUT/job['result'],job): done += 1
            else: missing.append(job)
        random.Random(353500).shuffle(missing)
        failures = []; start = time.time(); initial = done
        stage('UTILITY_RUNNING', done=done, total=330, reused=297, failed=0, supervisor=os.getpid())
        with ProcessPoolExecutor(max_workers=4, mp_context=mp.get_context('spawn')) as pool:
            futures = {pool.submit(worker, job):job for job in missing}
            for future in as_completed(futures):
                job = futures[future]
                try: future.result(); done += 1
                except Exception:
                    failures.append(dict(job=job, traceback=traceback.format_exc()))
                elapsed = time.time()-start
                finished = done-initial+len(failures)
                stage('UTILITY_RUNNING',done=done,total=330,reused=297,failed=len(failures),last_job=job,
                      eta_minutes=elapsed/max(1,finished)*(330-done-len(failures))/60)
        if failures:
            save(OUT/'REQUIRES_DIRECTION.json',dict(failures=failures,done=done,total=330))
            stage('UTILITY_REQUIRES_DIRECTION',done=done,total=330,failed=len(failures)); return
        stage('UTILITY_VERIFYING',done=330,total=330,failed=0)
        from experiments.priority35b_utility_analysis import finish
        finish()
        stage('UTILITY_VERIFIED_TABLES_READY',done=330,total=330,reused=297,failed=0,
              stop='utility only; await user; no leakage')


def live():
    text = subprocess.check_output(['ps','-axo','pid=,command='],text=True)
    result = []
    for line in text.splitlines():
        fields=line.strip().split(None,1)
        if len(fields)!=2 or int(fields[0])==os.getpid(): continue
        executable=Path(fields[1].split()[0]).name.lower()
        if 'python' in executable and 'priority35b_utility.py' in fields[1] and '--supervise' in fields[1]: result.append(line)
    if (OUT/'supervisor.lock.json').exists():
        try: os.kill(read(OUT/'supervisor.lock.json')['pid'],0)
        except ProcessLookupError: pass
        else: result.append('recorded supervisor PID alive')
    for path in OUT.glob('jobs/**/*.process.json'):
        try: os.kill(read(path)['pid'],0)
        except ProcessLookupError: pass
        else: result.append('recorded worker PID alive')
    return result


def launch(resume=False):
    verify_freeze()
    if live(): raise ValueError('live supervisor/worker; refuse duplicate')
    with ownership(): pass
    if (OUT/'REQUIRES_DIRECTION.json').exists() or (OUT/'UTILITY_COMPLETE.json').exists(): raise ValueError('failure or complete')
    prior=list(OUT.glob('launch_receipt_*.json'))
    if bool(prior)!=resume: raise ValueError('resume requires explicit infrastructure review')
    if resume:
        configure()
        for job in read(OUT/'execution_freeze.json')['jobs']:
            if (OUT/job['result']).with_suffix('.failure.json').exists():
                raise ValueError('scientific failure: no resume')
            local.valid(OUT/job['result'],job)  # hash-validate every completed skip
        if list(OUT.glob('FAILURE_*.json')):
            raise ValueError('supervisor error requires separate direction, not infrastructure resume')
        save(OUT/f'interruption_review_{time.time_ns()}.json',dict(at=local.p.now(),
            reason='explicit --resume infrastructure interruption review',live_workers=[],
            policy='same frozen registry/keys; completed hashes validated; all attempt journals retained'))
    command=[sys.executable,'-B','-u',str(Path(__file__).resolve()),'--supervise']
    with (OUT/'detached_stdout.log').open('a') as stdout,(OUT/'detached_stderr.log').open('a') as stderr:
        process=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,
                                 start_new_session=True,close_fds=True)
    receipt=dict(pid=process.pid,at=local.p.now(),command=command,freeze_sha256=sha(OUT/'execution_freeze.json'))
    save(OUT/f'launch_receipt_{time.time_ns()}.json',receipt); print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for flag in ('prepare','launch','supervise','resume'): parser.add_argument('--'+flag,action='store_true')
    args=parser.parse_args()
    try:
        if args.prepare: prepare()
        elif args.launch: launch(args.resume)
        elif args.supervise: supervise()
        else: parser.error('select utility mode')
    except Exception:
        save(OUT/f'FAILURE_{time.time_ns()}.json',dict(at=local.p.now(),traceback=traceback.format_exc()))
        stage('UTILITY_FAILED',error='preserved failure; requires direction'); raise
