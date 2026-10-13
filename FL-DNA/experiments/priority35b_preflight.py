"""Preparation ONLY: tests/read-only evidence audit and P35B readiness freeze.

There is deliberately no launch, training, target allocator or recovery entry.
"""
from __future__ import annotations
import datetime
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
from experiments import priority35b_design as design
from experiments import priority35b_firewall as firewall
firewall.previous.torch.set_num_threads(1)
OUT=ROOT/'artifacts/priority35b'
PROTOCOL=ROOT/'protocols/amendments/2026-10-10_priority35b_frontier_protocol.md'


def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(path):return json.loads(path.read_text())
def sha(path):return firewall.previous.sha(path)


def write(path, doc):
    if not path.resolve().is_relative_to(OUT.resolve()):raise ValueError('P35B output boundary')
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(doc,indent=2,sort_keys=True,allow_nan=False)+'\n')


def stage(name, **fields):
    row=dict(at=now(),stage=name,measurement_runs=0,new_real_target_allocations=0,
             full_priority35b_complete=False,**fields)
    write(OUT/'progress.json',row)
    with (OUT/'progress.log').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
    print(name,flush=True)


def prior_inputs():
    manifests={label:read(ROOT/path) for label,path in (
        ('a','artifacts/priority34a/sha256_manifest_final.json'),
        ('b','artifacts/priority34b_local_dp/sha256_manifest_final.json'))}
    inputs={};registry=[]
    def check(path,expected):
        relative=str(path.relative_to(ROOT))
        if relative not in inputs:
            actual=sha(path)
            if actual!=expected:raise ValueError('historical hash drift: '+relative)
            inputs[relative]=actual
        elif inputs[relative]!=expected:raise ValueError('inconsistent expected hash')
    for dataset in design.DATASETS:
        for seed in design.SEEDS:
            base_path=ROOT/f'artifacts/priority34a/jobs/{dataset}/baseline/{seed}.json'
            base=read(base_path)
            for method in ('baseline',*design.TRANSFORMS):
                path=ROOT/f'artifacts/priority34a/jobs/{dataset}/{method}/{seed}.json'
                relative=str(path.relative_to(ROOT));check(path,manifests['a'][relative]);doc=read(path)
                if doc['status']!='COMPLETED' or doc['config']['training']!={'alpha':.95,'lr':.001,'rounds':50}:
                    raise ValueError('P34A training contract mismatch')
                if not doc['no_bn_transmitted'] or doc['partition_sha256']!=base['partition_sha256']:
                    raise ValueError('P34A partition/BN mismatch')
                for name,digest in doc['config']['source_sha256'].items():check(ROOT/name,digest)
                for name,digest in doc['artifact_sha256'].items():check(ROOT/'artifacts/priority34a'/name,digest)
                registry.append(dict(dataset=dataset,seed=seed,method=method,path=relative,sha256=inputs[relative]))
            for clip in design.CLIPS:
                for epsilon in (1,3,10):
                    method=f'local_dp_{clip}_eps{epsilon}'
                    path=ROOT/f'artifacts/priority34b_local_dp/jobs/{dataset}/{method}/{seed}.json'
                    relative=str(path.relative_to(ROOT));check(path,manifests['b'][relative]);doc=read(path)
                    if doc['status']!='COMPLETED' or doc['accounting']['noise_sd']!=design.account(epsilon)['noise_sd']:
                        raise ValueError('P34B completion/noise contract mismatch')
                    if doc['partition_sha256']!=base['partition_sha256'] or not doc['no_bn_transmitted'] or doc['torch_threads']!=1:
                        raise ValueError('P34B Config-C identity mismatch')
                    for name,digest in doc['artifact_sha256'].items():check(ROOT/'artifacts/priority34b_local_dp'/name,digest)
                    registry.append(dict(dataset=dataset,seed=seed,method=method,path=relative,sha256=inputs[relative]))
    for relative,digest in read(ROOT/'artifacts/priority34b_local_dp/execution_freeze.json')['sources'].items():check(ROOT/relative,digest)
    for dataset in design.DATASETS:
        folder=ROOT/f'artifacts/priority32_multidataset/prepared/{dataset}';audit=read(folder/'audit.json')
        for name,digest in audit['prepared_sha256'].items():check(folder/name,digest)
    gate8=read(ROOT/'artifacts/priority34c/gates_n8.json');gate24=read(ROOT/'artifacts/priority34c/gates_n24.json')
    qualified=tuple((d,c) for d in design.DATASETS for c in design.CELLS
                    if gate8[d+'/'+c]['passed'] and gate24.get(d+'/'+c,{}).get('passed',False))
    if qualified!=design.QUALIFIED:raise ValueError('P34C qualification identity mismatch')
    for name in ('qualification_freeze.json','confirmatory_freeze.json'):
        for section in ('sources','inputs'):
            for relative,digest in read(ROOT/'artifacts/priority34c'/name)[section].items():check(ROOT/relative,digest)
    for path in [ROOT/'artifacts/priority34b_local_dp/COMPLETE.json',ROOT/'artifacts/priority34c/COMPLETE.json',
                 ROOT/'artifacts/priority34c/gates_n8.json',ROOT/'artifacts/priority34c/gates_n24.json',
                 ROOT/'reports/priority34c_report.md',ROOT/'reports/priority34b_local_dp_extension_report.md',
                 ROOT/'reports/priority35a_report.md',ROOT/'artifacts/priority35a/sanity_t/COMPLETE.json']:
        inputs[str(path.relative_to(ROOT))]=sha(path)
    if len(registry)!=297:raise ValueError('reuse registry must include 297 tabular utility jobs')
    return dict(reuse_jobs=registry,hashes=inputs,qualified_cells=qualified,
                training_or_recovery_replayed=False,metric_values_aggregated=False)


def prepare():
    freeze=read(OUT/'protocol_freeze.json')
    if sha(PROTOCOL)!=freeze['sha256']:raise ValueError('protocol changed since before tests')
    if (OUT/'readiness_freeze.json').exists():raise ValueError('already frozen; preserve readiness')
    folder=OUT/'preflight'/str(time.time_ns());folder.mkdir(parents=True)
    stage('TESTING_AND_IDENTITY_AUDIT')
    files=sorted(ROOT.glob('experiments/priority35b_*.py'))+sorted(ROOT.glob('tests/test_priority35b*.py'))
    commands=[[sys.executable,'-B','-m','unittest','discover','-s','tests','-p','test_priority35b*.py','-v'],
              [sys.executable,'-m','py_compile',*[str(f.relative_to(ROOT)) for f in files]],['git','diff','--check']]
    env=dict(os.environ,PYTHONPYCACHEPREFIX=str(OUT/'bytecode'))
    for i,command in enumerate(commands):
        result=subprocess.run(command,cwd=ROOT,env=env,text=True,capture_output=True)
        write(folder/f'command_{i}.json',dict(command=command,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr))
        if result.returncode:raise ValueError('preflight command failed; logs preserved')
    history=firewall.audit();write(OUT/'historical_firewall_audit.json',history)
    prior=prior_inputs();write(OUT/'historical_reuse_audit.json',prior)
    write(OUT/'design.json',dict(seeds=design.SEEDS,epsilons=design.EPSILONS,clips=design.CLIPS,
        qualified_cells=design.QUALIFIED,family=design.family(),family_size=144,figure=design.FIGURE,
        accounting=[design.account(e) for e in design.EPSILONS],measurements_authorized=False))
    checklist=read(OUT/'checklist.json')
    for item in checklist['items']:
        if item!='measurements':checklist['items'][item]='DONE'
    write(OUT/'checklist.json',checklist)
    stage('READY_PROTOCOL_TESTS_ONLY',tests_passed=15,holms_reserved=144,
          eligible_cells=8,reusable_utility_jobs=297,measurement_authorized=False)
    hashes={str(f.relative_to(ROOT)):sha(f) for f in files+[PROTOCOL]}
    for path in OUT.rglob('*'):
        if path.is_file() and 'bytecode' not in path.parts:hashes[str(path.relative_to(ROOT))]=sha(path)
    write(OUT/'readiness_freeze.json',dict(at=now(),status='PROTOCOL_TESTS_READY',
        amendment_sha256=sha(PROTOCOL),hashes=hashes,reference_hashes=prior['hashes'],
        real_targets_allocated=0,measurement_runs=0,measurements_authorized=False))
    print('P35B protocol/tests ready; no measurements; wait for user.',flush=True)


if __name__=='__main__':
    if len(sys.argv)!=1:raise SystemExit('No measurement/launch arguments are implemented.')
    try:prepare()
    except Exception as exc:
        write(OUT/f'preflight_failure_{time.time_ns()}.json',dict(at=now(),error=str(exc),measurements=0))
        stage('PREFLIGHT_FAILED',error=str(exc));raise
