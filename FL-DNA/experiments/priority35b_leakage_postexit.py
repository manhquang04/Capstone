"""Read-only post-exit integration of B3 evidence/gradient/metric audits.

This does not analyze or publish tables/figures and cannot mark COMPLETE.
It refuses to read measurement scores before actual supervisor/worker exit.
"""
import json
import os
import subprocess
from pathlib import Path
from experiments import priority35b_leakage_integrity as integrity


def process_snapshot():
    if os.name=='nt':
        raw=subprocess.check_output(['powershell','-NoProfile','-Command',
            'Get-CimInstance Win32_Process | Select-Object ProcessId,ParentProcessId,CommandLine | ConvertTo-Json'],text=True)
        doc=json.loads(raw)
        return [(int(r['ProcessId']),int(r['ParentProcessId']),r.get('CommandLine') or '') for r in (doc if isinstance(doc,list) else [doc])]
    rows=subprocess.check_output(['ps','-axo','pid=,ppid=,command='],text=True).splitlines()
    return [(int(p),int(parent),command) for p,parent,command in (row.strip().split(None,2) for row in rows if row.strip())]


def require_exit(snapshot, recorded, self_pid=None):
    self_pid=os.getpid() if self_pid is None else self_pid
    recorded=set(recorded)
    for pid,parent,command in snapshot:
        if pid==self_pid:continue
        lower=command.lower()
        if pid in recorded:
            raise ValueError('recorded PID still alive; fail closed even if PID reused: '+str(pid))
        if parent in recorded and 'python' in lower:
            raise ValueError('scientific child process still alive: '+str(pid))
        if 'python' in lower and 'priority35b_leakage' in lower and any(f in lower for f in ('--supervise','--job')):
            raise ValueError('B3 scientific process still alive: '+str(pid))
    return sorted(recorded)


def recorded_pids(out):
    recorded={integrity.read(out/'launch.json')['pid'],integrity.read(out/'supervisor.json')['pid']}
    recorded.update(integrity.read(p)['pid'] for p in (out/'jobs').glob('**/process.json'))
    return recorded


def port_free(port):
    import socket
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',port))


def verify_control_receipt(driver,data,truth,bundle,result):
    """Recompute control sampling exactly; independently score all 30 guesses."""
    from experiments import priority35b_leakage_payload_audit as payload
    from experiments import priority35b_leakage_metrics_audit as metric
    from privacy.seed_manager import derive_seed
    c=result['config']
    seed=derive_seed(351500,'p35b-b3-control',c['dataset'],c['cell'],c['index'])
    if bundle is None or bundle['control_seed']!=seed:raise ValueError('control seed receipt')
    # Frozen mean/mode construction; an independent decoder is not claimed.
    mm=driver.core.native.mean_mode_baseline(data,truth)
    payload.tensor_equal(mm,bundle['mean_mode'])
    mm_score=metric.independently_score(data,truth,mm)
    stored=result['controls']['mean_mode_metric']
    if mm_score['feature_accuracy']!=stored['feature_accuracy'] or mm_score['assignment']!=stored['assignment']:
        raise ValueError('mean/mode metric mismatch')
    max_difference=abs(mm_score['standardized_mse']-stored['standardized_mse'])
    if max_difference>1e-12:raise ValueError('mean/mode MSE mismatch')
    rng=driver.np.random.default_rng(seed);scores=[];first=None
    for i in range(30):
        guess=driver.torch.empty_like(truth)
        for _,(_,ids) in data.train_feature_index_map.items():
            rows=rng.integers(0,len(data.Xtrain),len(truth))
            guess[:,ids]=data.Xtrain[rows][:,ids]
        if i==0:first=guess.clone()
        scores.append(metric.independently_score(data,truth,guess)['feature_accuracy']*100)
    payload.tensor_equal(first,bundle['empirical_single'])
    recomputed=dict(mean_mode=mm_score['feature_accuracy']*100,empirical_mean=float(driver.np.mean(scores)),
                    empirical_single=scores[0],empirical_sd=float(driver.np.std(scores,ddof=1)))
    # The official percent path may multiply before averaging. Compare at a
    # prespecified audit-only absolute 1e-12 percentage-point resolution.
    for field,value in recomputed.items():
        error=abs(value-bundle['scores'][field]);max_difference=max(max_difference,error)
        if error>1e-12:raise ValueError('independent control score mismatch: '+field)
        if bundle['scores'][field]!=result['controls']['scores'][field]:raise ValueError('control JSON serialization mismatch')
    if bundle['mean_mode_metric']!=stored:raise ValueError('control metric serialization mismatch')
    return max_difference


def verify_all(root):
    """Call once post-exit. No measurement, optimization, or output writes."""
    root=Path(root);out=root/'artifacts/priority35b/b3'
    exited=require_exit(process_snapshot(),recorded_pids(out))
    # No inference from EOF or stale socket alone; both PID and lock checked.
    port_free(43576)
    if list(out.glob('*.lock')):raise ValueError('B3 lock file present')
    evidence=integrity.verify_evidence(root,exited)
    from experiments import priority35b_leakage as driver
    from experiments import priority35b_leakage_payload_audit as payload
    from experiments import priority35b_leakage_metrics_audit as metric
    from privacy.seed_manager import derive_seed
    driver.configure()
    f=driver.read(driver.FREEZE);targets=driver.read(out/'targets.json')['targets']
    target_map={tuple(t[k] for k in ('dataset','cell','index')):t for t in targets}
    secrets=driver.read(out/'private_noise.json')
    expected_keys={driver.key(c) for c in f['jobs'] if next(a for a in driver.read(out/'arms.json')[c['dataset']] if a['method']==c['arm'])['noise_sd']>0}
    if set(secrets)!=expected_keys:raise ValueError('private noise inventory')
    for secret in secrets.values():
        if type(secret)is not int or not 0<=secret<2**63:raise ValueError('invalid private noise seed')
    # Cache only one dataset/cell and row scores, not all ensemble bundles.
    datasets={};scores={};max_mse_difference=0.;max_control_difference=0.;controls=0
    for c in f['jobs']:
        key=(c['dataset'],c['cell']);target=target_map[(*key,c['index'])]
        if key not in datasets:datasets[key]=driver.core.PreparedDataset(*key)
        data=datasets[key]
        result,truth,rec,control=payload.audit_job(driver,c,target,data,secrets)
        seed=derive_seed(351400,'p35b-b3-reconstruction',c['dataset'],c['cell'],c['index'])
        if result['reconstruction_seed']!=seed:raise ValueError('paired reconstruction seed')
        u=dict(c,arm='unprotected');prior=driver.read(driver.folder(u)/'result.json')['controls']['mean_mode_metric']['standardized_mse']
        independent=metric.independently_score(data,truth,rec)
        max_mse_difference=max(max_mse_difference,metric.verify_score(result['score'],independent,prior))
        if c['arm']=='unprotected':
            max_control_difference=max(max_control_difference,verify_control_receipt(driver,data,truth,control,result));controls+=1
        elif control is not None:raise ValueError('unexpected protected control bundle')
        scores[driver.key(c)]=result['score']
    if controls!=312:raise ValueError('312 control sets required')
    selected=[]
    for d in driver.design.DATASETS:
        values={cell:[scores[driver.key(dict(dataset=d,cell=cell,index=i,arm='unprotected'))]['feature_accuracy'] for i in range(39)]
                for dd,cell in driver.design.QUALIFIED if dd==d}
        selected.append(driver.design.select_estimator(d,values))
    if selected!=driver.read(out/'estimator_selection.json')['panels']:raise ValueError('unprotected estimator selection')
    require_exit(process_snapshot(),recorded_pids(out));port_free(43576)
    return dict(status='POSTEXIT_EVIDENCE_PAYLOAD_METRICS_PASS_ONLY',evidence=evidence,
                jobs=9984,controls=312,max_mse_difference=max_mse_difference,
                max_control_difference=max_control_difference,exited_pids=exited,
                remaining='objective recomputation and serialized statistics; figure/report QA; full manifest/COMPLETE')
