"""Full post-exit P34B checkpoint/manifest/contract audit (no training)."""
from __future__ import annotations
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments import priority34b_core as b


def live_workers():
    rows=[]
    text=subprocess.check_output(['ps','-axo','pid=,ppid=,command='],text=True)
    for line in text.splitlines():
        parts=line.strip().split(None,2)
        if len(parts)!=3 or int(parts[0])==os.getpid():
            continue
        executable=Path(parts[2].split()[0]).name.lower()
        if 'python' in executable and ('spawn_main' in parts[2] or
               ('supervise_priority34b.py' in parts[2] and '--supervise' in parts[2])):
            rows.append(line)
    return rows


def main():
    if not (b.OUT/'READY_FOR_FINAL_AUDIT.json').exists() or live_workers():
        raise ValueError('not ready or live workloads remain')
    frozen=b.verified_freeze()
    manifest=b.read(b.OUT/'sha256_manifest.json')
    for relative,digest in manifest.items():
        if b.p.sha(ROOT/relative)!=digest:
            raise ValueError('manifest mismatch '+relative)
    if list(b.OUT.rglob('*.failure.json')) or (b.OUT/'REQUIRES_DIRECTION.json').exists():
        raise ValueError('failures unresolved')
    audit=b.read(b.OUT/'analysis/independent_recomputation.json')
    if audit['status']!='PASS' or audit['jobs']!=470 or audit['sign_tests']!=24:
        raise ValueError('independent audit incomplete')
    from experiments import priority34b_images as image
    count,arrays,assertions=0,0,0
    cifar_data=image.p31.dp.CIFAR10(str(ROOT/'datasets/cifar10'),train=True,download=False)
    split=b.read(image.p31.OUT/'split.json')
    vx,vy=image.tensors(cifar_data,split['validation_ids'])
    for job in frozen['jobs']:
        path=b.OUT/job['result']
        if not b.valid(path,job):
            raise ValueError('missing job')
        doc=b.read(path)
        count+=1
        if job['stage']=='tabular':
            prepared=b.p.OUT/'prepared'/job['dataset']
            width=b.np.load(prepared/'train_x.npy',mmap_mode='r').shape[1]
            checkpoint_name=next(n for n in doc['artifact_sha256'] if n.endswith('final_checkpoint.pt'))
            checkpoint=b.torch.load(b.OUT/checkpoint_name,map_location='cpu')
            for client in range(3):
                model=b.p.FraudMLP(width).cpu()
                bn,allowed=b.a.domains(model)
                b.a.assert_payload(checkpoint['global_non_bn'],bn,allowed)
                if set(checkpoint['client_bn'][client])!=bn:
                    raise ValueError('local BN checkpoint domain')
                state=model.state_dict(); state.update(checkpoint['global_non_bn']); state.update(checkpoint['client_bn'][client])
                model.load_state_dict(state); b.a.guard(model,'completion replay')
                for name in ['validation','test']:
                    x=b.np.load(prepared/(name+'_x.npy'),mmap_mode='r')
                    found=next(n for n in doc['artifact_sha256'] if n.endswith(f'client{client}_{name}_probabilities.npy'))
                    expected=b.np.load(b.OUT/found)
                    actual=b.a.checked_probabilities(model,x)
                    if not b.np.array_equal(expected,actual):
                        raise ValueError('checkpoint probabilities mismatch')
                    arrays+=1
            journal_name=next(n for n in doc['artifact_sha256'] if n.endswith('rounds.jsonl'))
            journals=[json.loads(line) for line in (b.OUT/journal_name).read_text().splitlines()]
            if [row['round'] for row in journals]!=list(range(1,51)):
                raise ValueError('training rounds missing')
            for row in journals:
                if set(row['transmitted_keys'])&set(row['bn_keys']) or not row['no_bn_transmitted']:
                    raise ValueError('BN transmission in journal')
                for client in row['bn_minima_all_steps']:
                    if any(not math.isfinite(value) or value<0 for value in client.values()):
                        raise ValueError('negative/nonfinite BN during training')
                assertions+=row['upload_assertions']
        elif job['stage']=='image_utility':
            found=next(n for n in doc['artifact_sha256'] if n.endswith('final_checkpoint.pt'))
            model=image.image_model(); model.load_state_dict(b.torch.load(b.OUT/found,map_location='cpu')); model.eval()
            from experiments.priority33c_checkpoint_replay import finite
            logits=[]
            with b.torch.no_grad():
                for start in range(0,len(vx),512):
                    value=model(vx[start:start+512]); finite(value,'completion image logits'); logits.append(value.numpy())
            found=next(n for n in doc['artifact_sha256'] if n.endswith('validation_logits.npy'))
            if not b.np.array_equal(b.np.concatenate(logits),b.np.load(b.OUT/found)):
                raise ValueError('image checkpoint predictions mismatch')
            arrays+=1; assertions+=doc['payload_assertions']
        if count%21==0:
            print(json.dumps(dict(event='completion_replay',jobs=count,arrays_bit_exact=arrays)),flush=True)
    if count!=470 or arrays!=1265 or assertions!=41250:
        raise ValueError('final replay/domain counts failed')
    check=subprocess.run(['git','diff','--check'],cwd=ROOT,capture_output=True,text=True)
    if check.returncode:
        raise ValueError(check.stderr+check.stdout)
    report=ROOT/'reports/priority34b_report.md'
    archive=ROOT/'reports/archive/priority34b_report_pre_exit_audit.md'
    if archive.exists() or (b.OUT/'COMPLETE.json').exists():
        raise ValueError('completion already sealed; never overwrite')
    archive.parent.mkdir(exist_ok=True); shutil.copy2(report,archive)
    if b.p.sha(archive)!=manifest[str(report.relative_to(ROOT))]:
        raise ValueError('historical report archive mismatch')
    text=report.read_text()
    pending='Status: scientific analysis complete; final supervisor-exit audit pending.'
    if text.count(pending)!=1:
        raise ValueError('report lifecycle mismatch')
    text=text.replace(pending,'Status: COMPLETE — full post-exit audit PASS.',1)
    text+='\n## Final verified completion\n\n470 new jobs,33 reused baselines; 1,265 new client/split or validation arrays bit-exact from checkpoints;41,250 no-BN upload assertions. Independent metrics, RDP calibration, CIs and fixed24test Holm PASS. Sources/input/result hashes, tests, py_compile/git diff --check PASS; no live worker. Pre-exit report archived byte-identically; scientific-manifest report digest resolves to archive/priority34b_report_pre_exit_audit.md. No scientific outcome replaced.\n'
    report.write_text(text)
    receipt=dict(status='PASS',at=b.p.now(),jobs=count,baseline_reused=33,
                 bit_exact_prediction_arrays=arrays,no_bn_assertions=assertions,live_workers=[],
                 manifest_files=len(manifest),independent=audit,git_diff_check='PASS',
                 preserved_report_sha256=b.p.sha(archive),final_report_sha256=b.p.sha(report))
    b.save(b.OUT/'final_verified_receipt.json',receipt)
    b.checklist(5,'complete'); b.progress('complete',470,470); b.event('final_verified_complete',jobs=470)
    b.save(b.OUT/'COMPLETE.json',receipt)
    final=dict(manifest)
    for path in b.OUT.rglob('*'):
        if path.is_file() and path.name!='sha256_manifest_final.json':
            final[str(path.relative_to(ROOT))]=b.p.sha(path)
    final[str(report.relative_to(ROOT))]=b.p.sha(report)
    final[str(archive.relative_to(ROOT))]=b.p.sha(archive)
    b.save(b.OUT/'sha256_manifest_final.json',final)
    print(json.dumps(receipt,indent=2),flush=True)


if __name__=='__main__':
    main()
