"""Independent native forward/metric/init/CSV audit and append-only T report."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import csv
import math
import time
from collections import OrderedDict
import numpy as np
import torch
from experiments import priority34a_local_bn as old
from experiments.priority35a_runtime import ROOT,OUT,Q,load,write,sha,check_hashes,now
from experiments.priority35a_sanity_t import P,PREVIOUS,AMENDMENT,REPORT,inputs,receipt,no_live,frozen,progress,checks
from experiments.priority35a_d1prime import receipt as previous_receipt
from experiments.priority35a_d1prime_verify import native_model,native_masks_gradient,IndependentFixedMask,gradient_objective,soft_and_hard
from experiments.priority35a_diagnostic import selected_cells
from experiments.priority35a_qualification import cell_id,valid_receipt,stage_targets
from experiments.priority35a_checkpoints import arrays_partitions,PREPARED
from experiments.priority35a_adam import require_finite_cpu
from experiments.priority35a_estimator import schema_blocks,public_seed
from experiments.priority35a_verify import independent_quality,compare_metrics
from experiments.priority35a_report import csv_write

SCALES=(.01,.1,.5)


def independent_gradient(public,private,masks,rows):
    model=native_model(public,private)
    for prefix,mask in masks.items():
        parent,index=prefix.rsplit('.',1)
        setattr(model.get_submodule(parent),index,IndependentFixedMask(mask))
    for name,param in model.named_parameters():
        if name in private['client_bn']:
            param.requires_grad_(False)
    loss=old.p.BinaryFocalLoss(alpha=.95,gamma=2.)(model(rows),private['labels'])
    params=OrderedDict((k,p) for k,p in model.named_parameters() if k in public['shared'])
    gradient=torch.cat([g.reshape(-1) for g in torch.autograd.grad(loss,tuple(params.values()))]).detach()
    require_finite_cpu(dict(gradient=gradient,loss=loss,**model.state_dict()),'T independent gradient')
    return gradient


def verify():
    torch.set_num_threads(1)
    frozen()
    if load(P/'MEASUREMENTS_FINISHED.json')['jobs']!=144:
        raise ValueError('incomplete T')
    history=load(Q/'history.json')['exclusions']
    prepared={}
    for dataset in ('baf','ieee_cis','paysim'):
        x,y,parts=arrays_partitions(dataset)
        ids=np.load(PREPARED/dataset/'train_source_ids.npy')
        prepared[dataset]=(x,y,parts,ids,np.argsort(ids))
    seen,observations={},{}
    truth_rows=load(P/'truth_checks.json')['rows']
    if [(r['cell'],r['index']) for r in truth_rows]!=[(c,i) for c in selected_cells() for i in range(8)]:
        raise ValueError('48 truth checks coverage')
    truth_csv=[]
    t2_csv=[]
    for saved in truth_rows:
        cell,index=saved['cell'],saved['index']
        identity,private,payload,stats,_=inputs(cell,index)
        r=valid_receipt(Q/'uploads'/identity/'receipt.json')
        public=torch.load(ROOT/r['public_path'],map_location='cpu',weights_only=False)
        target=stage_targets(cell,'n8')[index]
        if saved['identity']!=identity or saved['source_ids']!=target['source_ids'] or private['target']!=target:
            raise ValueError('original targets/source binding')
        if identity not in observations:
            x,y,parts,ids,order=prepared[cell['dataset']]
            pos=order[np.searchsorted(ids[order],target['source_ids'])]
            if ids[pos].tolist()!=target['source_ids'] or not set(pos).issubset(set(parts[target['client']])):
                raise ValueError('source partition firewall')
            already=seen.setdefault(cell['dataset'],set())
            if set(target['source_ids'])&(set(history[cell['dataset']])|already):
                raise ValueError('source-ID firewall')
            already.update(target['source_ids'])
            if not np.array_equal(x[pos],private['features'].numpy()) or not np.array_equal(y[pos],private['labels'].numpy().reshape(-1)):
                raise ValueError('prepared truth binding')
            masks,grads=native_masks_gradient(public,private)
            if any(not torch.equal(v,payload['masks'][k]) for k,v in masks.items()) or any(not torch.equal(v,payload['gradient'][k]) for k,v in grads.items()):
                raise ValueError('native mask/gradient capture binding')
            if not torch.equal(payload['labels'],private['labels']) or any(not torch.equal(v,payload['bn'][k]) for k,v in private['client_bn'].items()):
                raise ValueError('privileged labels/BN binding')
            observations[identity]=(masks,grads)
        masks,grads=observations[identity]
        expected=torch.cat([g.reshape(-1) for g in grads.values()])
        gradient=independent_gradient(public,private,masks,private['features'])
        denom=float(expected.double().norm())
        rel=float((gradient.double()-expected.double()).norm())/denom if denom else None
        forward=dict(forward_relative_difference=rel,forward_bit_exact=torch.equal(gradient,expected),
                     forward_max_abs=float((gradient-expected).abs().max()),zero_observed_norm=denom==0)
        for k,v in forward.items():
            if saved[k]!=v:
                raise ValueError('forward reproduction audit '+k)
        compare_metrics(saved['truth'],independent_quality(private['features'],private['features'],stats))
        prior=previous_receipt(PREVIOUS/'jobs/a'/cell_id(cell)/f'{index:02d}.json')
        output=torch.load(ROOT/prior['reconstruction_path'],map_location='cpu',weights_only=False)
        compare_metrics(saved['original'],independent_quality(private['features'],output['rows'],stats))
        values=[gradient_objective(public,private,masks,v,cell['instrument'],grads) for v in
                (private['features'],output['pooled_soft'],output['rows'])]
        if values!=[saved[k] for k in ('truth_objective','original_soft_objective','original_hard_objective')]:
            raise ValueError('independent float32 T1 objective differs')
        truth,soft,hard=values
        rr=None if abs(truth)<=1e-6 else soft/truth
        if saved['output_truth_ratio']!=rr or saved['truth_strictly_lower']!=(truth<soft):
            raise ValueError('T1 ratio/ordering')
        truth_csv.append(dict(cell=cell_id(cell),index=index,source_ids=' '.join(map(str,target['source_ids'])),
             T0_accuracy=saved['truth']['feature_accuracy'],truth_objective=truth,original_soft_objective=soft,
             original_hard_objective=hard,output_truth_ratio=rr,ratio_na_reason=saved['ratio_na_reason'],
             truth_strictly_lower=truth<soft,original_accuracy=saved['original']['feature_accuracy'],
             original_mse=saved['original']['standardized_mse'],**forward))
        blocks=schema_blocks(public['meta'])
        for scale in SCALES:
            result=receipt(P/'jobs'/cell_id(cell)/str(scale)/f'{index:02d}.json')
            recon=torch.load(ROOT/result['reconstruction_path'],map_location='cpu',weights_only=False)
            if result['job']['cell']!=cell or result['job']['index']!=index or result['job']['scale']!=scale:
                raise ValueError('T2 identity')
            if recon['synthetic'] or recon['starts']!=2 or recon['iterations']!=4000 or recon['torch_threads']!=1 or recon['scale']!=scale:
                raise ValueError('T2 fixed budget/thread')
            if not torch.equal(recon['labels'],private['labels']) or len(recon['members'])!=2 or len(recon['initializations'])!=2:
                raise ValueError('T2 labels/starts')
            member_values=[]
            for j,(member,init) in enumerate(zip(recon['members'],recon['initializations'])):
                seed=public_seed('sanity-T',identity,cell['instrument'],scale,j)
                std=torch.tensor(stats['std'],dtype=torch.float32)
                noise=torch.randn(private['features'].shape,generator=torch.Generator().manual_seed(seed))*std*scale
                perturbed=private['features']+noise
                columns=[]
                for block in blocks:
                    v=perturbed[:,block['indices']]
                    if block['categories'] is not None:
                        v=torch.maximum(v,torch.full_like(v,1e-6))
                        v=torch.log(v/v.sum(dim=1,keepdim=True))
                    columns.append(v)
                raw=torch.cat(columns,1)
                initial_soft,_=soft_and_hard(raw,blocks)
                if init['seed']!=seed or any(not torch.equal(init[k],v) for k,v in
                        dict(noise=noise,perturbed=perturbed,raw=raw,soft=initial_soft).items()):
                    raise ValueError('T2 independent initialization')
                require_finite_cpu(dict(raw=member['raw'],**member['oracle_affine']), 'T member')
                if any(not torch.equal(v,private['client_bn'][k]) for k,v in member['oracle_affine'].items()):
                    raise ValueError('fixed BN affine drift')
                if [t['iteration'] for t in member['trace']]!=[1,500,1000,1500,2000,2500,3000,3500,4000] or not all(math.isfinite(t['before_step_objective']) for t in member['trace']):
                    raise ValueError('T2 iteration/finite trace')
                member_soft,_=soft_and_hard(member['raw'],blocks)
                obj=gradient_objective(public,private,masks,member_soft,cell['instrument'],grads)
                if obj!=member['final_objective']:
                    raise ValueError('T2 independent member objective')
                member_values.append(obj)
            chosen=min(range(2),key=lambda j:(member_values[j],j))
            final_soft,final_hard=soft_and_hard(recon['members'][chosen]['raw'],blocks)
            if recon['chosen_member']!=chosen or not torch.equal(recon['rows'],final_hard) or not torch.equal(recon['pooled_soft'],final_soft):
                raise ValueError('T2 minimum observable selection/decode')
            final=gradient_objective(public,private,masks,final_soft,cell['instrument'],grads)
            hardobj=gradient_objective(public,private,masks,final_hard,cell['instrument'],grads)
            if final!=recon['final_objective']:
                raise ValueError('T2 final float32 objective')
            final_metric=independent_quality(private['features'],final_hard,stats)
            init=recon['initializations'][chosen]
            _,initial_hard=soft_and_hard(init['raw'],blocks)
            initial_metric=independent_quality(private['features'],initial_hard,stats)
            initial_obj=gradient_objective(public,private,masks,init['soft'],cell['instrument'],grads)
            returning=final_metric['feature_accuracy']==1 and final_metric['standardized_mse']<=1e-4
            t2_csv.append(dict(cell=cell_id(cell),index=index,scale=scale,source_ids=' '.join(map(str,target['source_ids'])),
                 initial_accuracy=initial_metric['feature_accuracy'],initial_mse=initial_metric['standardized_mse'],initial_objective=initial_obj,
                 final_accuracy=final_metric['feature_accuracy'],final_mse=final_metric['standardized_mse'],final_objective=final,
                 final_hard_objective=hardobj,chosen_start=chosen,truth_objective=truth,returns_truth=returning,
                 far_equal_or_lower_strict=not returning and final<=truth,
                 far_within_numerical_resolution=not returning and final<=truth+1e-6,
                 output_truth_ratio=None if abs(truth)<=1e-6 else final/truth))
    if len(observations)!=24 or len(t2_csv)!=144:
        raise ValueError('T complete coverage')
    return truth_csv,t2_csv


def summary(truth,t2):
    rows=[]
    truth_summary=[]
    for cell in selected_cells():
        identifier=cell_id(cell)
        group=[r for r in truth if r['cell']==identifier]
        if len(group)!=8:
            raise ValueError('eight truth units')
        truth_summary.append(dict(cell=identifier,T0_min_accuracy=min(r['T0_accuracy'] for r in group),
            bit_exact_count=sum(r['forward_bit_exact'] for r in group),max_relative=max(r['forward_relative_difference'] or 0 for r in group),
            median_truth_objective=float(np.median([r['truth_objective'] for r in group])),
            median_original_objective=float(np.median([r['original_soft_objective'] for r in group])),
            truth_lower_count=sum(r['truth_strictly_lower'] for r in group)))
        for scale in SCALES:
            units=[r for r in t2 if r['cell']==identifier and r['scale']==scale]
            if len(units)!=8 or sorted(r['index'] for r in units)!=list(range(8)):
                raise ValueError('eight paired T2 units')
            doc=dict(cell=identifier,scale=scale,n=8)
            for key in ('initial_accuracy','initial_mse','initial_objective','final_accuracy','final_mse','final_objective','final_hard_objective'):
                doc['median_'+key]=float(np.median([r[key] for r in units]))
            for key in ('returns_truth','far_equal_or_lower_strict','far_within_numerical_resolution'):
                doc[key+'_count']=sum(r[key] for r in units)
            rows.append(doc)
    defect=any(r['T0_accuracy']<1 or (r['forward_relative_difference'] or 0)>1e-6 or (r['zero_observed_norm'] and not r['forward_bit_exact']) for r in truth)
    return dict(truth_cells=truth_summary,T2_cells=rows,evaluation_code_defect=defect,
                original_qualification_unchanged=True,Holm_family_unchanged=6048,descriptive_only=True)


def export_verify(truth,t2):
    for name,rows in (('T0_T1',truth),('T2',t2)):
        csv_write(P/(name+'.csv'),rows,list(rows[0]))
    def read(path,template):
        with path.open(newline='') as f:
            records=list(csv.DictReader(f))
        if len(records)!=len(template):
            raise ValueError('CSV coverage')
        out=[]
        for record,expected in zip(records,template):
            parsed={}
            for key,value in expected.items():
                text=record[key]
                parsed[key]=(None if value is None else text=='True' if isinstance(value,bool)
                             else int(text) if isinstance(value,int) else float(text) if isinstance(value,float) else text)
                if parsed[key]!=value or (value is None and text!=''):
                    raise ValueError('CSV exact roundtrip '+key)
            out.append(parsed)
        return out
    a,b=read(P/'T0_T1.csv',truth),read(P/'T2.csv',t2)
    doc=summary(a,b)
    if doc!=summary(truth,t2):
        raise ValueError('independent CSV summaries differ')
    write(P/'summary.json',doc,exclusive=True)
    return doc


def section(doc):
    lines=['','## Post-hoc descriptive sanity diagnostic T (2026-10-10)','',
        'Earlier text and decisions above remain unchanged. T is a privileged diagnostic, not a qualification rerun or privacy comparison. Same24 distinct batches/six cells, b4/r10/given position-bound labels, actual dropout masks and true local BN; CPU/thread1.',
        '',f'Addendum SHA256 `{sha(AMENDMENT)}`; pre-measurement freeze SHA256 `{sha(P/"freeze.json")}`.',
        '', 'T0/T1: objectives evaluate true encoded rows directly; original objective uses stored soft output, not its hard-decoded metric representation. Per-batch soft/hard objectives and output/truth ratios are in `artifacts/priority35a/sanity_t/T0_T1.csv`. Ratios with abs(truth objective)<=1e-6 are NA, not infinity.',
        '', '| Cell | Minimum T0 accuracy | Bit-exact gradients /8 | Max relative mismatch | Median truth objective | Median original soft objective | Truth strictly lower /8 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in doc['truth_cells']:
        lines.append('| '+row['cell']+' | '+ ' | '.join(f'{row[k]:.9g}' for k in
            ('T0_min_accuracy','bit_exact_count','max_relative','median_truth_objective','median_original_objective','truth_lower_count'))+' |')
    lines+=['','T2: two starts x4000 iterations at each SD noise level. Initialization mapping to finite categorical logits was fixed before runs. Eight batches per cell/scale; starts/scales are not independent replicates. Final feature accuracy/MSE use hard-decoded rows, final objective uses soft rows.',
        '', '| Cell | Noise SD | Median initial accuracy | Median final accuracy | Median final std MSE | Median final soft objective | Returns truth /8 | Far, strict <= truth /8 | Far, within1e-6 /8 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in doc['T2_cells']:
        lines.append('| '+row['cell']+' | '+' | '.join(f'{row[k]:.9g}' for k in
            ('scale','median_initial_accuracy','median_final_accuracy','median_final_mse','median_final_objective','returns_truth_count','far_equal_or_lower_strict_count','far_within_numerical_resolution_count'))+' |')
    lines+=['','### Interpretation under the prospectively locked T rules','']
    if doc['evaluation_code_defect']:
        lines.append('T0/forward-model defect detected: P35A estimates are not usable as privacy evidence. This appended warning does not rewrite historical results.')
    else:
        lines.append('No T0/forward-model defect was detected in these tested batches. This checks the metric and privileged forward model only, not every component of the original estimators.')
        for row in doc['truth_cells']:
            group=[r for r in doc['T2_cells'] if r['cell']==row['cell']]
            returns=sum(r['returns_truth_count'] for r in group)
            far=sum(r['far_within_numerical_resolution_count'] for r in group)
            text=f"{returns}/24 scale-specific outputs return to truth; {far}/24 far outputs are numerically no higher than truth."
            if row['bit_exact_count']==8 and abs(row['median_truth_objective'])<=1e-6 and row['truth_lower_count'] and returns:
                text+=' Near-truth optimization success with higher-objective original outputs is consistent with an optimization/budget/landscape limitation.'
            if far:
                text+=' Far low-objective outputs indicate numerical non-identification by this objective; this is not a proof of exact model non-identifiability.'
            if not returns and not far:
                text+=' Cause remains unresolved: privileged near-truth optimization did not return to truth; landscape, budget and categorical representation remain possible limitations.'
            lines.append('- '+row['cell']+': '+text)
    lines+=['','No new sign tests or Holm adjustment; original0/72 and Holm6048 remain unchanged. Truth-informed starts are not realistic server capabilities, softmax cannot encode exact one-hot truth with finite logits, and these selected n8 batches do not establish privacy or general estimator correctness.',
        '', 'Independent audit reloads every target/output, native gradient/masks, firewall, initialization, budgets, metric, float32 objective and CSV summary. All144 T2 jobs and48 T0/T1 checks included. Separate SHA256 manifest/COMPLETE in `artifacts/priority35a/sanity_t/`. Commands: `.venv-phase1/bin/python -B experiments/priority35a_sanity_t.py --prepare`, `--launch`, `--finish`. No earlier measurement was rerun.', '']
    return '\n'.join(lines)


def finish():
    no_live()
    if (P/'COMPLETE.json').exists():
        raise ValueError('already COMPLETE')
    doc=frozen()
    progress('INDEPENDENT_VERIFYING')
    truth,t2=verify()
    summary_doc=export_verify(truth,t2)
    write(P/'independent_verification.json',dict(at=now(),status='PASS',truth_checks=48,T2_jobs=144,
          distinct_batches=24,source_firewall=True,native_gradient=True,float32_objectives_exact=True,
          metrics_independent=True,CSV_summary_recomputed=True),exclusive=True)
    commands=P/'verification_commands'/str(time.time_ns())
    checks(commands)
    no_live()
    frozen()
    base=(P/'base_report.md').read_bytes()
    addition=section(summary_doc).encode()
    if REPORT.read_bytes()!=base:
        raise ValueError('report prefix changed')
    with REPORT.open('ab') as f:
        f.write(addition)
    if REPORT.read_bytes()!=base+addition:
        raise ValueError('append-only audit')
    check_hashes({k:v for k,v in doc['hashes'].items() if k!=str(REPORT.relative_to(ROOT))})
    paths=[p for p in P.rglob('*') if p.is_file()]+[AMENDMENT,REPORT]+list((ROOT/'experiments').glob('priority35a_sanity_t*.py'))+[ROOT/'tests/test_priority35a_sanity_t.py']
    hashes={str(p.relative_to(ROOT)):sha(p) for p in paths}
    write(P/'manifest.json',dict(at=now(),hashes=hashes,base_report_sha256=doc['base_report_sha256'],
          base_report_bytes=doc['base_report_bytes'],append_only=True),exclusive=True)
    check_hashes(hashes)
    progress('COMPLETE',sanity_t_complete=True,sanity_t_jobs_done=144,sanity_t_pending=[],sanity_t_no_live_workers=True)
    write(P/'COMPLETE.json',dict(at=now(),status='VERIFIED_COMPLETE',manifest_sha256=sha(P/'manifest.json'),
          report_sha256=sha(REPORT),original_qualification=0,diagnostic='T',descriptive_only=True,
          independent_verification=True,evaluation_code_defect=summary_doc['evaluation_code_defect']),exclusive=True)
    paths=[OUT/'progress.json',OUT/'progress.log',OUT/'checklist.json',P/'COMPLETE.json']
    write(P/'administrative_terminal_seal.json',dict(at=now(),hashes={str(p.relative_to(ROOT)):sha(p) for p in paths}),exclusive=True)
    print(dict(status='T_VERIFIED_COMPLETE',report_sha256=sha(REPORT)),flush=True)
