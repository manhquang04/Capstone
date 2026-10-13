"""Independent D1-prime tensor, native RNG/masks, CSV and append-prefix audit."""
import os
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import csv
import math
from collections import OrderedDict
import numpy as np
import torch
from scipy.stats import binomtest
from experiments import priority34a_local_bn as old
from experiments.priority35a_runtime import ROOT, OUT, Q, load, write, sha, check_hashes, now
from experiments.priority35a_d1prime import P, AMENDMENT, REPORT, receipt, no_live, frozen, progress
from experiments.priority35a_d1prime_estimator import BUDGETS, MASK_LAYOUT
from experiments.priority35a_diagnostic import selected_cells
from experiments.priority35a_qualification import valid_receipt, cell_id, stage_targets
from experiments.priority35a_checkpoints import PREPARED, arrays_partitions
from experiments.priority35a_adam import require_finite_cpu
from experiments.priority35a_estimator import public_seed, schema_blocks
from experiments.priority35a_verify import independent_quality, compare_metrics, independent_gate
from experiments.priority35a_report import checks


def native_model(public, private):
    model = old.p.FraudMLP(private['features'].shape[1]).cpu()
    state = model.state_dict()
    state.update(private['client_bn'])
    state.update(public['shared'])
    model.load_state_dict(state, strict=True)
    model.train()
    require_finite_cpu(model.state_dict(), 'independent native model')
    return model


def native_masks_gradient(public, private):
    """Pre-hook RNG clone and dropout-of-ones, not production saved-backward masks."""
    with torch.random.fork_rng(devices=[]):
        model = native_model(public, private)
        masks, handles = {}, []
        def pre(prefix):
            def capture(module, inputs):
                with torch.random.fork_rng(devices=[]):
                    masks[prefix] = torch.nn.functional.dropout(torch.ones_like(inputs[0]), p=module.p, training=True).detach()
            return capture
        for prefix in MASK_LAYOUT:
            handles.append(model.get_submodule(prefix).register_forward_pre_hook(pre(prefix)))
        torch.manual_seed(private['dropout_seed'])
        try:
            logits = model(private['features'])
            loss = old.p.BinaryFocalLoss(alpha=.95, gamma=2.)(logits, private['labels'])
            params = OrderedDict((k, p) for k, p in model.named_parameters() if k in public['shared'])
            grads = OrderedDict(zip(params, [g.detach() for g in torch.autograd.grad(loss, tuple(params.values()))]))
        finally:
            for h in handles:
                h.remove()
    require_finite_cpu(dict(logits=logits, **grads, **masks), 'independent native masks/gradient')
    return masks, grads


class IndependentFixedMask(torch.nn.Module):
    def __init__(self, mask):
        super().__init__()
        self.register_buffer('scale', mask.detach().clone())

    def forward(self, x):
        if x.shape != self.scale.shape:
            raise ValueError('independent fixed-mask shape')
        return torch.mul(x, self.scale)


def gradient_objective(public, private, masks, rows, instrument, observed):
    model = native_model(public, private)
    for prefix in masks:
        parent, index = prefix.rsplit('.', 1)
        setattr(model.get_submodule(parent), index, IndependentFixedMask(masks[prefix]))
    for key, parameter in model.named_parameters():
        if key in private['client_bn']:
            parameter.requires_grad_(False)
    logits = model(rows)
    loss = old.p.BinaryFocalLoss(alpha=.95, gamma=2.)(logits, private['labels'])
    params = tuple(p for k, p in model.named_parameters() if k in public['shared'])
    grads = torch.autograd.grad(loss, params)
    candidate = torch.cat([g.reshape(-1) for g in grads])
    expected = torch.cat([g.reshape(-1) for g in observed.values()])
    value = 1-torch.nn.functional.cosine_similarity(candidate, expected, dim=0, eps=1e-12)
    if instrument == 'tableak':
        value = value+.1*(candidate-expected).square().sum()/(expected.square().sum()+1e-12)
    require_finite_cpu(dict(candidate=candidate, loss=loss, value=value, **model.state_dict()), 'independent objective')
    return float(value.detach())


def soft_and_hard(raw, blocks):
    soft, hard = [], []
    for block in blocks:
        v = raw[:, block['indices']]
        soft.append(v if block['categories'] is None else torch.softmax(v, dim=1))
        hard.append(v if block['categories'] is None else torch.nn.functional.one_hot(v.argmax(1), len(block['indices'])).to(v.dtype))
    return torch.cat(soft, 1), torch.cat(hard, 1)


def verify():
    torch.set_num_threads(1)
    frozen()
    table = load(P/'summary.json')['summaries']
    if len(table) != 12 or [(d['cell'],d['condition']) for d in table] != [(c,k) for c in selected_cells() for k in ('a','b')]:
        raise ValueError('missing/duplicate diagnostic cells')
    history = load(Q/'history.json')
    observations, info, seen_ids = {}, {}, {}
    for dataset in ('baf','ieee_cis','paysim'):
        x,y,parts = arrays_partitions(dataset)
        ids = np.load(PREPARED/dataset/'train_source_ids.npy')
        info[dataset] = (x,y,parts,ids,np.argsort(ids))
        seen_ids[dataset] = set()
    for cell in selected_cells():
        dataset = cell['dataset']
        x,y,parts,ids,order = info[dataset]
        for i,target in enumerate(stage_targets(cell,'n8')):
            identity = f'{dataset}_r10_b4_n8_{i:02d}'
            if identity in observations:
                continue
            r = valid_receipt(Q/'uploads'/identity/'receipt.json')
            public = torch.load(ROOT/r['public_path'],map_location='cpu',weights_only=False)
            private = torch.load(ROOT/r['private_path'],map_location='cpu',weights_only=False)
            if private['target'] != target or set(target['source_ids']) & (set(history['exclusions'][dataset]) | seen_ids[dataset]):
                raise ValueError('original-ID firewall failure')
            positions = order[np.searchsorted(ids[order],target['source_ids'])]
            if ids[positions].tolist()!=target['source_ids'] or not set(positions).issubset(set(parts[target['client']])):
                raise ValueError('partition/source binding')
            if not np.array_equal(x[positions],private['features'].numpy()) or not np.array_equal(y[positions],private['labels'].numpy().reshape(-1)):
                raise ValueError('prepared truth binding')
            seen_ids[dataset].update(target['source_ids'])
            masks,grads = native_masks_gradient(public,private)
            captured = receipt(P/'observations'/identity/'receipt.json')
            pp = ROOT/captured['payload_path']
            if pp.stat().st_mode & 0o077:
                raise ValueError('privileged payload permission')
            payload = torch.load(pp,map_location='cpu',weights_only=False)
            if set(payload)!={'shared','gradient','bn','masks','labels','meta','ordinal'}:
                raise ValueError('knowledge whitelist')
            for key in masks:
                if not torch.equal(masks[key],payload['masks'][key]):
                    raise ValueError('independent dropout-of-ones mask mismatch')
            for key in grads:
                if not torch.equal(grads[key],payload['gradient'][key]) or not torch.equal(public['shared'][key],payload['shared'][key]):
                    raise ValueError('native gradient/shared mismatch')
            if not torch.equal(payload['labels'],private['labels']) or any(not torch.equal(v,payload['bn'][k]) for k,v in private['client_bn'].items()):
                raise ValueError('ordered labels/BN mismatch')
            if payload['ordinal']!=identity or payload['meta']!=public['meta']:
                raise ValueError('public ordinal/schema mismatch')
            observations[identity]=(public,private,masks,grads)
    manifest=load(P/'target_manifest.json')
    if manifest['distinct_batches']!=24 or [t['identity'] for t in manifest['targets']]!=list(observations):
        raise ValueError('target manifest coverage')
    tests=0
    for saved in table:
        cell,condition=saved['cell'],saved['condition']
        identifier=cell_id(cell)
        if load(P/'per_target'/condition/(identifier+'.json'))!=saved:
            raise ValueError('summary binding')
        stats=load(Q/'evaluator'/(cell['dataset']+'.json'))
        guesses,bundles=[],[]
        count,iterations=BUDGETS[condition]
        for i in range(8):
            identity=f"{cell['dataset']}_r10_b4_n8_{i:02d}"
            public,private,masks,grads=observations[identity]
            r=receipt(P/'jobs'/condition/identifier/f'{i:02d}.json')
            recon=torch.load(ROOT/r['reconstruction_path'],map_location='cpu',weights_only=False)
            if r['job']['cell']!=cell or r['job']['condition']!=condition or r['job']['index']!=i or r['job']['payload_path']!=receipt(P/'observations'/identity/'receipt.json')['payload_path']:
                raise ValueError('job identity')
            check_hashes({r['job']['payload_path']:r['job']['payload_sha256']})
            if recon['synthetic'] or recon['torch_threads']!=1 or recon['starts']!=count or recon['iterations']!=iterations or len(recon['members'])!=count:
                raise ValueError('fixed scientific budget/thread')
            if not recon['dropout_masks_supplied'] or not recon['ordered_labels_supplied'] or not torch.equal(recon['labels'],private['labels']):
                raise ValueError('known-label/mask condition')
            blocks=schema_blocks(public['meta'])
            soft_members=[]
            objectives=[]
            for j,m in enumerate(recon['members']):
                require_finite_cpu(dict(raw=m['raw'],**m['oracle_affine']), 'independent member')
                if m['raw'].shape != private['features'].shape or m['trace'][-1]['iteration']!=iterations or any(not math.isfinite(t['before_step_objective']) for t in m['trace']):
                    raise ValueError('member shape/budget/trace')
                if any(not torch.equal(v,private['client_bn'][k]) for k,v in m['oracle_affine'].items()):
                    raise ValueError('fixed oracle affine')
                soft,_=soft_and_hard(m['raw'],blocks)
                value=gradient_objective(public,private,masks,soft,cell['instrument'],grads)
                if value!=m['final_objective']:
                    raise ValueError('member objective exact reload')
                soft_members.append(soft);objectives.append(value)
            if condition=='a':
                chosen=min(range(count),key=lambda j:(objectives[j],j))
                pooled,rows=soft_and_hard(recon['members'][chosen]['raw'],blocks)
                if recon['chosen_member']!=chosen:
                    raise ValueError('observable selection')
            else:
                if recon['chosen_member'] is not None:
                    raise ValueError('ensemble not post-selected')
                # Separate sort implementation of torch lower median; physical row indices fixed.
                pooled=torch.stack(soft_members).sort(dim=0).values[(count-1)//2]
                columns=[]
                for block in blocks:
                    v=pooled[:,block['indices']]
                    columns.append(v if block['categories'] is None else torch.nn.functional.one_hot(v.argmax(1),len(block['indices'])).to(v.dtype))
                rows=torch.cat(columns,1)
            if not torch.equal(pooled,recon['pooled_soft']) or not torch.equal(rows,recon['rows']):
                raise ValueError('independent pooling/decode mismatch')
            if gradient_objective(public,private,masks,pooled,cell['instrument'],grads)!=recon['final_objective']:
                raise ValueError('pooled objective mismatch')
            require_finite_cpu(dict(rows=rows,pooled=pooled), 'independent result')
            guesses.append(rows.numpy());bundles.append(private)
        independent_rows=[]
        for i,bundle in enumerate(bundles):
            truth=bundle['features'].numpy()
            prior=independent_quality(truth,bundle['mean_mode'],stats)
            mse=prior['standardized_mse']
            row=dict(index=i,estimator=independent_quality(truth,guesses[i],stats,mse),mean_mode=prior,
                     empirical_marginal=independent_quality(truth,bundle['empirical_marginal'],stats,mse),
                     decoy=independent_quality(truth,guesses[(i+1)%8],stats,mse))
            if saved['rows'][i]['index']!=i or saved['rows'][i]['source_ids']!=bundle['target']['source_ids']:
                raise ValueError('per-target identity')
            for arm in ('estimator','mean_mode','empirical_marginal','decoy'):
                compare_metrics(saved['rows'][i][arm],row[arm])
            independent_rows.append(row)
        result=independent_gate(independent_rows,8)
        if result!=saved['comparisons'] or result['passed']!=saved['diagnostic_beats_all_controls']:
            raise ValueError('independent descriptive tests')
        for arm in ('estimator','mean_mode','empirical_marginal','decoy'):
            for metric in ('feature_accuracy','standardized_mse','bounded_score'):
                if not math.isclose(saved['medians'][arm][metric],float(np.median([r[arm][metric] for r in independent_rows])),rel_tol=1e-12,abs_tol=1e-14):
                    raise ValueError('independent median')
        tests+=3
    return dict(status='PASS',native_masks_independent=True,gradient_bit_exact=True,original_source_firewall=True,
                distinct_batches=24,reconstructions=96,descriptive_tests=tests,pooling_independent_sort=True)


def export_and_csv_verify():
    table=load(P/'summary.json')['summaries']
    with (P/'per_target.csv').open('x',newline='') as f:
        fields=['cell','condition','index','source_ids','arm','feature_accuracy','standardized_mse','bounded_score']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for d in table:
            for row in d['rows']:
                for arm in ('estimator','mean_mode','empirical_marginal','decoy'):
                    writer.writerow(dict(cell=cell_id(d['cell']),condition=d['condition'],index=row['index'],
                                         source_ids=';'.join(map(str,row['source_ids'])),arm=arm,
                                         **{k:row[arm][k] for k in fields[-3:]}))
    with (P/'comparisons.csv').open('x',newline='') as f:
        fields=['cell','condition','control','n','wins','losses','ties','p_one_sided','median_difference']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for d in table:
            for control,test in d['comparisons']['tests'].items():
                writer.writerow(dict(cell=cell_id(d['cell']),condition=d['condition'],control=control,**test))
    with (P/'per_target.csv').open() as f:
        rows=list(csv.DictReader(f))
    if len(rows)!=384:
        raise ValueError('complete CSV rows')
    bound={}
    for d in table:
        for row in d['rows']:
            for arm in ('estimator','mean_mode','empirical_marginal','decoy'):
                bound[(cell_id(d['cell']),d['condition'],row['index'],arm)]=row
    indexed={}
    for row in rows:
        key=(row['cell'],row['condition'],int(row['index']),row['arm'])
        if key not in bound or key in indexed or row['source_ids']!=';'.join(map(str,bound[key]['source_ids'])):
            raise ValueError('CSV complete identity binding')
        for metric in ('feature_accuracy','standardized_mse','bounded_score'):
            value=float(row[metric])
            if not math.isfinite(value) or value!=bound[key][row['arm']][metric]:
                raise ValueError('CSV numeric binding')
        indexed[key]=row
    with (P/'comparisons.csv').open() as f:
        comparisons=list(csv.DictReader(f))
    if len(comparisons)!=36 or len({(r['cell'],r['condition'],r['control']) for r in comparisons})!=36:
        raise ValueError('CSV comparisons coverage')
    for row in comparisons:
        differences=np.array([float(indexed[(row['cell'],row['condition'],i,'estimator')]['feature_accuracy'])-
                              float(indexed[(row['cell'],row['condition'],i,row['control'])]['feature_accuracy']) for i in range(8)])
        w,l=int((differences>0).sum()),int((differences<0).sum())
        p=float(binomtest(w,w+l,alternative='greater').pvalue) if w+l else 1.
        if (int(row['n']),int(row['wins']),int(row['losses']),int(row['ties']))!=(8,w,l,8-w-l) or float(row['p_one_sided'])!=p or float(row['median_difference'])!=float(np.median(differences)):
            raise ValueError('independent CSV sign recomputation')
    return dict(status='PASS',paired_n=8,complete_per_target_rows=384,exact_tests_recomputed=36)


def section(audit,csv_audit,commands):
    table=load(P/'summary.json')['summaries']
    lines=['','## Additional post-hoc descriptive diagnostic: D1-prime (2026-10-10)','',
           'This appended section does not revise any earlier result or interpretation. Original report byte prefix and original qualification/COMPLETE/manifests remain preserved. Qualification remains 0/72; Holm6048 is unchanged. This is a privileged descriptive diagnostic, not evidence for deployable Config-C privacy or new qualification.',
           '',f'Pre-execution addendum `{AMENDMENT.relative_to(ROOT)}`, SHA-256 `{sha(AMENDMENT)}`; freeze SHA-256 `{sha(P/"freeze.json")}`.',
           '', 'Six cells, b4/r10/labels given, same eight D1 targets. a: true local BN and actual dropout masks, 2x4000 steps. b: same plus 30x1500 members, coordinate lower-median pooling after categorical softmax, physical mask/label row positions retained. Same objectives, initialization seed rule, Adam .03 and annealing; only budget/ensemble changes. c: NOT_RUN, prospectively omitted. TabLeak reference budget [official code](https://github.com/eth-sri/tableak); this is a budget and position-aware ensemble adaptation, not a full reference reproduction.',
           '', 'Direct user approval additionally supplies labels attached to mask row positions. Original D1 knew an unordered multiset. Two IEEE-CIS batches have mixed labels; for them improvements involve both masks and label-position information. All other batches have homogeneous labels. No true feature rows, source IDs or private RNG enter the estimator. BN remains train mode and true affine is fixed in this privileged diagnostic.',
           '', 'Unit n=8 paired batches per cell, not ensemble members; 24 distinct reused batches. Data-free references unchanged; decoy is cyclic next-target reconstruction under the same condition/cell. Exact one-sided sign tests, exact ties excluded from binomial n, all three p<.05 for descriptive beats-all. Unadjusted post-hoc classifications lie outside unchanged Holm6048.',
           '', '| Cell | Condition | Median feature accuracy | Mean/mode W/L/T; median delta; p | Marginal W/L/T; median delta; p | Decoy W/L/T; median delta; p | Beats all (descriptive) |',
           '|---|---|---:|---|---|---|---|']
    for d in table:
        values=[]
        for c in ('mean_mode','empirical_marginal','decoy'):
            t=d['comparisons']['tests'][c]
            values.append(f"{t['wins']}/{t['losses']}/{t['ties']}; {t['median_difference']:.9g}; {t['p_one_sided']:.9g}")
        lines.append(f"| {cell_id(d['cell'])} | D1-prime-{d['condition']} | {d['medians']['estimator']['feature_accuracy']:.9g} | "+' | '.join(values)+f" | {d['diagnostic_beats_all_controls']} |")
    lines+=['','### Interpretation under the pre-run D1-prime rules','']
    for cell in selected_cells():
        group={d['condition']:d for d in table if d['cell']==cell}
        a,b=group['a']['diagnostic_beats_all_controls'],group['b']['diagnostic_beats_all_controls']
        text=('Estimators work in the privileged known-mask/ordered-label condition; unknown dropout is consistent with an obstacle in original D1 (mixed IEEE labels confound dropout-only attribution).'
              if a else 'Larger budget/ensemble passes where the original-sized known-mask budget does not; budget/ensemble limitation under this privileged condition.'
              if b else 'Both fail: estimators remain inadequate for this model under the fixed evaluation; original negative qualification says nothing about privacy.')
        lines.append('- '+cell_id(cell)+': '+text)
    lines+=['','Small selected n8 samples, unadjusted exploratory tests, finite budgets, fixed BN, ordered-label privilege and changed pooling limit causal claims. A-versus-b significance differences alone do not prove an interaction. No change to earlier conclusions and no evidence of transform superiority versus DP.',
            '',f'Independent tensor/native mask/gradient/firewall/budget/objective/pooling/metric audit: {audit["status"]}; CSV recomputation: {csv_audit["status"]}, all36 tests,384 per-target arm rows. Tests/py_compile/git diff --check logs: `{commands.relative_to(ROOT)}`. CPU/thread1; no measurement exclusions or replacement.',
            '', 'Commands: `.venv-phase1/bin/python -B experiments/priority35a_d1prime.py --prepare`, `--launch`, `--finish`. Results, CSVs and separate SHA-256 manifest/COMPLETE: `artifacts/priority35a/d1prime/`. Original report prefix snapshot: `artifacts/priority35a/d1prime/base_report.md`.', '']
    return '\n'.join(lines)


def finish():
    no_live()
    if (P/'COMPLETE.json').exists():
        raise ValueError('D1-prime already complete; no overwrite')
    doc=frozen()
    progress('INDEPENDENT_VERIFYING')
    audit=verify()
    write(P/'independent_tensor_verification.json',dict(at=now(),**audit),exclusive=True)
    csv_audit=export_and_csv_verify()
    write(P/'independent_csv_verification.json',dict(at=now(),**csv_audit),exclusive=True)
    commands=P/'verification_commands'/str(__import__('time').time_ns())
    checks(commands)
    no_live()
    base=(P/'base_report.md').read_bytes()
    if REPORT.read_bytes()!=base or sha(REPORT)!=doc['base_report_sha256']:
        raise ValueError('report prefix changed before append')
    addition=section(audit,csv_audit,commands).encode()
    with REPORT.open('ab') as f:
        f.write(addition)
    if REPORT.read_bytes()!=base+addition:
        raise ValueError('report append prefix audit')
    # Old manifest's report digest is a historical prefix assertion after authorized append.
    check_hashes({p:h for p,h in doc['hashes'].items() if p!=str(REPORT.relative_to(ROOT))})
    paths=[p for p in P.rglob('*') if p.is_file() and p.suffix in ('.json','.pt','.csv','.md','.log')]
    paths += [AMENDMENT,REPORT]+list((ROOT/'experiments').glob('priority35a_d1prime*.py'))+[ROOT/'tests/test_priority35a_d1prime.py']
    hashes={str(p.relative_to(ROOT)):sha(p) for p in paths}
    write(P/'manifest.json',dict(at=now(),hashes=hashes,base_report_sha256=doc['base_report_sha256'],
          base_report_bytes=doc['base_report_bytes'],report_append_only=True,original_P35A_COMPLETE_unchanged=True),exclusive=True)
    check_hashes(hashes)
    progress('COMPLETE',d1prime_complete=True,d1prime_jobs_done=96,d1prime_pending=[],d1prime_no_live_workers=True)
    checklist=load(OUT/'checklist.json')
    checklist.update(d1prime_complete=True,d1prime_independent_verification_pass=True,d1prime_report_append_prefix_verified=True)
    write(OUT/'checklist.json',checklist)
    write(P/'COMPLETE.json',dict(at=now(),status='VERIFIED_COMPLETE',manifest_sha256=sha(P/'manifest.json'),
          report_sha256=sha(REPORT),original_report_prefix_sha256=doc['base_report_sha256'],
          original_qualification=0,descriptive_only=True,independent_tensor_pass=True,independent_csv_pass=True),exclusive=True)
    terminal=[OUT/'progress.json',OUT/'progress.log',OUT/'checklist.json',P/'COMPLETE.json']
    write(P/'administrative_terminal_seal.json',dict(at=now(),hashes={str(p.relative_to(ROOT)):sha(p) for p in terminal}),exclusive=True)
    print(dict(status='D1PRIME_VERIFIED_COMPLETE',report_sha256=sha(REPORT)),flush=True)
