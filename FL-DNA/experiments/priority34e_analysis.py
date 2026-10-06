"""Independent P34E reload/paired statistics/report; never trains or searches32bit."""
import json
import math
import os
import shutil
import subprocess
import time
from pathlib import Path
import numpy as np
from scipy import stats
from sklearn.metrics import f1_score,roc_auc_score,average_precision_score
from experiments.priority34e import ROOT,OUT,read,write,verify_freeze,progress,checklist,SEEDS,KS
from experiments import priority34a_local_bn as a
from experiments.priority34e_utility import valid_result
ENDPOINTS=('f1','auc_roc','pr_auc')

def summary(values):
    values=np.asarray(values,dtype=float)
    if len(values)!=11 or not np.isfinite(values).all():raise ValueError('exact11 finite pairs required')
    mean=float(values.mean());sd=float(values.std(ddof=1));half=float(stats.t.ppf(.975,10))*sd/math.sqrt(11)
    check_mean=math.fsum(values)/11
    check_sd=math.sqrt(math.fsum((v-check_mean)**2 for v in values)/10)
    check_half=float(stats.t.isf(.025,10))*check_sd/math.sqrt(11)
    np.testing.assert_allclose([mean,sd,half],[check_mean,check_sd,check_half],atol=1e-12,rtol=0)
    return dict(n=11,mean_delta=mean,sd=sd,ci95=[mean-half,mean+half],median_delta=float(np.median(values)),
                iqr=[float(v) for v in np.quantile(values,[.25,.75])],descriptive=True)

def finish():
    manifest=verify_freeze();checklist('independent_audit','in_progress');progress('independent_audit',198)
    records={};max_error=0.;prediction_arrays=0;upload_assertions=0
    for job in manifest['jobs']:
        config=job['config'];k=config['K'];dataset=config['dataset'];seed=config['seed']
        path=OUT/job['result']
        if not valid_result(path,config):raise ValueError('missing job; no partial analysis')
        doc=read(path)
        if doc['device']!='cpu' or doc['torch_threads']!=doc['torch_interop_threads'] or doc['torch_threads']!=1:raise ValueError('thread/device')
        folder=a.p.OUT/'prepared'/dataset
        x=np.load(folder/'train_x.npy',mmap_mode='r');y=np.load(folder/'train_y.npy');cats=np.load(folder/'train_categories.npy')
        parts=a.p._mild_non_iid_client_indices(a.p.pd.DataFrame({'type':cats}),y,k,seed)
        import hashlib
        assert [hashlib.sha256(v.tobytes()).hexdigest() for v in parts]==doc['partition_sha256']
        assert list(map(len,parts))==doc['client_rows']
        assert [int(y[v].sum()) for v in parts]==doc['client_fraud_counts']
        outputs=doc['artifact_sha256']
        checkpoints=[name for name in outputs if name.endswith('final_checkpoint.pt')]
        journals=[name for name in outputs if name.endswith('rounds.jsonl')]
        if len(checkpoints)!=1 or len(journals)!=1:raise ValueError('checkpoint/journal provenance')
        checkpoint=a.torch.load(OUT/checkpoints[0],map_location='cpu')
        model=a.p.FraudMLP(x.shape[1]).cpu();bn,allowed=a.domains(model)
        a.assert_payload(checkpoint['global_non_bn'],bn,allowed)
        assert len(checkpoint['client_bn'])==k
        rounds=[json.loads(line) for line in (OUT/journals[0]).read_text().splitlines()]
        assert [row['round'] for row in rounds]==list(range(1,51))
        for row in rounds:
            assert row['transmitted_keys']==list(allowed) and row['bn_keys']==sorted(bn)
            assert row['no_bn_transmitted'] and row['upload_assertions']==k
            assert len(row['bn_minima_all_steps'])==k and len(row['bn_minima_per_client'])==k
            for minima in row['bn_minima_all_steps']+row['bn_minima_per_client']:
                assert all(math.isfinite(v) and v>=0 for v in minima.values())
        upload_assertions+=50*k
        recomputed=[]
        for client,state in zip(doc['clients'],checkpoint['client_bn']):
            assert set(state)==bn
            combined=model.state_dict();combined.update(checkpoint['global_non_bn']);combined.update(state)
            model.load_state_dict(combined);a.guard(model,'independent_final')
            values={}
            for split in ('validation','test'):
                features=np.load(folder/(split+'_x.npy'),mmap_mode='r');labels=np.load(folder/(split+'_y.npy'))
                matches=[name for name in outputs if name.endswith(f"client{client['client']}_{split}_probabilities.npy")]
                if len(matches)!=1:raise ValueError('probability provenance')
                probability=np.load(OUT/matches[0],allow_pickle=False)
                if len(probability)!=len(labels) or not np.isfinite(probability).all():raise ValueError('invalid probabilities')
                # Independently reload the final client checkpoint and reproduce every prediction.
                model.eval();blocks=[]
                with a.torch.no_grad():
                    for begin in range(0,len(features),4096):
                        logits=model(a.torch.from_numpy(np.array(features[begin:begin+4096],copy=True)))
                        if not a.torch.isfinite(logits).all():raise ValueError('nonfinite replay')
                        blocks.append(a.torch.sigmoid(logits).reshape(-1).numpy())
                if not np.array_equal(np.concatenate(blocks),probability):raise ValueError('checkpoint predictions mismatch')
                prediction_arrays+=1
                threshold=client['threshold']
                if split=='validation' and a.p.tune_threshold(labels,probability)!=threshold:raise ValueError('threshold mismatch')
                values[split]=dict(f1=float(f1_score(labels,probability>=threshold,zero_division=0)),auc_roc=float(roc_auc_score(labels,probability)),pr_auc=float(average_precision_score(labels,probability)))
                for endpoint in ENDPOINTS:
                    error=abs(values[split][endpoint]-client[split][endpoint]);max_error=max(max_error,error)
                    if error>1e-12:raise ValueError('client metric mismatch')
            recomputed.append(values)
        for split in ('validation','test'):
            for endpoint in ENDPOINTS:
                observed=math.fsum(c[split][endpoint] for c in recomputed)/k
                if abs(observed-doc[split][endpoint])>1e-12:raise ValueError('unweighted mean mismatch')
        records[(dataset,k,config['method'],seed)]=doc
    assert len(records)==198
    paired=[];summaries=[]
    for dataset in a.p.DATASETS:
        for k in KS:
            for method in a.p.METHODS[1:]:
                for endpoint in ENDPOINTS:
                    deltas=[]
                    for seed in SEEDS:
                        base=records[(dataset,k,'baseline',seed)];dna=records[(dataset,k,method,seed)]
                        assert base['partition_sha256']==dna['partition_sha256']
                        delta=dna['test'][endpoint]-base['test'][endpoint];deltas.append(delta)
                        paired.append(dict(dataset=dataset,K=k,method=method,endpoint=endpoint,seed=seed,baseline=base['test'][endpoint],dna=dna['test'][endpoint],delta=delta))
                    summaries.append(dict(dataset=dataset,K=k,method=method,endpoint=endpoint,**summary(deltas)))
    # Independent chunk coverage and recognized seed check; no32-bit enumeration.
    search=read(OUT/'seed_search/result.json');public=read(OUT/'seed_search/public_metadata.json')
    from dna_encoder.transform_defense_v2 import _derive_seed
    all_scores=[];active=[]
    for begin in range(0,2**20,16384):
        chunk=read(OUT/f'seed_search/chunk_{begin:07d}.json')
        scorefile=OUT/f'seed_search/scores_{begin:07d}.npy'
        assert chunk['begin']==begin and chunk['end']==begin+16384 and chunk['count']==16384
        assert a.p.sha(scorefile)==chunk['scores_sha256']
        scored=np.load(scorefile,allow_pickle=False);assert scored.shape==(16384,)
        # Independent NumPy RNG expression, including validation of every rejected candidate.
        check=np.zeros(16384,dtype=np.uint8)
        for index,candidate in enumerate(range(begin,begin+16384)):
            seed=_derive_seed(_derive_seed(candidate,public['tensor_index']),29)
            indices=np.sort(np.random.default_rng(seed).choice(public['padded_size'],public['sketch_size'],replace=False))
            check[index]=np.array_equal(indices,np.asarray(public['sampled_indices']))
        if not np.array_equal(scored,check):raise ValueError('independent seed score mismatch')
        all_scores.append(scored);active.append(chunk['active_seconds'])
    scores=np.concatenate(all_scores);private=np.load(OUT/'seed_search/private_validator.npz',allow_pickle=False)
    true=int(private['true_base_seed']);true_score=int(scores[true]);rank=1+int(sum(scores>true_score));ties=int(sum(scores==true_score));fps=int(sum(scores))-true_score
    assert search['candidates']==2**20 and not search['full32_executed']
    assert [rank,ties,fps]==[search['true_rank'],search['tied_at_true_score'],search['false_positives']]
    assert bool(true_score==1 and rank==1 and ties==1)==search['recognition_validated']
    assert search['matching_candidates']==np.flatnonzero(scores).tolist()
    assert math.isclose(search['active_seconds'],math.fsum(active),abs_tol=1e-9)
    assert math.isclose(search['full32_worst_seconds'],4096*math.fsum(active),rel_tol=1e-12)
    write(OUT/'analysis.json',dict(summaries=summaries,paired=paired,seed_search=search,
        independent_audit=dict(status='PASS',jobs=198,checkpoint_prediction_arrays=prediction_arrays,upload_assertions=upload_assertions,max_metric_error=max_error)),once=True)
    checklist('independent_audit','complete');checklist('report_seal','in_progress')
    subprocess.run(['git','diff','--check'],cwd=ROOT,check=True)
    for path in list((ROOT/'experiments').glob('priority34e*.py'))+list((ROOT/'tests').glob('test_priority34e*.py')):
        compile(path.read_text(),str(path),'exec')
    subprocess.run([os.sys.executable,'-B','-m','unittest','tests.test_priority34e'],cwd=ROOT,check=True)
    report=ROOT/'reports/priority34e_report.md'
    lines=['# Priority34E — client scaling and v2 seed-space feasibility','',
        'Status: independently verified execution and descriptive analysis complete.','',
        '## More clients: utility','',
        'Fixed total P32/P34A data: PaySim13, IEEE-CIS476, BAF58. K10/K20, 11 paired seeds343000–343010 each, baseline/v1/v2,198 jobs.50 rounds, Adam0.001, focal0.95/2, batch1024, one local epoch. All BN state stays local. Fraud split evenly; natural-category nonfraud55% to primary client,45% across others. Validation-only client thresholds; unweighted mean of K client test metrics per replicate. Clients are not independent replicates.','',
        'Pointwise descriptive Student-t95% CIs (df10) for mean paired DNA−baseline; not simultaneous confidence, NI, equivalence or superiority tests. No incomplete pairs/excluded seeds.','',
        '| Dataset | K | DNA | Endpoint | Mean delta | 95% CI | SD | Median | IQR |','| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for row in summaries:
        lines.append(f"| {row['dataset']} | {row['K']} | {row['method']} | {row['endpoint']} | {row['mean_delta']:.9g} | {row['ci95']} | {row['sd']:.9g} | {row['median_delta']:.9g} | {row['iqr']} |")
    lines+=['','## Seed-space recognition','',
        'Original API metadata includes the derived seed directly; literal observation requires no search. The experiment removes seed but retains sampled_indices. Recognition regenerates the ordered index list for each candidate base seed and scores exact equality. One first-layer128×58 BAF update sketch, ratio0.95/eta0.01; observer receives no raw update/true seed. This is protocol-metadata-assisted recognition, NOT identification from a metadata-free sketch. This simulation lifts locally and has no implemented wire channel whose metadata exposure is proven here.','',
        f"Exhaustive20-bit candidates: {search['candidates']}; true rank: {rank}; tied at true score: {ties}; false positives: {fps}; unique-rank1 validation: {search['recognition_validated']}. Matching candidates: {search['matching_candidates']}.",
        f"Active search time {search['active_seconds']:.6g}s, throughput {search['candidates_per_second']:.6g} candidates/s. Full32 extrapolated worst case {search['full32_worst_seconds']:.6g}s ({search['full32_worst_seconds']/3600:.6g}h); uniform-position expected {search['full32_uniform_expected_seconds']:.6g}s. Chunk-rate sensitivity seconds: {search['full32_chunk_rate_range_seconds']}. No full32 search executed. Setup/downtime/audit excluded from throughput.",
        f"Quantization-grid rounding residual: {search['grid_error']:.9g}; decode diagnostics: {search['decoded_diagnostics']}.",
        'Grid consistency of q is seed independent, and full padded orthogonal lift L2 is seed invariant; neither alone identifies the seed.20-bit uniqueness does not prove full32 uniqueness, cryptographic security or recovery of original records. Benchmark uses only this machine/protocol/observation; it is not an optimized GPU or distributed attack.','',
        '## Reproducibility and independent audit','',
        json.dumps(manifest['environment'],sort_keys=True),
        f'CPU only, torch intra/inter-op1 per process; sequential seed timing before four utility workers. Independently reloaded all198 results, {prediction_arrays} checkpoint prediction arrays and {upload_assertions} BN-firewall uploads. All20-bit candidate scores independently recomputed. Exact completed-job/chunk hash skip on explicit resume; interrupted attempts and failures retained. No numerical/BN clamp, seed replacement or scientific retuning.',
        'All P34E unit tests, Python compilation and git diff --check PASS. Earlier artifacts, datasets, Latex/ and external_defenses/ untouched. Old P34B Goal remains paused. Amendment: protocols/amendments/2026-10-06_priority34e_clients_seed_search.md. Full hashes: artifacts/priority34e/final_manifest.json; verification seal: artifacts/priority34e/COMPLETE.json.','']
    with report.open('x') as f:f.write('\n'.join(lines))
    shutil.copyfile(ROOT.parent/'PROJECT.md',OUT/'PROJECT_before_completion.md')
    checklist('report_seal','complete');progress('VERIFIED_COMPLETE',198,recognition_validated=search['recognition_validated'])
    files={str(path.relative_to(ROOT)):a.p.sha(path) for path in OUT.rglob('*') if path.is_file() and path.name not in ('final_manifest.json','COMPLETE.json','supervisor.lock.json') and not path.name.startswith(('stdout_','stderr_'))}
    files.update(manifest['sources']);files.update(manifest['inputs']);files[str(report.relative_to(ROOT))]=a.p.sha(report)
    write(OUT/'final_manifest.json',dict(files=files,count=len(files),excluded_mutable='live stdout/stderr and supervisor lifetime lock; separately sealed after exit'),once=True)
    for name,digest in files.items():assert a.p.sha(ROOT/name)==digest
    # No submitted training jobs remain; this supervisor is administrative only now.
    write(OUT/'COMPLETE.json',dict(status='verified COMPLETE',report_sha256=a.p.sha(report),manifest_sha256=a.p.sha(OUT/'final_manifest.json'),
        utility_jobs=198,seed_candidates=2**20,recognition_validated=search['recognition_validated'],no_live_training=True,goal_changes=False,at=time.time()),once=True)
    with (ROOT.parent/'PROJECT.md').open('a') as f:f.write('\n### Priority34E completed — 2026-10-06\n\n198 K10/K20 local-BN utility jobs and exhaustive20-bit seed-redacted metadata\nrecognition benchmark independently verified. Descriptive paired11-seed CIs;\nno full32 search or metadata-free security claim. Report FL-DNA/reports/priority34e_report.md.\nPre-note PROJECT snapshot retained. Frozen/earlier sources, datasets, Latex/\nand external_defenses/ preserved; P34B Goal unchanged.\n')

if __name__=='__main__':finish()
