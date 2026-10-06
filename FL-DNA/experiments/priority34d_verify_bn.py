"""Independent BN result integration audit; requires completed science first."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.stats import binomtest
from experiments import priority34d_bn as d
from experiments import priority34d_independent_bn as independent
from experiments.priority34d_independent_metrics import standardized_mse


def close(actual,expected,kind='metric'):
    tolerances={'metric':(1e-6,1e-10),'array':(1e-6,1e-8),
                'capture':(5e-5,5e-6),'calibration':(2e-6,1e-8)}
    rtol,atol=tolerances[kind]
    actual,expected=np.asarray(actual),np.asarray(expected)
    if actual.shape!=expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError('nonfinite/shape mismatch in independent '+kind)
    if not np.allclose(actual,expected,rtol=rtol,atol=atol):
        raise ValueError('independent '+kind+' differs from frozen experiment')


def sign(left,right):
    effect=np.asarray(left)-np.asarray(right)
    wins,losses=int((effect>0).sum()),int((effect<0).sum())
    return dict(wins=wins,losses=losses,ties=len(effect)-wins-losses,
                p=float(binomtest(wins,wins+losses,.5,alternative='greater').pvalue) if wins+losses else 1.)


def verify():
    complete=d.read(d.BASE/'BN_COMPLETE.json')
    assert complete['freeze_sha256']==d.sha(d.FREEZE)
    rows=subprocess.check_output(['ps','-axo','pid,command'],text=True).splitlines()
    if any('python' in row.lower() and 'priority34d_' in row and
           any(flag in row for flag in ('--supervise','--job')) for row in rows):
        raise RuntimeError('wait for all scientific workers to exit before result audit')
    d.verify(); d.c.verify(); assert all(d.c.valid(seed) for seed in d.SEEDS)
    manifest=d.read(d.BASE/'targets.json')
    prepared=d.b.p.OUT/'prepared/baf'
    source=np.load(prepared/'test_source_ids.npy')
    x=np.load(prepared/'test_x.npy',mmap_mode='r')
    training=np.load(prepared/'train_x.npy',mmap_mode='r')
    prior=np.asarray(training,dtype=np.float64).mean(0)
    std=np.asarray(training,dtype=np.float64).std(0); std[std<1e-12]=1
    private=d.read(d.BASE/'private_noise_seeds.json')  # audit only; never emitted
    endpoints=[]; passed_checkpoints=[]
    for seed in d.SEEDS:
        public=d.model(seed)
        linear,bn=public.network[0],public.network[1]
        weight=linear.weight.detach().numpy(); bias=linear.bias.detach().numpy()
        running=bn.running_mean.detach().numpy(); momentum=float(bn.momentum)
        outcome=complete['checkpoints'][str(seed)]
        captures_by_stage={}
        qualified=True
        for stage,size in d.SIZES.items():
            if not qualified:
                break
            captures=[]; recovered=[]
            for index in range(size):
                assert d.valid(seed,stage,index), 'missing validated BN output'
                folder=d.folder(seed,stage,index)
                capture=d.torch.load(folder/'private_capture.pt',map_location='cpu',weights_only=False)
                target=manifest['groups'][str(seed)][stage]
                ids=[int(source[i]) for i in target['indices'][index]]
                assert capture['source_ids']==target['source_ids'][index]==ids
                batch=np.asarray(x[target['indices'][index]],dtype=np.float64)
                close(capture['true_mean'],batch.mean(0),'array')
                # Dense public first-linear identity, independent of model forward.
                expected_delta=momentum*(batch@weight.astype(np.float64).T+bias-running).mean(0)
                close(capture['raw'].numpy(),expected_delta,'capture')
                payloads=d.torch.load(folder/'received_payloads.pt',map_location='cpu',weights_only=False)
                assert d.torch.equal(payloads['unprotected']['q'],capture['raw'])
                arrays=np.load(folder/'recoveries.npz',allow_pickle=False)
                result=d.read(folder/'result.json')
                assert result['source_ids']==ids and result['intra_threads']==result['inter_threads']==1
                metrics={}
                for method,received in payloads.items():
                    guess=independent.decode(weight,bias,running,momentum,received)
                    close(arrays[method],guess,'array')
                    metrics[method]=standardized_mse(guess,capture['true_mean'],std)
                    close(result['metrics'][method],metrics[method])
                assert set(metrics)==set(result['metrics'])==set(arrays.files)
                recovered.append(np.asarray(arrays['unprotected'])); captures.append(capture)
                if stage=='confirmatory':
                    calibration=d.read(d.BASE/str(seed)/'distortion_calibration.json')
                    for method in d.METHODS:
                        expected=d.b.payload(public,capture['raw'],method,index)
                        received=payloads[method]
                        np.testing.assert_array_equal(np.asarray(received['q']),np.asarray(expected['q']))
                        assert received['metadata']==expected['metadata']
                        cal=calibration[method]; raw=capture['raw'].double()
                        clipped=raw*min(1.,cal['clip']/max(float(raw.norm()),1e-12))
                        generator=d.torch.Generator().manual_seed(private[str(seed)][stage][method][index])
                        q=(clipped+d.torch.randn(raw.shape,generator=generator,dtype=d.torch.float64)*cal['clip']*cal['sigma']).to(capture['raw'].dtype)
                        assert d.torch.equal(payloads['dp_'+method]['q'],q), 'DP payload not bit-exact'
                    endpoints.append(dict(checkpoint=seed,index=index,source_ids=ids,metrics=metrics))
            captures_by_stage[stage]=captures
            if stage in ('n8','n24'):
                stored=d.read(d.BASE/str(seed)/(stage+'_gate.json'))
                metrics=[dict(recovered_mse=standardized_mse(recovered[i],cap['true_mean'],std),
                              prior_mse=standardized_mse(prior,cap['true_mean'],std),
                              decoy_mse=standardized_mse(recovered[(i+1)%size],cap['true_mean'],std))
                         for i,cap in enumerate(captures)]
                for index,row in enumerate(metrics):
                    for key,value in row.items():
                        close(stored['rows'][index][key],value)
                tests={name:sign([r[name+'_mse'] for r in metrics],[r['recovered_mse'] for r in metrics])
                       for name in ('prior','decoy')}
                assert tests==stored['tests'], 'qualification sign counts/p differ'
                qualified=all(test['p']<.05 for test in tests.values())
                assert qualified==stored['passed']
            elif stage=='development':
                stored=d.read(d.BASE/str(seed)/'distortion_calibration.json')
                clip=1.01*max(float(np.linalg.norm(cap['raw'].numpy().astype(np.float64))) for cap in captures)
                for method in d.METHODS:
                    differences=[]; norms=[]
                    for index,cap in enumerate(captures):
                        received=d.b.payload(public,cap['raw'],method,index)
                        decoded=independent.defender_decode_for_distortion(received)
                        differences.append(float(np.linalg.norm(decoded-cap['raw'].numpy())))
                        generator=d.torch.Generator().manual_seed(private[str(seed)][stage][method][index])
                        norms.append(float(d.torch.randn(cap['raw'].shape,generator=generator,dtype=d.torch.float64).norm()))
                    sigma=float(np.median(differences)/(clip*np.median(norms)))
                    close(stored[method]['clip'],clip,'calibration')
                    close(stored[method]['sigma'],sigma,'calibration')
                    close(stored[method]['distortions'],differences,'calibration')
                    close(stored[method]['unit_noise_norms'],norms,'calibration')
        if qualified:
            assert outcome['status']=='CONFIRMATORY_COMPLETE'
            assert len(captures_by_stage['confirmatory'])==39
            passed_checkpoints.append(seed)
        else:
            assert outcome['status']=='NOT_ASSESSABLE'
    result=dict(status='PASS_BN_STAGE_ONLY',priority34d_complete=False,at=time.time(),
                freeze_sha256=d.sha(d.FREEZE),qualified_checkpoints=passed_checkpoints,
                endpoints=endpoints,independent_public_decoder=True,DP_payloads_bit_exact=True)
    d.write(d.OUT/'audits/independent_bn_results.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--verify',action='store_true')
    args=parser.parse_args()
    if not args.verify:
        parser.error('choose --verify, only after completed science')
    try:
        result=verify()
        print(json.dumps(dict(status=result['status'],qualified_checkpoints=result['qualified_checkpoints'],
                              endpoints=len(result['endpoints']))))
    except Exception as error:
        d.write(d.OUT/('audits/bn_verifier_failure_'+str(time.time_ns())+'.json'),
                dict(error=repr(error),at=time.time(),priority34d_complete=False))
        raise
